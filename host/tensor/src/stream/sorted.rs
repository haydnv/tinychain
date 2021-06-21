use std::convert::TryFrom;

use afarray::{Array, ArrayExt, ArrayInstance};
use arrayfire as af;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::UIntType;

use crate::dense::{BlockListFile, PER_BLOCK};
use crate::{Coord, Shape, TensorAccess, TensorType};

use super::{ReadValueAt, ValueReader};

pub async fn sorted_coords<FD, FS, D, T, C>(
    txn: &T,
    shape: &Shape,
    coords: C,
) -> TCResult<impl Stream<Item = TCResult<Coord>> + Unpin>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    C: Stream<Item = TCResult<Coord>> + Send,
    D::FileClass: From<TensorType>,
{
    let txn_id = *txn.id();
    let file: FD = txn
        .context()
        .create_file_tmp(txn_id, TensorType::Dense)
        .await?;

    let offsets = sort_coords::<FD, FS, D, T, _>(file, txn_id, coords, shape).await?;
    let coords = offsets_to_coords(shape, offsets.into_stream(txn_id));
    Ok(coords)
}

pub async fn sorted_values<'a, FD, FS, T, D, A, C>(
    txn: T,
    source: A,
    coords: C,
) -> TCResult<ValueReader<'a, impl Stream<Item = TCResult<Coord>>, D, T, A>>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    C: Stream<Item = TCResult<Coord>> + Send + 'a,
    A: TensorAccess + ReadValueAt<D> + 'a,
    D::FileClass: From<TensorType>,
{
    let sorted_coords = sorted_coords::<FD, FS, D, T, C>(&txn, source.shape(), coords).await?;
    Ok(ValueReader::new(sorted_coords, txn, source))
}

async fn sort_coords<FD, FS, D, T, S>(
    file: FD,
    txn_id: TxnId,
    coords: S,
    shape: &Shape,
) -> TCResult<BlockListFile<FD, FS, D, T>>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    S: Stream<Item = TCResult<Coord>> + Send,
{
    let blocks = coords_to_offsets(shape, coords).map_ok(|block| ArrayExt::from(block).into());

    let block_list =
        BlockListFile::from_blocks(file, txn_id, None, UIntType::U64.into(), Box::pin(blocks))
            .await?;

    block_list.merge_sort(txn_id).await?;
    Ok(block_list)
}

fn coords_to_offsets<S: Stream<Item = TCResult<Coord>>>(
    shape: &Shape,
    coords: S,
) -> impl Stream<Item = TCResult<af::Array<u64>>> {
    let ndim = shape.len() as u64;
    let coord_bounds = coord_bounds(shape);
    let af_coord_bounds: af::Array<u64> =
        af::Array::new(&coord_bounds, af::Dim4::new(&[ndim, 1, 1, 1]));

    coords
        .chunks(PER_BLOCK)
        .map(|block| block.into_iter().collect::<TCResult<Vec<Coord>>>())
        .map_ok(move |block| {
            let num_coords = block.len();
            let block = block.into_iter().flatten().collect::<Vec<u64>>();
            af::Array::new(&block, af::Dim4::new(&[ndim, num_coords as u64, 1, 1]))
        })
        .map_ok(move |block| {
            let offsets = af::mul(&block, &af_coord_bounds, true);
            af::sum(&offsets, 0)
        })
        .map_ok(|block| af::moddims(&block, af::Dim4::new(&[block.elements() as u64, 1, 1, 1])))
}

fn offsets_to_coords<'a, S: Stream<Item = TCResult<Array>> + Unpin + 'a>(
    shape: &Shape,
    offsets: S,
) -> impl Stream<Item = TCResult<Coord>> + Unpin + 'a {
    let ndim = shape.len() as u64;
    let coord_bounds = coord_bounds(shape);
    let af_coord_bounds: af::Array<u64> =
        af::Array::new(&coord_bounds, af::Dim4::new(&[1, ndim, 1, 1]));
    let af_shape: af::Array<u64> = af::Array::new(&shape.to_vec(), af::Dim4::new(&[1, ndim, 1, 1]));
    let ndim = shape.len();

    offsets
        .map_ok(move |block| {
            let offsets = af::div(block.type_cast::<u64>().af(), &af_coord_bounds, true);
            af::modulo(&offsets, &af_shape, true)
        })
        .map_ok(|coord_block| {
            let mut coords = vec![0u64; coord_block.elements()];
            af::transpose(&coord_block, false).host(&mut coords);
            coords
        })
        .map_ok(move |coords| {
            stream::iter(coords.into_iter())
                .chunks(ndim)
                .map(TCResult::<Coord>::Ok)
        })
        .try_flatten()
}

#[inline]
pub fn coord_bounds(shape: &Shape) -> Coord {
    (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect()
}

pub fn block_offsets(
    af_indices: &af::Array<u64>,
    af_offsets: &af::Array<u64>,
    start: f64,
    block_id: u64,
) -> (af::Array<u64>, f64) {
    assert_eq!(af_indices.elements(), af_offsets.elements());

    let num_to_update = af::sum_all(&af::eq(
        af_indices,
        &af::constant(block_id, af::Dim4::new(&[1, 1, 1, 1])),
        true,
    ))
    .0;

    if num_to_update == 0 {
        return (af::Array::new_empty(af::Dim4::default()), start);
    }

    debug_assert!((start as usize + num_to_update as usize) <= af_offsets.elements());

    let num_to_update = num_to_update as f64;
    let block_offsets = af::index(
        af_offsets,
        &[af::Seq::new(start, (start + num_to_update) - 1f64, 1f64)],
    );

    (block_offsets, (start + num_to_update))
}

pub fn coord_block<I: Iterator<Item = Coord>>(
    coords: I,
    coord_bounds: &[u64],
    per_block: usize,
    ndim: usize,
    num_coords: u64,
) -> (Coord, af::Array<u64>, af::Array<u64>) {
    let coords: Vec<u64> = coords.flatten().collect();
    assert!(coords.len() > 0);
    assert!(ndim > 0);

    let af_per_block = af::constant(per_block as u64, af::Dim4::new(&[1, 1, 1, 1]));
    let af_coord_bounds = af::Array::new(coord_bounds, af::Dim4::new(&[ndim as u64, 1, 1, 1]));

    let af_coords = af::Array::new(
        &coords,
        af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]),
    );
    let af_coords = af::mul(&af_coords, &af_coord_bounds, true);
    let af_coords = af::sum(&af_coords, 0);

    let af_offsets = af::modulo(&af_coords, &af_per_block, true);
    let af_indices = af_coords / af_per_block;

    let af_block_ids = af::set_unique(&af_indices, true);
    let mut block_ids = vec![0u64; af_block_ids.elements()];
    af_block_ids.host(&mut block_ids);
    (block_ids, af_indices, af_offsets)
}
