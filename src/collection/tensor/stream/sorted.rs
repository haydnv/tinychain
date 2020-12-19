use std::convert::TryFrom;

use arrayfire as af;
use futures::future;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};

use crate::general::TCResult;
use crate::scalar::UIntType;
use crate::transaction::Txn;

use super::super::dense::{PER_BLOCK, Array, BlockListFile};
use super::super::bounds::Shape;
use super::super::TensorAccessor;
use super::{ReadValueAt, ValueReader};

pub async fn sorted_coords<C: Stream<Item = TCResult<Vec<u64>>> + Send>(txn: &Txn, shape: &Shape, coords: C, num_coords: u64) -> TCResult<impl Stream<Item = TCResult<Vec<u64>>> + Unpin> {
    let subcontext = txn.subcontext_tmp().await?;
    let offsets = sort_coords(subcontext, coords, num_coords, shape).await?;
    let coords = offsets_to_coords(shape, offsets.into_stream(*txn.id()));
    Ok(coords)
}

pub async fn sorted_values<'a, T: TensorAccessor + ReadValueAt + 'a, C: Stream<Item = TCResult<Vec<u64>>> + Send + 'a>(txn: &'a Txn, source: &'a T, coords: C, num_coords: u64) -> TCResult<ValueReader<'a, impl Stream<Item = TCResult<Vec<u64>>>, T>> {
    let sorted_coords = sorted_coords(txn, source.shape(), coords, num_coords).await?;
    Ok(ValueReader::new(sorted_coords, txn, source))
}

async fn sort_coords<S: Stream<Item = TCResult<Vec<u64>>> + Send>(
    txn: Txn,
    coords: S,
    num_coords: u64,
    shape: &Shape,
) -> TCResult<BlockListFile> {
    let blocks =
        coords_to_offsets(shape, coords).and_then(|block| future::ready(Array::try_from(block)));

    let block_list = BlockListFile::from_blocks(
        &txn,
        Shape::from(vec![num_coords]),
        UIntType::U64.into(),
        Box::pin(blocks),
    ).await?;

    block_list.merge_sort(txn.id()).await?;
    Ok(block_list)
}

fn coords_to_offsets<S: Stream<Item = TCResult<Vec<u64>>>>(
    shape: &Shape,
    coords: S,
) -> impl Stream<Item = TCResult<af::Array<u64>>> {
    let ndim = shape.len() as u64;
    let coord_bounds = coord_bounds(shape);
    let af_coord_bounds: af::Array<u64> =
        af::Array::new(&coord_bounds, af::Dim4::new(&[ndim, 1, 1, 1]));

    coords
        .chunks(PER_BLOCK)
        .map(|block| block.into_iter().collect::<TCResult<Vec<Vec<u64>>>>())
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

fn offsets_to_coords<'a, S: Stream<Item = TCResult<Array>> + Unpin + 'a>(shape: &Shape, offsets: S) -> impl Stream<Item = TCResult<Vec<u64>>> + Unpin + 'a {
    let ndim = shape.len() as u64;
    let coord_bounds = coord_bounds(shape);
    let af_coord_bounds: af::Array<u64> =
        af::Array::new(&coord_bounds, af::Dim4::new(&[1, ndim, 1, 1]));
    let af_shape: af::Array<u64> = af::Array::new(&shape.to_vec(), af::Dim4::new(&[1, ndim, 1, 1]));
    let ndim = shape.len();

    offsets.map_ok(|block| block.into_af_array::<u64>())
        .map_ok(move |block| {
            let offsets = af::div(&block, &af_coord_bounds, true);
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
                .map(TCResult::<Vec<u64>>::Ok)
        })
        .try_flatten()
}

pub fn coord_bounds(shape: &Shape) -> Vec<u64> {
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

    if num_to_update == 0f64 {
        return (af::Array::new_empty(af::Dim4::default()), start);
    }

    assert!((start + num_to_update) as usize <= af_offsets.elements());

    let block_offsets = af::index(
        af_offsets,
        &[af::Seq::new(start, (start + num_to_update) - 1f64, 1f64)],
    );

    (block_offsets, (start + num_to_update))
}

pub fn coord_block<I: Iterator<Item = Vec<u64>>>(
    coords: I,
    coord_bounds: &[u64],
    per_block: usize,
    ndim: usize,
    num_coords: u64,
) -> (Vec<u64>, af::Array<u64>, af::Array<u64>) {
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

#[cfg(test)]
mod tests {
    use crate::collection::tensor::Bounds;
    use super::*;

    #[test]
    fn test_coord_block() {
        let shape = Shape::from(vec![2, 3, 4]);
        let bounds = coord_bounds(&shape);
        let coords: Vec<Vec<u64>> = Bounds::all(&shape).affected().collect();

        let num_coords = coords.len() as u64;
        let (block_ids, af_indices, af_offsets) = coord_block(
            coords.into_iter(),
            &bounds,
            PER_BLOCK,
            shape.len(),
            num_coords,
        );

        let mut indices = vec![0u64; af_indices.elements()];
        af_indices.host(&mut indices);

        let mut offsets = vec![0u64; af_offsets.elements()];
        af_offsets.host(&mut offsets);

        assert_eq!(block_ids, vec![0]);
        assert_eq!(indices, vec![0; 24]);
        assert_eq!(offsets, (0..24).collect::<Vec<u64>>());
    }

    #[test]
    fn test_block_offsets() {
        let shape = Shape::from(vec![2, 3, 4]);
        let bounds = coord_bounds(&shape);
        let coords: Vec<Vec<u64>> = Bounds::all(&shape).affected().collect();

        let num_coords = coords.len() as u64;
        let (_, af_indices, af_offsets) = coord_block(
            coords.into_iter(),
            &bounds,
            PER_BLOCK,
            shape.len(),
            num_coords,
        );

        let (af_block_offsets, new_start) = block_offsets(&af_indices, &af_offsets, 0f64, 0u64);
        let mut block_offsets = vec![0u64; af_block_offsets.elements()];
        af_block_offsets.host(&mut block_offsets);

        assert_eq!(new_start, 24f64);
        assert_eq!(block_offsets, (0..24).collect::<Vec<u64>>());
    }
}
