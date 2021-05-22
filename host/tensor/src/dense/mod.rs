use std::marker::PhantomData;

use afarray::Array;
use arrayfire as af;
use async_trait::async_trait;
use futures::stream::{StreamExt, TryStreamExt};
use number_general::{Number, NumberType};

use tc_error::*;
use tc_transact::fs::{BlockData, Dir, File};
use tc_transact::{Transaction, TxnId};
use tcgeneric::{TCBoxTryFuture, TCTryStream};

use super::{Bounds, Coord, Read, ReadValueAt, Shape, TensorAccess};

pub use file::BlockListFile;

mod file;

// = 1 mibibyte / 64 bits (must be the same as Array::max_size())
const PER_BLOCK: usize = 131_072;

#[async_trait]
pub trait DenseAccess<F: File<Array>, D: Dir, T: Transaction<D>>:
    ReadValueAt<D, T> + TensorAccess + Send + Sync + 'static
{
    fn accessor(self) -> DenseAccessor<F, D, T>;

    fn block_stream<'a>(&'a self, txn: &'a T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let blocks = self
                .value_stream(txn)
                .await?
                .chunks(Array::max_size() as usize)
                .map(|values| values.into_iter().collect::<TCResult<Vec<Number>>>())
                .map_ok(Array::from);

            let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn value_stream<'a>(&'a self, txn: &'a T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let values = self.block_stream(txn).await?;

            let values = values
                .map_ok(|array| array.to_vec())
                .map_ok(|values| {
                    values
                        .into_iter()
                        .map(Ok)
                        .collect::<Vec<TCResult<Number>>>()
                })
                .map_ok(futures::stream::iter)
                .try_flatten();

            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()>;

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()>;
}

#[derive(Clone)]
pub enum DenseAccessor<F, D, T> {
    File(BlockListFile<F, D, T>),
}

impl<F: Send, D: Send, T: Send> TensorAccess for DenseAccessor<F, D, T> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::File(file) => file.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::File(file) => file.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::File(file) => file.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::File(file) => file.size(),
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> DenseAccess<F, D, T> for DenseAccessor<F, D, T> {
    fn accessor(self) -> Self {
        self
    }

    fn block_stream<'a>(&'a self, txn: &'a T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        match self {
            Self::File(file) => file.block_stream(txn),
        }
    }

    fn value_stream<'a>(&'a self, txn: &'a T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        match self {
            Self::File(file) => file.value_stream(txn),
        }
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        match self {
            Self::File(file) => file.write_value(txn_id, bounds, number).await,
        }
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        match self {
            Self::File(file) => file.write_value_at(txn_id, coord, value),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> ReadValueAt<D, T> for DenseAccessor<F, D, T> {
    fn read_value_at<'a>(&'a self, txn: &'a T, coord: Coord) -> Read<'a> {
        match self {
            Self::File(file) => file.read_value_at(txn, coord),
        }
    }
}

impl<F, D, T> From<BlockListFile<F, D, T>> for DenseAccessor<F, D, T> {
    fn from(file: BlockListFile<F, D, T>) -> Self {
        Self::File(file)
    }
}

#[derive(Clone)]
pub struct DenseTensor<F, B> {
    blocks: B,
    file: PhantomData<F>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: Clone + DenseAccess<F, D, T>> ReadValueAt<D, T>
    for DenseTensor<F, B>
{
    fn read_value_at<'a>(&'a self, txn: &'a T, coord: Coord) -> Read<'a> {
        self.blocks.read_value_at(txn, coord)
    }
}

impl<F: Send, B: TensorAccess> TensorAccess for DenseTensor<F, B> {
    fn dtype(&self) -> NumberType {
        self.blocks.dtype()
    }

    fn ndim(&self) -> usize {
        self.blocks.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.blocks.shape()
    }

    fn size(&self) -> u64 {
        self.blocks.size()
    }
}

fn block_offsets(
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

fn coord_block<I: Iterator<Item = Coord>>(
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
