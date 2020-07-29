use std::convert::TryInto;
use std::iter;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use arrayfire as af;
use async_trait::async_trait;
use futures::future::{self, BoxFuture};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use itertools::Itertools;

use crate::error;
use crate::state::file::block::BlockId;
use crate::state::file::File;
use crate::transaction::{Txn, TxnId};
use crate::value::class::{Impl, NumberType};
use crate::value::{Number, TCResult, TCStream};

use super::array::Array;
use super::bounds::{Bounds, Shape};
use super::stream::ValueStream;
use super::*;

const PER_BLOCK: usize = 131_072; // = 1 mibibyte / 64 bits

#[async_trait]
trait BlockList: TensorView {
    fn block_stream(self, txn_id: TxnId) -> TCStream<TCResult<Array>>;

    fn value_stream(self, txn_id: TxnId, bounds: Bounds) -> TCStream<TCResult<Number>>;

    async fn write_value(self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()>;

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> BoxFuture<'a, TCResult<()>>;
}

#[derive(Clone)]
pub struct BlockListFile {
    file: Arc<File<Array>>,
    dtype: NumberType,
    shape: Shape,
    coord_bounds: Vec<u64>,
}

impl BlockListFile {
    pub async fn constant(txn: Arc<Txn>, shape: Shape, value: Number) -> TCResult<BlockListFile> {
        let size = shape.size();

        let value_clone = value.clone();
        let blocks = (0..(size / PER_BLOCK as u64))
            .map(move |_| Ok(Array::constant(value_clone.clone(), PER_BLOCK)));
        let trailing_len = (size % (PER_BLOCK as u64)) as usize;
        if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(Ok(Array::constant(value.clone(), trailing_len))));
            BlockListFile::from_blocks(txn, shape, value.class(), stream::iter(blocks)).await
        } else {
            BlockListFile::from_blocks(txn, shape, value.class(), stream::iter(blocks)).await
        }
    }

    pub async fn from_blocks<S: Stream<Item = TCResult<Array>> + Send + Unpin>(
        txn: Arc<Txn>,
        shape: Shape,
        dtype: NumberType,
        blocks: S,
    ) -> TCResult<BlockListFile> {
        let file = txn
            .context()
            .create_tensor(txn.id().clone(), "block_tensor".parse()?)
            .await?;

        blocks
            .enumerate()
            .map(|(i, r)| r.map(|block| (BlockId::from(i), block)))
            .map_ok(|(id, block)| file.create_block(txn.id().clone(), id, block))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        let coord_bounds = (0..shape.len())
            .map(|axis| shape[axis + 1..].iter().product())
            .collect();

        Ok(BlockListFile {
            dtype,
            shape,
            file,
            coord_bounds,
        })
    }
}

impl TensorView for BlockListFile {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.shape.size()
    }
}

#[async_trait]
impl BlockList for BlockListFile {
    fn block_stream(self, txn_id: TxnId) -> TCStream<TCResult<Array>> {
        let blocks = Box::pin(
            stream::iter(0..(self.size() / PER_BLOCK as u64))
                .map(BlockId::from)
                .then(move |block_id| self.file.clone().get_block_owned(txn_id.clone(), block_id)),
        );

        Box::pin(blocks.and_then(|block| future::ready(Ok(block.deref().clone()))))
    }

    fn value_stream(self, txn_id: TxnId, bounds: Bounds) -> TCStream<TCResult<Number>> {
        if bounds == self.shape().all() {
            return Box::pin(ValueStream::new(self.block_stream(txn_id)));
        }

        assert!(self.shape().contains_bounds(&bounds));

        let ndim = bounds.ndim();

        let coord_bounds = af::Array::new(
            &self.coord_bounds,
            af::Dim4::new(&[self.ndim() as u64, 1, 1, 1]),
        );

        let selected = stream::iter(bounds.affected())
            .chunks(PER_BLOCK)
            .then(move |coords| {
                let (block_ids, af_indices, af_offsets, num_coords) =
                    coord_block(coords.into_iter(), &coord_bounds, PER_BLOCK, ndim);

                let this = self.clone();
                let txn_id = txn_id.clone();

                async move {
                    let mut start = 0.0f64;
                    let mut values = vec![];
                    for block_id in block_ids {
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, num_coords, start, block_id);

                        match this.file.clone().get_block(&txn_id, block_id.into()).await {
                            Ok(block) => {
                                let array: &Array = block.deref().try_into().unwrap();
                                values.extend(array.get(block_offsets));
                            }
                            Err(cause) => return stream::iter(vec![Err(cause)]),
                        }

                        start = new_start;
                    }

                    let values: Vec<TCResult<Number>> = values.drain(..).map(Ok).collect();
                    stream::iter(values)
                }
            });

        Box::pin(Box::pin(selected).flatten())
    }

    async fn write_value(self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        if !self.shape().contains_bounds(&bounds) {
            return Err(error::bad_request("Bounds out of bounds", bounds));
        }

        let ndim = bounds.ndim();

        let coord_bounds = af::Array::new(
            &self.coord_bounds,
            af::Dim4::new(&[self.ndim() as u64, 1, 1, 1]),
        );

        stream::iter(bounds.affected())
            .chunks(PER_BLOCK)
            .map(|coords| {
                let (block_ids, af_indices, af_offsets, num_coords) =
                    coord_block(coords.into_iter(), &coord_bounds, PER_BLOCK, ndim);

                let this = self.clone();
                let value = value.clone();
                let txn_id = txn_id.clone();

                Ok(async move {
                    let mut start = 0.0f64;
                    for block_id in block_ids {
                        let value = value.clone();
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, num_coords, start, block_id);

                        let mut block = this
                            .file
                            .get_block(&txn_id, block_id.into())
                            .await?
                            .upgrade()
                            .await?;
                        let value = Array::constant(value, (new_start - start) as usize);
                        block.deref_mut().set(block_offsets, &value)?;
                        start = new_start;
                    }

                    Ok(())
                })
            })
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(error::bad_request(
                    "Invalid coordinate",
                    format!("[{}]", coord.iter().map(|x| x.to_string()).join(", ")),
                ));
            } else if value.class() != self.dtype() {
                return Err(error::bad_request(
                    "Wrong class for tensor value",
                    value.class(),
                ));
            }

            let offset: u64 = self
                .coord_bounds
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();
            let block_id: u64 = offset / PER_BLOCK as u64;
            let mut block = self
                .file
                .get_block(&txn_id, block_id.into())
                .await?
                .upgrade()
                .await?;
            block
                .deref_mut()
                .set_value((offset % PER_BLOCK as u64) as usize, value)
        })
    }
}

pub struct DenseTensor {
    blocks: Box<dyn BlockList>,
}

impl TensorView for DenseTensor {
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

impl TensorTransform for DenseTensor {
    fn as_type(&self, _dtype: NumberType) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn broadcast(&self, _shape: Shape) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn expand_dims(&self, _axis: usize) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn slice(&self, _bounds: Bounds) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn transpose(&self, _permutation: Option<Vec<usize>>) -> TCResult<Self> {
        Err(error::not_implemented())
    }
}

fn block_offsets(
    af_indices: &af::Array<u64>,
    af_offsets: &af::Array<u64>,
    num_coords: u64,
    start: f64,
    block_id: u64,
) -> (af::Array<u64>, f64) {
    let num_to_update = af::sum_all(&af::eq(
        af_indices,
        &af::constant(block_id, af_indices.dims()),
        false,
    ))
    .0;
    let block_offsets = af::index(
        af_offsets,
        &[
            af::Seq::new(block_id as f64, block_id as f64, 1.0f64),
            af::Seq::new(start, (start + num_to_update) - 1.0f64, 1.0f64),
        ],
    );
    let block_offsets = af::moddims(&block_offsets, af::Dim4::new(&[num_coords as u64, 1, 1, 1]));

    (block_offsets, (start + num_to_update))
}

fn coord_block<I: Iterator<Item = Vec<u64>>>(
    coords: I,
    coord_bounds: &af::Array<u64>,
    per_block: usize,
    ndim: usize,
) -> (Vec<u64>, af::Array<u64>, af::Array<u64>, u64) {
    let coords: Vec<u64> = coords.flatten().collect();
    let num_coords = coords.len() / ndim;
    let af_coords_dim = af::Dim4::new(&[num_coords as u64, ndim as u64, 1, 1]);
    let af_coords = af::Array::new(&coords, af_coords_dim) * af::tile(coord_bounds, af_coords_dim);
    let af_coords = af::sum(&af_coords, 1);
    let af_per_block = af::constant(
        per_block as u64,
        af::Dim4::new(&[1, num_coords as u64, 1, 1]),
    );
    let af_offsets = af_coords.copy() % af_per_block.copy();
    let af_indices = af_coords / af_per_block;
    let af_block_ids = af::set_unique(&af_indices, true);

    let mut block_ids: Vec<u64> = Vec::with_capacity(af_block_ids.elements());
    af_block_ids.host(&mut block_ids);
    (block_ids, af_indices, af_offsets, num_coords as u64)
}
