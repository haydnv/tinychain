use std::convert::TryInto;
use std::iter::{self, FromIterator};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;
use num::integer::div_ceil;

use crate::block::BlockId;
use crate::block::File;
use crate::class::Instance;
use crate::error;
use crate::general::{TCBoxTryFuture, TCResult, TCTryStream};
use crate::scalar::number::*;
use crate::scalar::Value;
use crate::transaction::{Transact, Txn, TxnId};

use super::super::bounds::*;
use super::super::stream::{block_offsets, coord_block, coord_bounds, ReadValueAt};
use super::super::TensorAccessor;

use super::array::Array;
use super::{BlockListSlice, DenseAccess, DenseAccessor};
use crate::collection::tensor::stream::Read;

pub const PER_BLOCK: usize = 131_072; // = 1 mibibyte / 64 bits

#[derive(Clone)]
pub struct BlockListFile {
    file: Arc<File<Array>>,
    dtype: NumberType,
    shape: Shape,
}

impl BlockListFile {
    pub async fn constant(txn: &Txn, shape: Shape, value: Number) -> TCResult<BlockListFile> {
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
        txn: &Txn,
        shape: Shape,
        dtype: NumberType,
        blocks: S,
    ) -> TCResult<BlockListFile> {
        let file = txn.context().await?;

        blocks
            .enumerate()
            .map(|(i, r)| r.map(|block| (BlockId::from(i), block)))
            .map_ok(|(id, block)| file.clone().create_block(txn.id().clone(), id, block))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(BlockListFile { dtype, shape, file })
    }

    pub async fn from_values<S: Stream<Item = Number> + Send + Unpin>(
        txn: &Txn,
        shape: Shape,
        dtype: NumberType,
        values: S,
    ) -> TCResult<BlockListFile> {
        let file = txn.context().await?;

        let mut i = 0u64;
        let mut values = values.chunks(PER_BLOCK);
        while let Some(chunk) = values.next().await {
            let block_id = BlockId::from(i);
            let block = Array::cast_from_values(chunk, dtype)?;
            file.clone()
                .create_block(txn.id().clone(), block_id, block)
                .await?;

            i += 1;
        }

        Ok(BlockListFile { dtype, shape, file })
    }

    pub fn into_stream(self, txn_id: TxnId) -> impl Stream<Item = TCResult<Array>> + Unpin {
        // TODO: add a method in File to delete the block and return its contents

        let num_blocks = div_ceil(self.size(), PER_BLOCK as u64);
        let blocks = stream::iter((0..num_blocks).into_iter().map(BlockId::from))
            .then(move |block_id| self.file.clone().get_block_owned(txn_id, block_id))
            .map_ok(|block| block.deref().clone());

        Box::pin(blocks)
    }

    pub async fn merge_sort(&self, txn_id: &TxnId) -> TCResult<()> {
        let num_blocks = div_ceil(self.size(), PER_BLOCK as u64);
        if num_blocks == 1 {
            let block_id = BlockId::from(0u64);
            let mut block = self
                .file
                .get_block(txn_id, block_id)
                .await?
                .upgrade()
                .await?;

            block.sort();
            return Ok(());
        }

        for block_id in 0..(num_blocks - 1) {
            let next_block_id = BlockId::from(block_id + 1);
            let block_id = BlockId::from(block_id);

            let left = self.file.get_block(txn_id, block_id);
            let right = self.file.get_block(txn_id, next_block_id);
            let (left, right) = try_join!(left, right)?;
            let (mut left, mut right) = try_join!(left.upgrade(), right.upgrade())?;

            let mut block = Array::concatenate(&left, &right)?;
            block.sort();

            let (left_sorted, right_sorted) = block.split(PER_BLOCK)?;
            *left = left_sorted;
            *right = right_sorted;
        }

        Ok(())
    }
}

impl TensorAccessor for BlockListFile {
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
impl DenseAccess for BlockListFile {
    type Slice = BlockListSlice<Self>;

    fn accessor(self) -> DenseAccessor {
        DenseAccessor::File(self)
    }

    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let file = &self.file;
            let block_stream = Box::pin(
                stream::iter(0..(div_ceil(self.size(), PER_BLOCK as u64)))
                    .map(BlockId::from)
                    .then(move |block_id| file.get_block(txn.id(), block_id)),
            );

            let block_stream =
                block_stream.and_then(|block| future::ready(Ok(block.deref().clone())));

            let block_stream: TCTryStream<'a, Array> = Box::pin(block_stream);
            Ok(block_stream)
        })
    }

    async fn slice(&self, _txn: &Txn, bounds: Bounds) -> TCResult<Self::Slice> {
        BlockListSlice::new(self.clone(), bounds)
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        debug!("BlockListFile::write_value {} at {}", value, bounds);

        if !self.shape().contains_bounds(&bounds) {
            return Err(error::bad_request("Bounds out of bounds", bounds));
        } else if bounds.len() == self.ndim() && bounds.is_coord() {
            return self.write_value_at(txn_id, bounds.try_into()?, value).await;
        }

        let bounds = self.shape().slice_bounds(bounds);
        let coord_bounds = coord_bounds(self.shape());

        stream::iter(bounds.affected())
            .chunks(PER_BLOCK)
            .map(|coords| {
                let ndim = coords[0].len();
                let num_coords = coords.len() as u64;
                let (block_ids, af_indices, af_offsets) = coord_block(
                    coords.into_iter(),
                    &coord_bounds,
                    PER_BLOCK,
                    ndim,
                    num_coords,
                );

                let file = &self.file;
                let value = value.clone();
                let txn_id = txn_id;

                Ok(async move {
                    let mut start = 0.0f64;
                    for block_id in block_ids {
                        let value = value.clone();
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, start, block_id);

                        let block_id = BlockId::from(block_id);
                        let mut block = file.get_block(&txn_id, block_id).await?.upgrade().await?;

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

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(error::bad_request(
                    "Invalid coordinate",
                    format!("[{:?}]", coord),
                ));
            }

            let value = value.into_type(self.dtype);

            let offset: u64 = coord_bounds(self.shape())
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();

            let block_id = BlockId::from(offset / PER_BLOCK as u64);

            let mut block = self
                .file
                .get_block(&txn_id, block_id)
                .await?
                .upgrade()
                .await?;

            block
                .deref_mut()
                .set_value((offset % PER_BLOCK as u64) as usize, value)
        })
    }
}

impl ReadValueAt for BlockListFile {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a> {
        Box::pin(async move {
            debug!(
                "read value at {:?} from BlockListFile with shape {}",
                coord,
                self.shape()
            );

            if !self.shape().contains_coord(&coord) {
                return Err(error::bad_request(
                    "Coordinate is out of bounds",
                    Value::from_iter(coord),
                ));
            }

            let offset: u64 = coord_bounds(self.shape())
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();
            debug!("coord {:?} is offset {}", coord, offset);

            let block_id = BlockId::from(offset / PER_BLOCK as u64);
            let block = self.file.get_block(txn.id(), block_id).await?;

            debug!(
                "read offset {} from block of length {}",
                (offset % PER_BLOCK as u64),
                block.len()
            );
            let value = block.get_value((offset % PER_BLOCK as u64) as usize);

            Ok((coord, value))
        })
    }
}

#[async_trait]
impl Transact for BlockListFile {
    async fn commit(&self, txn_id: &TxnId) {
        self.file.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.file.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.file.finalize(txn_id).await
    }
}
