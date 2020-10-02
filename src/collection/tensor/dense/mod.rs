use std::convert::{TryFrom, TryInto};
use std::iter;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use arrayfire as af;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use itertools::Itertools;
use num::integer::div_ceil;

use crate::block::BlockId;
use crate::block::File;
use crate::class::{Instance, TCBoxTryFuture, TCResult, TCStream, TCTryStream};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{AxisBounds, Bounds, Shape};
use super::class::TensorInstance;
use super::*;

pub mod array;

use array::Array;

const ERR_CORRUPT: &str = "DenseTensor corrupted! Please file a bug report.";
const PER_BLOCK: usize = 131_072; // = 1 mibibyte / 64 bits

trait BlockList: TensorInstance + Transact + 'static {
    fn block_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let dtype = self.dtype();
            let blocks = self
                .value_stream(txn)
                .await?
                .chunks(PER_BLOCK)
                .map(|values| values.into_iter().collect::<TCResult<Vec<Number>>>())
                .and_then(move |values| future::ready(Array::try_from_values(values, dtype)));

            let blocks: TCTryStream<Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn value_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let values = self
                .block_stream(txn)
                .await?
                .and_then(|array| future::ready(Ok(array.into_values())))
                .map_ok(|mut values| values.drain(..).map(Ok).collect::<Vec<TCResult<Number>>>())
                .map_ok(stream::iter)
                .try_flatten();

            let values: TCTryStream<Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>>;

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number>;

    fn read_value_at_owned<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        coord: Vec<u64>,
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move { self.read_value_at(&txn, &coord).await })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCBoxTryFuture<'a, ()>;

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()>;
}

#[derive(Clone)]
struct BlockListCombine {
    left: Arc<dyn BlockList>,
    right: Arc<dyn BlockList>,
    combinator: fn(&Array, &Array) -> Array,
    value_combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
}

impl BlockListCombine {
    fn new(
        left: Arc<dyn BlockList>,
        right: Arc<dyn BlockList>,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<BlockListCombine> {
        if left.shape() != right.shape() {
            return Err(error::bad_request(
                &format!("Cannot combine shape {} with shape", left.shape()),
                right.shape(),
            ));
        }

        Ok(BlockListCombine {
            left,
            right,
            combinator,
            value_combinator,
            dtype,
        })
    }
}

impl TensorInstance for BlockListCombine {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.left.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.left.shape()
    }

    fn size(&self) -> u64 {
        self.left.size()
    }
}

impl BlockList for BlockListCombine {
    fn block_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let left = self.left.clone().block_stream(txn.clone());
            let right = self.right.clone().block_stream(txn);
            let (left, right) = try_join!(left, right)?;

            let combinator = self.combinator;
            let blocks = left
                .zip(right)
                .map(|(l, r)| Ok((l?, r?)))
                .map_ok(move |(l, r)| combinator(&l, &r));
            let blocks: TCTryStream<Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let rebase = transform::Slice::new(self.left.shape().clone(), bounds.clone())?;
            let left = Arc::new(BlockListSlice {
                source: self.left.clone(),
                rebase,
            });

            let rebase = transform::Slice::new(self.right.shape().clone(), bounds)?;
            let right = Arc::new(BlockListSlice {
                source: self.right.clone(),
                rebase,
            });

            let slice = Arc::new(BlockListCombine::new(
                left,
                right,
                self.combinator,
                self.value_combinator,
                self.dtype,
            )?);
            slice.value_stream(txn).await
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let left = self.left.read_value_at(txn, coord);
            let right = self.right.read_value_at(txn, coord);
            let (left, right) = try_join!(left, right)?;
            let combinator = self.value_combinator;
            Ok(combinator(left, right))
        })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
        _number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }

    fn write_value_at<'a>(
        &'a self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

#[async_trait]
impl Transact for BlockListCombine {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }
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
        let file = txn.context().await?;

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

    pub async fn from_values<S: Stream<Item = Number> + Send + Unpin>(
        txn: Arc<Txn>,
        shape: Shape,
        dtype: NumberType,
        values: S,
    ) -> TCResult<BlockListFile> {
        let file = txn.context().await?;

        let mut i = 0u64;
        let mut values = values.chunks(PER_BLOCK);
        while let Some(chunk) = values.next().await {
            let block_id = BlockId::from(i);
            let block = Array::try_from_values(chunk, dtype)?;
            file.create_block(txn.id().clone(), block_id, block).await?;
            i += 1;
        }

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

    async fn merge_sort(&self, txn_id: &TxnId) -> TCResult<()> {
        let num_blocks = div_ceil(self.size(), PER_BLOCK as u64);
        if num_blocks == 1 {
            let mut block = self
                .file
                .get_block(txn_id, 0u64.into())
                .await?
                .upgrade()
                .await?;
            block.sort();
            return Ok(());
        }

        for block_id in 0..(num_blocks - 1) {
            let left = self.file.get_block(txn_id, block_id.into());
            let right = self.file.get_block(txn_id, (block_id + 1).into());
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

impl TensorInstance for BlockListFile {
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

impl BlockList for BlockListFile {
    fn block_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let block_stream = Box::pin(
                stream::iter(0..(div_ceil(self.size(), PER_BLOCK as u64)))
                    .map(BlockId::from)
                    .then(move |block_id| {
                        self.file
                            .clone()
                            .get_block_owned(txn.id().clone(), block_id)
                    }),
            );

            let block_stream =
                block_stream.and_then(|block| future::ready(Ok(block.deref().clone())));

            let block_stream: TCTryStream<Array> = Box::pin(block_stream);

            Ok(block_stream)
        })
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            if bounds == self.shape().all() {
                return self.value_stream(txn).await;
            }

            if !self.shape.contains_bounds(&bounds) {
                return Err(error::bad_request("Invalid bounds", bounds));
            }

            let ndim = bounds.ndim();

            let coord_bounds = af::Array::new(
                &self.coord_bounds,
                af::Dim4::new(&[self.ndim() as u64, 1, 1, 1]),
            );

            let selected =
                stream::iter(bounds.affected())
                    .chunks(PER_BLOCK)
                    .then(move |mut coords| {
                        let (block_ids, af_indices, af_offsets, num_coords) =
                            coord_block(coords.drain(..), &coord_bounds, PER_BLOCK, ndim);

                        let this = self.clone();
                        let txn = txn.clone();

                        Box::pin(async move {
                            let mut start = 0.0f64;
                            let mut values = vec![];
                            for block_id in block_ids {
                                let (block_offsets, new_start) = block_offsets(
                                    &af_indices,
                                    &af_offsets,
                                    num_coords,
                                    start,
                                    block_id,
                                );

                                match this.file.clone().get_block(txn.id(), block_id.into()).await {
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
                        })
                    });

            let selected: TCTryStream<Number> = Box::pin(selected.flatten());
            Ok(selected)
        })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
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
                            let (block_offsets, new_start) = block_offsets(
                                &af_indices,
                                &af_offsets,
                                num_coords,
                                start,
                                block_id,
                            );

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
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            if !self.shape().contains_coord(coord) {
                let coord: Vec<String> = coord.iter().map(|c| c.to_string()).collect();
                return Err(error::bad_request(
                    "Coordinate is out of bounds",
                    coord.join(", "),
                ));
            }

            let offset: u64 = self
                .coord_bounds
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();
            let block_id: u64 = offset / PER_BLOCK as u64;
            let block = self.file.get_block(txn.id(), block_id.into()).await?;
            block
                .deref()
                .get_value((offset % PER_BLOCK as u64) as usize)
                .ok_or_else(|| error::internal(ERR_CORRUPT))
        })
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
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

#[async_trait]
impl Transact for BlockListFile {
    async fn commit(&self, txn_id: &TxnId) {
        self.file.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.file.rollback(txn_id).await
    }
}

struct BlockListBroadcast {
    source: Arc<dyn BlockList>,
    rebase: transform::Broadcast,
}

impl TensorInstance for BlockListBroadcast {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

impl BlockList for BlockListBroadcast {
    fn value_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        let bounds = Bounds::all(self.shape());
        self.value_stream_slice(txn, bounds)
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let num_coords = bounds.size();
            let source_bounds = self.rebase.invert_bounds(bounds);
            let source_coords = source_bounds.affected();
            let coords = source_coords
                .map(move |coord| rebase.map_coord(coord))
                .flatten()
                .map(TCResult::Ok);

            let values = sort_coords(
                txn.subcontext_tmp().await?,
                stream::iter(coords),
                num_coords,
                self.shape(),
            )
            .await?
            .and_then(move |coord| self.clone().read_value_at_owned(txn.clone(), coord));
            let values: TCTryStream<Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let coord = self.rebase.invert_coord(coord);
            self.source.read_value_at(txn, &coord).await
        })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
        _number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }

    fn write_value_at<'a>(
        &'a self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

#[async_trait]
impl Transact for BlockListBroadcast {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }
}

struct BlockListCast {
    source: Arc<dyn BlockList>,
    dtype: NumberType,
}

impl TensorInstance for BlockListCast {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

impl BlockList for BlockListCast {
    fn block_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let dtype = self.dtype;
            let blocks: TCStream<TCResult<Array>> = self.source.clone().block_stream(txn).await?;
            let cast = blocks.map_ok(move |array| array.into_type(dtype));
            let cast: TCTryStream<Array> = Box::pin(cast);
            Ok(cast)
        })
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let dtype = self.dtype;
            let value_stream = self
                .source
                .clone()
                .value_stream_slice(txn, bounds)
                .await?
                .map_ok(move |value| value.into_type(dtype));

            let value_stream: TCTryStream<Number> = Box::pin(value_stream);
            Ok(value_stream)
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        let dtype = self.dtype;
        Box::pin(
            self.source
                .read_value_at(txn, coord)
                .map_ok(move |value| value.into_type(dtype)),
        )
    }

    fn write_value<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.source.clone().write_value(txn_id, bounds, number)
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.source.write_value_at(txn_id, coord, value)
    }
}

#[async_trait]
impl Transact for BlockListCast {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }
}

struct BlockListExpand {
    source: Arc<dyn BlockList>,
    rebase: transform::Expand,
}

impl TensorInstance for BlockListExpand {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim() + 1
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.shape().size()
    }
}

impl BlockList for BlockListExpand {
    fn block_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        self.source.clone().block_stream(txn)
    }

    fn value_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        self.source.clone().value_stream(txn)
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.clone().value_stream_slice(txn, bounds)
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let coord = self.rebase.invert_coord(coord);
            self.source.read_value_at(txn, &coord).await
        })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.clone().write_value(txn_id, bounds, number)
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

#[async_trait]
impl Transact for BlockListExpand {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }
}

type Reductor = fn(&DenseTensor, Arc<Txn>) -> TCBoxTryFuture<Number>;

struct BlockListReduce {
    source: DenseTensor,
    rebase: transform::Reduce,
    reductor: Reductor,
}

impl BlockListReduce {
    fn new(source: DenseTensor, axis: usize, reductor: Reductor) -> TCResult<BlockListReduce> {
        transform::Reduce::new(source.shape().clone(), axis).map(|rebase| BlockListReduce {
            source,
            rebase,
            reductor,
        })
    }
}

impl TensorInstance for BlockListReduce {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.shape().size()
    }
}

impl BlockList for BlockListReduce {
    fn value_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let values = stream::iter(Bounds::all(self.shape()).affected()).then(move |coord| {
                let reductor = self.reductor;
                let txn = txn.clone();
                let source = self.source.clone();
                let source_bounds = self.rebase.invert_coord(&coord);
                Box::pin(async move { reductor(&source.slice(source_bounds)?, txn).await })
            });

            let values: TCTryStream<Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let (source_bounds, slice_reduce_axis) = self.rebase.invert_bounds(bounds);
            let slice = self.source.slice(source_bounds)?;
            BlockListReduce::new(slice, slice_reduce_axis, self.reductor)
                .map(Arc::new)?
                .value_stream(txn)
                .await
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let reductor = self.reductor;
            let source_bounds = self.rebase.invert_coord(coord);
            reductor(&self.source.slice(source_bounds)?, txn.clone()).await
        })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
        _number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }

    fn write_value_at<'a>(
        &'a self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

#[async_trait]
impl Transact for BlockListReduce {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }
}

struct BlockListReshape {
    source: Arc<dyn BlockList>,
    rebase: transform::Reshape,
}

impl TensorInstance for BlockListReshape {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.rebase.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

impl BlockList for BlockListReshape {
    fn block_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        self.source.clone().block_stream(txn)
    }

    fn value_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        self.source.clone().value_stream(txn)
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            if self.ndim() == 1 {
                let (start, end) = self.rebase.offsets(&bounds);
                let source_bounds: Bounds = vec![AxisBounds::from(start..end)].into();
                let rebase =
                    transform::Slice::new(self.source.shape().clone(), source_bounds.clone())?;

                let slice = Arc::new(BlockListSlice {
                    source: self.source.clone(),
                    rebase,
                });

                let value_stream = stream::iter(source_bounds.affected())
                    .zip(slice.value_stream(txn).await?)
                    .map(|(coord, r)| r.map(|value| (coord, value)))
                    .try_filter(move |(coord, _)| future::ready(bounds.contains_coord(coord)))
                    .map_ok(|(_, value)| value);
                let value_stream: TCTryStream<Number> = Box::pin(value_stream);
                Ok(value_stream)
            } else {
                let rebase = transform::Reshape::new(
                    self.source.shape().clone(),
                    vec![self.source.size()].into(),
                )?;

                let flat = Arc::new(BlockListReshape {
                    source: self.source.clone(),
                    rebase,
                });

                let rebase = transform::Reshape::new(flat.shape().clone(), self.shape().clone())?;
                let unflat = Arc::new(BlockListReshape {
                    source: flat,
                    rebase,
                });

                unflat.value_stream_slice(txn, bounds).await
            }
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let coord = self.rebase.invert_coord(coord);
            self.source.read_value_at(txn, &coord).await
        })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            stream::iter(bounds.affected())
                .map(|coord| Ok(self.write_value_at(txn_id.clone(), coord, value.clone())))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.source
            .write_value_at(txn_id, self.rebase.invert_coord(&coord), value)
    }
}

#[async_trait]
impl Transact for BlockListReshape {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }
}

struct BlockListSlice {
    source: Arc<dyn BlockList>,
    rebase: transform::Slice,
}

impl TensorInstance for BlockListSlice {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.rebase.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.rebase.size()
    }
}

impl BlockList for BlockListSlice {
    fn value_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        self.source
            .clone()
            .value_stream_slice(txn, self.rebase.bounds().clone())
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        self.source
            .clone()
            .value_stream_slice(txn, self.rebase.invert_bounds(bounds))
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let coord = self.rebase.invert_coord(coord);
            self.source.read_value_at(txn, &coord).await
        })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.clone().write_value(txn_id, bounds, number)
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

#[async_trait]
impl Transact for BlockListSlice {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }
}

struct BlockListSparse {
    source: SparseTensor,
}

impl BlockListSparse {
    fn new(source: SparseTensor) -> BlockListSparse {
        BlockListSparse { source }
    }
}

impl TensorInstance for BlockListSparse {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

impl BlockList for BlockListSparse {
    fn block_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let dtype = self.dtype();
            let ndim = self.ndim();
            let source = self.source.clone();
            let source_size = source.size();

            let block_offsets = ((PER_BLOCK as u64)..self.size()).step_by(PER_BLOCK);
            let block_stream = stream::iter(block_offsets)
                .map(|offset| (offset - PER_BLOCK as u64, offset))
                .then(move |(start, end)| {
                    let source = source.clone();
                    let txn = txn.clone();

                    Box::pin(async move {
                        let mut filled: Vec<(Vec<u64>, Number)> = source
                            .reshape(vec![source_size].into())?
                            .slice(Bounds::from(vec![AxisBounds::In(start..end)]))?
                            .filled(txn)
                            .await?
                            .try_collect()
                            .await?;

                        let mut block = Array::constant(dtype.zero(), PER_BLOCK);
                        if filled.is_empty() {
                            return Ok(block);
                        }

                        let (mut coords, values): (Vec<Vec<u64>>, Vec<Number>) =
                            filled.drain(..).unzip();
                        let coords: Vec<u64> = coords.drain(..).flatten().collect();
                        let coords = af::Array::new(
                            &coords,
                            af::Dim4::new(&[ndim as u64, coords.len() as u64, 1, 1]),
                        );
                        let values = Array::try_from_values(values, dtype)?;
                        block.set(coords, &values)?;

                        Ok(block)
                    })
                });

            let block_stream: TCTryStream<Array> = Box::pin(block_stream);
            Ok(block_stream)
        })
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let source = self.source.clone().slice(bounds)?;
            let blocks = Arc::new(BlockListSparse::new(source));
            blocks.value_stream(txn).await
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        self.source.read_value(txn, coord)
    }

    fn write_value<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            self.source
                .clone()
                .write_value(txn_id, bounds, number)
                .await
        })
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.source.write_value_at(txn_id, coord, value)
    }
}

#[async_trait]
impl Transact for BlockListSparse {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }
}

struct BlockListTranspose {
    source: Arc<dyn BlockList>,
    rebase: transform::Transpose,
}

impl TensorInstance for BlockListTranspose {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

impl BlockList for BlockListTranspose {
    fn value_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let values = stream::iter(Bounds::all(self.shape()).affected())
                .then(move |coord| self.clone().read_value_at_owned(txn.clone(), coord));
            let values: TCTryStream<Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let values = stream::iter(bounds.affected())
                .then(move |coord| self.clone().read_value_at_owned(txn.clone(), coord));
            let values: TCTryStream<Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let coord = self.rebase.invert_coord(coord);
            self.source.read_value_at(txn, &coord).await
        })
    }

    fn write_value<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.clone().write_value(txn_id, bounds, number)
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

#[async_trait]
impl Transact for BlockListTranspose {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }
}

pub struct BlockListUnary {
    source: Arc<dyn BlockList>,
    transform: fn(&Array) -> Array,
    value_transform: fn(Number) -> Number,
    dtype: NumberType,
}

impl TensorInstance for BlockListUnary {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

impl BlockList for BlockListUnary {
    fn block_stream<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let transform = self.transform;
            let blocks = self
                .source
                .clone()
                .block_stream(txn)
                .await?
                .map_ok(move |array| transform(&array));

            let blocks: TCTryStream<Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn value_stream_slice<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let rebase = transform::Slice::new(self.source.shape().clone(), bounds)?;
            let slice = Arc::new(BlockListSlice {
                source: self.source.clone(),
                rebase,
            });

            slice.value_stream(txn).await
        })
    }

    fn read_value_at<'a>(
        &'a self,
        txn: &'a Arc<Txn>,
        coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        let transform = self.value_transform;
        Box::pin(self.source.read_value_at(txn, coord).map_ok(transform))
    }

    fn write_value<'a>(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
        _number: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }

    fn write_value_at<'a>(
        &'a self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

#[async_trait]
impl Transact for BlockListUnary {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }
}

#[derive(Clone)]
pub struct DenseTensor {
    blocks: Arc<dyn BlockList>,
}

impl DenseTensor {
    pub async fn constant(txn: Arc<Txn>, shape: Shape, value: Number) -> TCResult<DenseTensor> {
        let blocks = Arc::new(BlockListFile::constant(txn, shape, value).await?);
        Ok(DenseTensor { blocks })
    }

    pub fn from_sparse(sparse: SparseTensor) -> DenseTensor {
        let blocks = Arc::new(BlockListSparse::new(sparse));
        DenseTensor { blocks }
    }

    pub async fn value_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<TCResult<Number>>> {
        self.blocks.clone().value_stream(txn).await
    }

    fn read_value_owned<'a>(self, txn: Arc<Txn>, coord: Vec<u64>) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move { self.read_value(&txn, &coord).await })
    }

    fn combine(
        &self,
        other: &Self,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<Self> {
        let (this, that) = broadcast(self, other)?;

        let blocks = Arc::new(BlockListCombine::new(
            this.blocks.clone(),
            that.blocks.clone(),
            combinator,
            value_combinator,
            dtype,
        )?);

        Ok(DenseTensor { blocks })
    }
}

impl TensorInstance for DenseTensor {
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

impl TensorBoolean for DenseTensor {
    fn all(&self, txn: Arc<Txn>) -> TCBoxTryFuture<bool> {
        Box::pin(async move {
            let mut blocks = self.blocks.clone().block_stream(txn).await?;
            while let Some(array) = blocks.next().await {
                if !array?.all() {
                    return Ok(false);
                }
            }

            Ok(true)
        })
    }

    fn any(&'_ self, txn: Arc<Txn>) -> TCBoxTryFuture<'_, bool> {
        Box::pin(async move {
            let mut blocks = self.blocks.clone().block_stream(txn).await?;
            while let Some(array) = blocks.next().await {
                if array?.any() {
                    return Ok(true);
                }
            }

            Ok(false)
        })
    }

    fn and(&self, other: &Self) -> TCResult<Self> {
        self.combine(other, Array::and, Number::and, NumberType::Bool)
    }

    fn not(&self) -> TCResult<Self> {
        let blocks = Arc::new(BlockListUnary {
            source: self.blocks.clone(),
            transform: Array::not,
            value_transform: Number::not,
            dtype: NumberType::Bool,
        });

        Ok(DenseTensor { blocks })
    }

    fn or(&self, other: &Self) -> TCResult<Self> {
        self.combine(other, Array::or, Number::or, NumberType::Bool)
    }

    fn xor(&self, other: &Self) -> TCResult<Self> {
        self.combine(other, Array::xor, Number::xor, NumberType::Bool)
    }
}

#[async_trait]
impl TensorCompare for DenseTensor {
    async fn eq(&self, other: &Self, _txn: Arc<Txn>) -> TCResult<DenseTensor> {
        self.combine(
            other,
            Array::eq,
            <Number as NumberInstance>::eq,
            NumberType::Bool,
        )
    }

    fn gt(&self, other: &Self) -> TCResult<Self> {
        self.combine(
            other,
            Array::gt,
            <Number as NumberInstance>::gt,
            NumberType::Bool,
        )
    }

    async fn gte(&self, other: &Self, _txn: Arc<Txn>) -> TCResult<DenseTensor> {
        self.combine(
            other,
            Array::eq,
            <Number as NumberInstance>::gte,
            NumberType::Bool,
        )
    }

    fn lt(&self, other: &Self) -> TCResult<Self> {
        self.combine(
            other,
            Array::eq,
            <Number as NumberInstance>::lt,
            NumberType::Bool,
        )
    }

    async fn lte(&self, other: &Self, _txn: Arc<Txn>) -> TCResult<DenseTensor> {
        self.combine(
            other,
            Array::eq,
            <Number as NumberInstance>::lte,
            NumberType::Bool,
        )
    }

    fn ne(&self, other: &Self) -> TCResult<Self> {
        self.combine(
            other,
            Array::eq,
            <Number as NumberInstance>::ne,
            NumberType::Bool,
        )
    }
}

impl TensorMath for DenseTensor {
    fn abs(&self) -> TCResult<Self> {
        let is_abs = match self.dtype() {
            NumberType::Bool => true,
            NumberType::UInt(_) => true,
            _ => false,
        };

        if is_abs {
            return Ok(self.clone());
        }

        let blocks = Arc::new(BlockListUnary {
            source: self.blocks.clone(),
            transform: Array::abs,
            value_transform: <Number as NumberInstance>::abs,
            dtype: NumberType::Bool,
        });

        Ok(DenseTensor { blocks })
    }

    fn add(&self, other: &Self) -> TCResult<Self> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, Array::add, <Number as NumberInstance>::add, dtype)
    }

    fn multiply(&self, other: &Self) -> TCResult<Self> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(
            other,
            Array::multiply,
            <Number as NumberInstance>::multiply,
            dtype,
        )
    }
}

impl TensorIO for DenseTensor {
    fn mask<'a>(&'a self, _txn: &'a Arc<Txn>, _other: Self) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { Err(error::not_implemented("DenseTensor::mask")) })
    }

    fn read_value<'a>(&'a self, txn: &'a Arc<Txn>, coord: &'a [u64]) -> TCBoxTryFuture<Number> {
        self.blocks.read_value_at(txn, coord)
    }

    fn write<'a>(&'a self, txn: Arc<Txn>, bounds: Bounds, other: Self) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let slice = self.slice(bounds)?;
            let other = other
                .broadcast(slice.shape().clone())?
                .as_type(self.dtype())?;

            let txn_id = txn.id().clone();
            let txn_id_clone = txn_id.clone();
            stream::iter(Bounds::all(slice.shape()).affected())
                .map(move |coord| {
                    Ok(other
                        .clone()
                        .read_value_owned(txn.clone(), coord.to_vec())
                        .map_ok(|value| (coord, value)))
                })
                .try_buffer_unordered(2)
                .and_then(|(coord, value)| slice.write_value_at(txn_id_clone.clone(), coord, value))
                .try_fold((), |_, _| future::ready(Ok(())))
                .await?;

            Ok(())
        })
    }

    fn write_value(
        &'_ self,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCBoxTryFuture<'_, ()> {
        self.blocks.clone().write_value(txn_id, bounds, value)
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.blocks.write_value_at(txn_id, coord, value)
    }
}

impl TensorReduce for DenseTensor {
    fn product(&self, _axis: usize) -> TCResult<Self> {
        Err(error::not_implemented("DenseTensor::product"))
    }

    fn product_all(&self, txn: Arc<Txn>) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            self.blocks
                .clone()
                .block_stream(txn)
                .await?
                .map_ok(|array| array.product())
                .try_fold(self.dtype().one(), |product, block_product| {
                    future::ready(Ok(product * block_product))
                })
                .await
        })
    }

    fn sum(&self, _axis: usize) -> TCResult<Self> {
        Err(error::not_implemented("DenseTensor::sum"))
    }

    fn sum_all(&self, txn: Arc<Txn>) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            self.blocks
                .clone()
                .block_stream(txn)
                .await?
                .map_ok(|array| array.sum())
                .try_fold(self.dtype().one(), |sum, block_sum| {
                    future::ready(Ok(sum + block_sum))
                })
                .await
        })
    }
}

impl TensorTransform for DenseTensor {
    fn as_type(&self, dtype: NumberType) -> TCResult<Self> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }

        let blocks = Arc::new(BlockListCast {
            source: self.blocks.clone(),
            dtype,
        });

        Ok(DenseTensor { blocks })
    }

    fn broadcast(&self, shape: Shape) -> TCResult<Self> {
        if &shape == self.shape() {
            return Ok(self.clone());
        }

        Err(error::not_implemented("DenseTensor::broadcast"))
    }

    fn expand_dims(&self, axis: usize) -> TCResult<Self> {
        let rebase = transform::Expand::new(self.shape().clone(), axis)?;
        let blocks = Arc::new(BlockListExpand {
            source: self.blocks.clone(),
            rebase,
        });

        Ok(DenseTensor { blocks })
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Self> {
        if bounds == Bounds::all(self.shape()) {
            return Ok(self.clone());
        }

        let rebase = transform::Slice::new(self.shape().clone(), bounds)?;
        let blocks = Arc::new(BlockListSlice {
            source: self.blocks.clone(),
            rebase,
        });

        Ok(DenseTensor { blocks })
    }

    fn reshape(&self, shape: Shape) -> TCResult<Self> {
        if &shape == self.shape() {
            return Ok(self.clone());
        }

        let rebase = transform::Reshape::new(self.shape().clone(), shape)?;
        let blocks = Arc::new(BlockListReshape {
            source: self.blocks.clone(),
            rebase,
        });

        Ok(DenseTensor { blocks })
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        if permutation == Some((0..self.ndim()).collect()) {
            return Ok(self.clone());
        }

        let rebase = transform::Transpose::new(self.shape().clone(), permutation)?;
        let blocks = Arc::new(BlockListTranspose {
            source: self.blocks.clone(),
            rebase,
        });

        Ok(DenseTensor { blocks })
    }
}

#[async_trait]
impl Transact for DenseTensor {
    async fn commit(&self, txn_id: &TxnId) {
        self.blocks.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.blocks.rollback(txn_id).await
    }
}

impl From<BlockListFile> for DenseTensor {
    fn from(blocks: BlockListFile) -> DenseTensor {
        let blocks = Arc::new(blocks);
        DenseTensor { blocks }
    }
}

pub async fn sort_coords<S: Stream<Item = TCResult<Vec<u64>>> + Send + Unpin + 'static>(
    txn: Arc<Txn>,
    coords: S,
    num_coords: u64,
    shape: &Shape,
) -> TCResult<impl Stream<Item = TCResult<Vec<u64>>>> {
    let ndim = shape.len();
    let coord_bounds: Vec<u64> = (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect();
    let coord_bounds: af::Array<u64> =
        af::Array::new(&coord_bounds, af::Dim4::new(&[ndim as u64, 1, 1, 1]));
    let coord_bounds_copy = coord_bounds.copy();
    let shape: af::Array<u64> =
        af::Array::new(&shape.to_vec(), af::Dim4::new(&[ndim as u64, 1, 1, 1]));

    let blocks = coords
        .chunks(PER_BLOCK)
        .map(|block| block.into_iter().collect::<TCResult<Vec<Vec<u64>>>>())
        .map_ok(move |block| {
            let num_coords = block.len();
            let block = block.into_iter().flatten().collect::<Vec<u64>>();
            af::Array::new(
                &block,
                af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]),
            )
        })
        .map_ok(move |block| af::sum(&(block * coord_bounds.copy()), 1))
        .and_then(|block| future::ready(Array::try_from(block)));

    let blocks: TCTryStream<Array> = Box::pin(blocks);
    let block_list = BlockListFile::from_blocks(
        txn.clone(),
        Shape::from(vec![num_coords]),
        NumberType::uint64(),
        blocks,
    )
    .await
    .map(Arc::new)?;
    block_list.merge_sort(txn.id()).await?;

    let coords = block_list
        .block_stream(txn)
        .await?
        .map_ok(|block| block.into_af_array::<u64>())
        .map_ok(|block| af::moddims(&block, af::Dim4::new(&[1, block.elements() as u64, 1, 1])))
        .map_ok(move |block| {
            let dims = af::Dim4::new(&[ndim as u64, block.elements() as u64, 1, 1]);
            let block = af::tile(&block, dims);
            let coord_bounds = af::tile(&coord_bounds_copy, dims);
            let shape = af::tile(&shape, dims);
            (block / coord_bounds) % shape
        })
        .map_ok(|coord_block| {
            let mut coords: Vec<u64> = Vec::with_capacity(coord_block.elements());
            coord_block.host(&mut coords);
            coords
        })
        .map_ok(move |coords| {
            stream::iter(coords.into_iter())
                .chunks(ndim)
                .map(Result::<Vec<u64>, error::TCError>::Ok)
        })
        .try_flatten();

    Ok(coords)
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
