use std::sync::Arc;

use arrayfire as af;
use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;

use crate::class::{TCBoxTryFuture, TCResult, TCStream, TCTryStream};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{AxisBounds, Bounds, Shape};
use super::class::TensorInstance;
use super::*;

mod array;
mod file;

pub use array::Array;
pub use file::*;

#[async_trait]
trait BlockList: TensorInstance + Transact + 'static {
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let dtype = self.dtype();
            let blocks = self
                .value_stream(txn)
                .await?
                .chunks(file::PER_BLOCK)
                .map(|values| values.into_iter().collect::<TCResult<Vec<Number>>>())
                .and_then(move |values| future::ready(Array::try_from_values(values, dtype)));

            let blocks: TCTryStream<Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn value_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let values = self
                .block_stream(txn)
                .await?
                .and_then(|array| future::ready(Ok(array.into_values())))
                .map_ok(|values| {
                    values
                        .into_iter()
                        .map(Ok)
                        .collect::<Vec<TCResult<Number>>>()
                })
                .map_ok(stream::iter)
                .try_flatten();

            let values: TCTryStream<Number> = Box::pin(values);
            Ok(values)
        })
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>>;

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number>;

    fn read_value_at_owned<'a>(
        self: Arc<Self>,
        txn: Txn,
        coord: Vec<u64>,
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move { self.read_value_at(&txn, &coord).await })
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCResult<()>;

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()>;
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

#[async_trait]
impl BlockList for BlockListCombine {
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
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

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
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
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let left = self.left.read_value_at(txn, coord);
        let right = self.right.read_value_at(txn, coord);
        let (left, right) = try_join!(left, right)?;
        let combinator = self.value_combinator;
        Ok(combinator(left, right))
    }

    async fn write_value(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
        _number: Number,
    ) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(
        &self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
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

#[async_trait]
impl BlockList for BlockListBroadcast {
    fn value_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        let bounds = Bounds::all(self.shape());
        self.value_stream_slice(txn, bounds)
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        let rebase = self.rebase.clone();
        let num_coords = bounds.size();
        let source_bounds = self.rebase.invert_bounds(bounds);
        let source_coords = source_bounds.affected();
        let coords = source_coords
            .map(move |coord| rebase.map_coord(coord))
            .flatten();

        let values = sort_coords(
            txn.subcontext_tmp().await?,
            stream::iter(coords.map(TCResult::Ok)),
            num_coords,
            self.shape(),
        )
        .await?
        .and_then(move |coord| self.clone().read_value_at_owned(txn.clone(), coord));
        let values: TCTryStream<Number> = Box::pin(values);
        Ok(values)
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
        _number: Number,
    ) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(
        &self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, _txn_id: &TxnId) {
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

#[async_trait]
impl BlockList for BlockListCast {
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let dtype = self.dtype;
            let blocks: TCStream<TCResult<Array>> = self.source.clone().block_stream(txn).await?;
            let cast = blocks.map_ok(move |array| array.into_type(dtype));
            let cast: TCTryStream<Array> = Box::pin(cast);
            Ok(cast)
        })
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        let dtype = self.dtype;
        let value_stream = self
            .source
            .clone()
            .value_stream_slice(txn, bounds)
            .await?
            .map_ok(move |value| value.into_type(dtype));

        let value_stream: TCTryStream<Number> = Box::pin(value_stream);
        Ok(value_stream)
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let dtype = self.dtype;
        self.source
            .read_value_at(txn, coord)
            .map_ok(move |value| value.into_type(dtype))
            .await
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCResult<()> {
        self.source
            .clone()
            .write_value(txn_id, bounds, number)
            .await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
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

#[async_trait]
impl BlockList for BlockListExpand {
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        self.source.clone().block_stream(txn)
    }

    fn value_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        self.source.clone().value_stream(txn)
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.clone().value_stream_slice(txn, bounds).await
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCResult<()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source
            .clone()
            .write_value(txn_id, bounds, number)
            .await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

type Reductor = fn(&DenseTensor, Txn) -> TCBoxTryFuture<Number>;

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

#[async_trait]
impl BlockList for BlockListReduce {
    fn value_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
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

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        let (source_bounds, slice_reduce_axis) = self.rebase.invert_bounds(bounds);
        let slice = self.source.slice(source_bounds)?;
        BlockListReduce::new(slice, slice_reduce_axis, self.reductor)
            .map(Arc::new)?
            .value_stream(txn)
            .await
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let reductor = self.reductor;
        let source_bounds = self.rebase.invert_coord(coord);
        reductor(&self.source.slice(source_bounds)?, txn.clone()).await
    }

    async fn write_value(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
        _number: Number,
    ) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(
        &self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, _txn_id: &TxnId) {
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

#[async_trait]
impl BlockList for BlockListReshape {
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        self.source.clone().block_stream(txn)
    }

    fn value_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        self.source.clone().value_stream(txn)
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        if self.ndim() == 1 {
            let (start, end) = self.rebase.offsets(&bounds);
            let source_bounds: Bounds = vec![AxisBounds::from(start..end)].into();
            let rebase = transform::Slice::new(self.source.shape().clone(), source_bounds.clone())?;

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
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCResult<()> {
        stream::iter(bounds.affected())
            .map(|coord| Ok(self.write_value_at(txn_id, coord, value.clone())))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
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

#[async_trait]
impl BlockList for BlockListSlice {
    fn value_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        self.source
            .clone()
            .value_stream_slice(txn, self.rebase.bounds().clone())
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        self.source
            .clone()
            .value_stream_slice(txn, self.rebase.invert_bounds(bounds))
            .await
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCResult<()> {
        debug!("BlockListSlice::write_value {} at {}", value, bounds);

        let bounds = self.rebase.invert_bounds(bounds);
        self.source.clone().write_value(txn_id, bounds, value).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
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

#[async_trait]
impl BlockList for BlockListSparse {
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let dtype = self.dtype();
            let ndim = self.ndim();
            let source = self.source.clone();
            let source_size = source.size();

            let block_offsets = ((file::PER_BLOCK as u64)..self.size()).step_by(PER_BLOCK);
            let block_stream = stream::iter(block_offsets)
                .map(|offset| (offset - file::PER_BLOCK as u64, offset))
                .then(move |(start, end)| {
                    let source = source.clone();
                    let txn = txn.clone();

                    Box::pin(async move {
                        let filled: Vec<(Vec<u64>, Number)> = source
                            .reshape(vec![source_size].into())?
                            .slice(Bounds::from(vec![AxisBounds::In(start..end)]))?
                            .filled(txn)
                            .await?
                            .try_collect()
                            .await?;

                        let mut block = Array::constant(dtype.zero(), file::PER_BLOCK);
                        if filled.is_empty() {
                            return Ok(block);
                        }

                        let (coords, values): (Vec<Vec<u64>>, Vec<Number>) =
                            filled.into_iter().unzip();
                        let coords: Vec<u64> = coords.into_iter().flatten().collect();
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

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        let source = self.source.clone().slice(bounds)?;
        let blocks = Arc::new(BlockListSparse::new(source));
        blocks.value_stream(txn).await
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.source.read_value(txn, coord).await
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCResult<()> {
        self.source
            .clone()
            .write_value(txn_id, bounds, number)
            .await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
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

#[async_trait]
impl BlockList for BlockListTranspose {
    fn value_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Number>> {
        Box::pin(async move {
            let values = stream::iter(Bounds::all(self.shape()).affected())
                .then(move |coord| self.clone().read_value_at_owned(txn.clone(), coord));
            let values: TCTryStream<Number> = Box::pin(values);
            Ok(values)
        })
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        let values = stream::iter(bounds.affected())
            .then(move |coord| self.clone().read_value_at_owned(txn.clone(), coord));
        let values: TCTryStream<Number> = Box::pin(values);
        Ok(values)
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        number: Number,
    ) -> TCResult<()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source
            .clone()
            .write_value(txn_id, bounds, number)
            .await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
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

#[async_trait]
impl BlockList for BlockListUnary {
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
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

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        let rebase = transform::Slice::new(self.source.shape().clone(), bounds)?;
        let slice = Arc::new(BlockListSlice {
            source: self.source.clone(),
            rebase,
        });

        slice.value_stream(txn).await
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let transform = self.value_transform;
        self.source
            .read_value_at(txn, coord)
            .map_ok(transform)
            .await
    }

    async fn write_value(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
        _number: Number,
    ) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(
        &self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<()> {
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

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}

#[derive(Clone)]
pub struct DenseTensor {
    blocks: Arc<dyn BlockList>,
}

impl DenseTensor {
    pub async fn constant(txn: &Txn, shape: Shape, value: Number) -> TCResult<DenseTensor> {
        let blocks = Arc::new(BlockListFile::constant(txn, shape, value).await?);
        Ok(DenseTensor { blocks })
    }

    pub fn from_sparse(sparse: SparseTensor) -> DenseTensor {
        let blocks = Arc::new(BlockListSparse::new(sparse));
        DenseTensor { blocks }
    }

    pub async fn value_stream(&self, txn: Txn) -> TCResult<TCStream<TCResult<Number>>> {
        self.blocks.clone().value_stream(txn).await
    }

    fn read_value_owned<'a>(self, txn: Txn, coord: Vec<u64>) -> TCBoxTryFuture<'a, Number> {
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

#[async_trait]
impl TensorBoolean for DenseTensor {
    async fn all(&self, txn: Txn) -> TCResult<bool> {
        let mut blocks = self.blocks.clone().block_stream(txn).await?;
        while let Some(array) = blocks.next().await {
            if !array?.all() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(&self, txn: Txn) -> TCResult<bool> {
        let mut blocks = self.blocks.clone().block_stream(txn).await?;
        while let Some(array) = blocks.next().await {
            if array?.any() {
                return Ok(true);
            }
        }

        Ok(false)
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
    async fn eq(&self, other: &Self, _txn: Txn) -> TCResult<DenseTensor> {
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

    async fn gte(&self, other: &Self, _txn: Txn) -> TCResult<DenseTensor> {
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

    async fn lte(&self, other: &Self, _txn: Txn) -> TCResult<DenseTensor> {
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

#[async_trait]
impl TensorIO for DenseTensor {
    async fn mask(&self, _txn: &Txn, _other: Self) -> TCResult<()> {
        Err(error::not_implemented("DenseTensor::mask"))
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.blocks.read_value_at(txn, coord).await
    }

    async fn write(&self, txn: Txn, bounds: Bounds, other: Self) -> TCResult<()> {
        let slice = self.slice(bounds)?;
        let other = other
            .broadcast(slice.shape().clone())?
            .as_type(self.dtype())?;

        let txn_id = txn.id().clone();
        let txn_id_clone = txn_id;
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
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        self.blocks.clone().write_value(txn_id, bounds, value).await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        self.blocks.write_value_at(txn_id, coord, value).await
    }
}

impl TensorReduce for DenseTensor {
    fn product(&self, _axis: usize) -> TCResult<Self> {
        Err(error::not_implemented("DenseTensor::product"))
    }

    fn product_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
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

    fn sum_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        self.blocks.finalize(txn_id).await
    }
}

impl From<BlockListFile> for DenseTensor {
    fn from(blocks: BlockListFile) -> DenseTensor {
        let blocks = Arc::new(blocks);
        DenseTensor { blocks }
    }
}
