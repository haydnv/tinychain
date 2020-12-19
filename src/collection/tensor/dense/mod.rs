use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;

use crate::class::Instance;
use crate::collection::{from_dense, Collection};
use crate::error;
use crate::general::{TCBoxTryFuture, TCResult, TCStream, TCTryStream};
use crate::handler::*;
use crate::scalar::number::*;
use crate::scalar::{MethodType, PathSegment};
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{Bounds, Shape};
use super::class::{Tensor, TensorInstance, TensorType};
use super::sparse::{DenseAccessor, SparseAccess, SparseTensor};
use super::stream::*;
use super::transform;
use super::{
    IntoView, TensorAccessor, TensorBoolean, TensorCompare, TensorDualIO, TensorIO, TensorMath,
    TensorReduce, TensorTransform, TensorUnary, ERR_NONBIJECTIVE_WRITE,
};

mod array;
mod file;

pub use array::Array;
pub use file::*;

#[async_trait]
pub trait BlockList: TensorAccessor + Transact + 'static {
    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let dtype = self.dtype();
            let blocks = self
                .value_stream(txn)
                .await?
                .chunks(file::PER_BLOCK)
                .map(|values| values.into_iter().collect::<TCResult<Vec<Number>>>())
                .and_then(move |values| future::ready(Array::try_from_values(values, dtype)));

            let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
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

            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    async fn value_stream_slice<'a>(
        &'a self,
        txn: &'a Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>>;

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number>;

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()>;

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()>;
}

#[derive(Clone)]
pub struct BlockListDyn {
    source: Arc<dyn BlockList>,
}

impl BlockListDyn {
    pub fn new<T: BlockList>(source: T) -> BlockListDyn {
        Self {
            source: Arc::new(source),
        }
    }
}

impl TensorAccessor for BlockListDyn {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl BlockList for BlockListDyn {
    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        self.source.block_stream(txn)
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        self.source.value_stream(txn)
    }

    async fn value_stream_slice<'a>(
        &'a self,
        txn: &'a Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        self.source.value_stream_slice(txn, bounds).await
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.source.read_value_at(txn, coord).await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        self.source.write_value(txn_id, bounds, number).await
    }

    fn write_value_at(
        &self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'_, ()> {
        self.source.write_value_at(txn_id, coord, value)
    }
}

#[async_trait]
impl Transact for BlockListDyn {
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

#[derive(Clone)]
pub struct BlockListCombine<L: BlockList, R: BlockList> {
    left: L,
    right: R,
    combinator: fn(&Array, &Array) -> Array,
    value_combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
}

impl<L: BlockList, R: BlockList> BlockListCombine<L, R> {
    fn new(
        left: L,
        right: R,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<BlockListCombine<L, R>> {
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

impl<L: BlockList, R: BlockList> TensorAccessor for BlockListCombine<L, R> {
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
impl<L: Clone + BlockList, R: Clone + BlockList> BlockList for BlockListCombine<L, R> {
    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let left = self.left.block_stream(txn);
            let right = self.right.block_stream(txn);
            let (left, right) = try_join!(left, right)?;

            let combinator = self.combinator;
            let blocks = left
                .zip(right)
                .map(|(l, r)| Ok((l?, r?)))
                .map_ok(move |(l, r)| combinator(&l, &r));

            let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    async fn value_stream_slice<'a>(
        &'a self,
        _txn: &'a Txn,
        _bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        // let rebase = transform::Slice::new(self.left.shape().clone(), bounds.clone())?;
        // let left = BlockListSlice {
        //     source: self.left.clone(),
        //     rebase,
        // };
        //
        // let rebase = transform::Slice::new(self.right.shape().clone(), bounds)?;
        // let right = BlockListSlice {
        //     source: self.right.clone(),
        //     rebase,
        // };
        //
        // let slice: BlockListCombine<BlockListSlice<L>, BlockListSlice<R>> = BlockListCombine::new(
        //         left,
        //         right,
        //         self.combinator,
        //         self.value_combinator,
        //         self.dtype,
        //     )?;
        //
        // slice.value_stream(txn).await
        unimplemented!()
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let left = self.left.read_value_at(txn, coord);
        let right = self.right.read_value_at(txn, coord);
        let (left, right) = try_join!(left, right)?;
        let combinator = self.value_combinator;
        Ok(combinator(left, right))
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
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
impl<L: BlockList, R: BlockList> Transact for BlockListCombine<L, R> {
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
pub struct BlockListBroadcast<T: BlockList> {
    source: T,
    rebase: transform::Broadcast,
}

impl<T: BlockList> TensorAccessor for BlockListBroadcast<T> {
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
impl<T: BlockList> BlockList for BlockListBroadcast<T> {
    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        let bounds = Bounds::all(self.shape());
        self.value_stream_slice(txn, bounds)
    }

    async fn value_stream_slice<'a>(
        &'a self,
        txn: &'a Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        let rebase = self.rebase.clone();
        let source_bounds = self.rebase.invert_bounds(bounds);
        let coords = source_bounds
            .affected()
            .map(move |coord| rebase.map_coord(coord))
            .flatten();
        let coords = stream::iter(coords.map(TCResult::Ok));

        let values = sorted_values(txn, self, coords, source_bounds.size()).await?;
        Ok(Box::pin(values.map_ok(|(_, value)| value)))
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
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

impl<T: BlockList> ReadValueAt for BlockListBroadcast<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a> {
        Box::pin(async move {
            let value = BlockList::read_value_at(self, txn, &coord).await?;
            Ok((coord, value))
        })
    }
}

#[async_trait]
impl<T: BlockList> Transact for BlockListBroadcast<T> {
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
pub struct BlockListCast<T: BlockList> {
    source: T,
    dtype: NumberType,
}

impl<T: BlockList> TensorAccessor for BlockListCast<T> {
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
impl<T: BlockList> BlockList for BlockListCast<T> {
    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let dtype = self.dtype;
            let blocks: TCStream<'a, TCResult<Array>> = self.source.block_stream(txn).await?;
            let cast = blocks.map_ok(move |array| array.into_type(dtype));
            let cast: TCTryStream<'a, Array> = Box::pin(cast);
            Ok(cast)
        })
    }

    async fn value_stream_slice<'a>(
        &'a self,
        txn: &'a Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        let dtype = self.dtype;
        let value_stream = self.source.value_stream_slice(txn, bounds).await?;

        let value_stream: TCTryStream<'a, Number> =
            Box::pin(value_stream.map_ok(move |value| value.into_type(dtype)));
        Ok(value_stream)
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let dtype = self.dtype;
        self.source
            .read_value_at(txn, coord)
            .map_ok(move |value| value.into_type(dtype))
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        self.source.write_value(txn_id, bounds, number).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
        self.source.write_value_at(txn_id, coord, value)
    }
}

impl<T: BlockList> ReadValueAt for BlockListCast<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a> {
        Box::pin(async move {
            let value = BlockList::read_value_at(self, txn, &coord).await?;
            Ok((coord, value))
        })
    }
}

#[async_trait]
impl<T: BlockList> Transact for BlockListCast<T> {
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

#[derive(Clone)]
pub struct BlockListExpand<T: BlockList> {
    source: T,
    rebase: transform::Expand,
}

impl<T: BlockList> TensorAccessor for BlockListExpand<T> {
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
impl<T: BlockList> BlockList for BlockListExpand<T> {
    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        self.source.block_stream(txn)
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        self.source.value_stream(txn)
    }

    async fn value_stream_slice<'a>(
        &'a self,
        txn: &'a Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.value_stream_slice(txn, bounds).await
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.write_value(txn_id, bounds, number).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

#[async_trait]
impl<T: BlockList> Transact for BlockListExpand<T> {
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

// TODO: &Txn, not Txn
type Reductor = fn(&DenseTensor<BlockListDyn>, Txn) -> TCBoxTryFuture<Number>;

#[derive(Clone)]
pub struct BlockListReduce<T: Clone + BlockList> {
    source: DenseTensor<T>,
    rebase: transform::Reduce,
    reductor: Reductor,
}

impl<T: Clone + BlockList> BlockListReduce<T> {
    fn new(
        source: DenseTensor<T>,
        axis: usize,
        reductor: Reductor,
    ) -> TCResult<BlockListReduce<T>> {
        let rebase = transform::Reduce::new(source.shape().clone(), axis)?;

        Ok(BlockListReduce {
            source,
            rebase,
            reductor,
        })
    }
}

impl<T: Clone + BlockList> TensorAccessor for BlockListReduce<T> {
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
impl<T: Clone + BlockList> BlockList for BlockListReduce<T> {
    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let values = stream::iter(Bounds::all(self.shape()).affected()).then(move |coord| {
                let reductor = self.reductor;
                let txn = txn.clone();
                let source_bounds = self.rebase.invert_coord(&coord);
                Box::pin(async move {
                    let slice = self.source.slice(source_bounds)?;
                    reductor(&slice.into_dyn(), txn).await
                })
            });

            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    async fn value_stream_slice<'a>(
        &'a self,
        _txn: &'a Txn,
        _bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        // let (source_bounds, slice_reduce_axis) = self.rebase.invert_bounds(bounds);
        // let slice = self.source.slice(source_bounds)?;
        // BlockListReduce::new(slice, slice_reduce_axis, self.reductor)?
        //     .value_stream(txn)
        //     .await
        unimplemented!()
    }

    async fn read_value_at(&self, _txn: &Txn, _coord: &[u64]) -> TCResult<Number> {
        // let reductor = self.reductor;
        // let source_bounds = self.rebase.invert_coord(coord);
        // let slice = self.source.slice(source_bounds)?;
        // reductor(&slice.into_dyn(), txn.clone()).await
        unimplemented!()
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
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
impl<T: Clone + BlockList> Transact for BlockListReduce<T> {
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
pub struct BlockListSlice<T> {
    source: T,
    rebase: transform::Slice,
}

impl<T: BlockList> TensorAccessor for BlockListSlice<T> {
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
impl<T: BlockList> BlockList for BlockListSlice<T> {
    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        self.source
            .value_stream_slice(txn, self.rebase.bounds().clone())
    }

    async fn value_stream_slice<'a>(
        &'a self,
        txn: &'a Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        self.source
            .value_stream_slice(txn, self.rebase.invert_bounds(bounds))
            .await
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        debug!("BlockListSlice::write_value {} at {}", value, bounds);

        let bounds = self.rebase.invert_bounds(bounds);
        self.source.write_value(txn_id, bounds, value).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

#[async_trait]
impl<T: BlockList> Transact for BlockListSlice<T> {
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

#[derive(Clone)]
pub struct BlockListSparse<T: Clone + SparseAccess> {
    source: SparseTensor<T>,
}

impl<T: Clone + SparseAccess> BlockListSparse<T> {
    fn new(source: SparseTensor<T>) -> Self {
        BlockListSparse { source }
    }
}

impl<T: Clone + SparseAccess> TensorAccessor for BlockListSparse<T> {
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
impl<T: Clone + SparseAccess> BlockList for BlockListSparse<T> {
    fn block_stream<'a>(&'a self, _txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(future::ready(Err(error::not_implemented(
            "BlockListSparse::block_stream",
        ))))
    }

    async fn value_stream_slice<'a>(
        &'a self,
        _txn: &'a Txn,
        _bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        // let source = self.source.slice(bounds)?;
        // let blocks = BlockListSparse::new(source);
        // blocks.value_stream(txn).await
        unimplemented!()
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.source.read_value(txn, coord).await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
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
impl<T: Clone + SparseAccess> Transact for BlockListSparse<T> {
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

#[derive(Clone)]
pub struct BlockListTranspose<T: BlockList> {
    source: T,
    rebase: transform::Transpose,
}

impl<T: BlockList> TensorAccessor for BlockListTranspose<T> {
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
impl<T: BlockList> BlockList for BlockListTranspose<T> {
    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let coords = stream::iter(Bounds::all(self.shape()).affected().map(TCResult::Ok));
            let values: TCTryStream<'a, (Vec<u64>, Number)> =
                Box::pin(ValueReader::new(coords, txn, self));
            let values: TCTryStream<'a, Number> = Box::pin(values.map_ok(|(_, value)| value));
            Ok(values)
        })
    }

    async fn value_stream_slice<'a>(
        &'a self,
        txn: &'a Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        let coords = stream::iter(bounds.affected().map(TCResult::Ok));
        let values: TCTryStream<'a, (Vec<u64>, Number)> =
            Box::pin(ValueReader::new(coords, txn, self));
        let values: TCTryStream<'a, Number> = Box::pin(values.map_ok(|(_, value)| value));
        Ok(values)
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value_at(txn, &coord).await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.write_value(txn_id, bounds, number).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

impl<T: BlockList> ReadValueAt for BlockListTranspose<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a> {
        Box::pin(async move {
            let value = BlockList::read_value_at(self, txn, &coord).await?;
            Ok((coord, value))
        })
    }
}

#[async_trait]
impl<T: BlockList> Transact for BlockListTranspose<T> {
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

#[derive(Clone)]
pub struct BlockListUnary<T: BlockList> {
    source: T,
    transform: fn(&Array) -> Array,
    value_transform: fn(Number) -> Number,
    dtype: NumberType,
}

impl<T: BlockList> TensorAccessor for BlockListUnary<T> {
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
impl<T: BlockList> BlockList for BlockListUnary<T> {
    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let transform = self.transform;
            let blocks = self.source.block_stream(txn).await?;
            let blocks: TCTryStream<'a, Array> =
                Box::pin(blocks.map_ok(move |array| transform(&array)));
            Ok(blocks)
        })
    }

    async fn value_stream_slice<'a>(
        &'a self,
        _txn: &'a Txn,
        _bounds: Bounds,
    ) -> TCResult<TCTryStream<'a, Number>> {
        // let rebase = transform::Slice::new(self.source.shape().clone(), bounds)?;
        // let slice = BlockListSlice {
        //     source: self.source.clone(),
        //     rebase,
        // };
        //
        // slice.value_stream(txn).await
        unimplemented!()
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let transform = self.value_transform;
        self.source
            .read_value_at(txn, coord)
            .map_ok(transform)
            .await
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
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
impl<T: BlockList> Transact for BlockListUnary<T> {
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
pub struct DenseTensor<T: Clone + BlockList> {
    blocks: T,
}

impl<T: Clone + BlockList> DenseTensor<T> {
    pub fn into_dyn(self) -> DenseTensor<BlockListDyn> {
        let blocks = BlockListDyn::new(self.clone_into());
        DenseTensor { blocks }
    }

    pub fn clone_into(&self) -> T {
        self.blocks.clone()
    }

    pub async fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCResult<TCTryStream<'a, Number>> {
        self.blocks.value_stream(txn).await
    }

    fn combine<OT: Clone + BlockList>(
        &self,
        other: &DenseTensor<OT>,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<DenseTensor<BlockListCombine<T, OT>>> {
        if self.shape() != other.shape() {
            return Err(error::unsupported(format!(
                "Cannot combine tensors with different shapes: {}, {}",
                self.shape(),
                other.shape()
            )));
        }

        let blocks = BlockListCombine::new(
            self.blocks.clone(),
            other.blocks.clone(),
            combinator,
            value_combinator,
            dtype,
        )?;

        Ok(DenseTensor { blocks })
    }
}

impl<T: Clone + BlockList> Instance for DenseTensor<T> {
    type Class = TensorType;

    fn class(&self) -> TensorType {
        TensorType::Dense
    }
}

impl<T: Clone + BlockList> TensorInstance for DenseTensor<T> {
    type Dense = Self;
    type Sparse = SparseTensor<DenseAccessor<T>>;

    fn into_dense(self) -> Self::Dense {
        self
    }

    fn into_sparse(self) -> Self::Sparse {
        from_dense(self)
    }
}

impl<T: Clone + BlockList> IntoView for DenseTensor<T> {
    fn into_view(self) -> Tensor {
        let blocks = BlockListDyn::new(self.clone_into());
        Tensor::Dense(DenseTensor { blocks })
    }
}

impl<T: Clone + BlockList> ReadValueAt for DenseTensor<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a> {
        Box::pin(async move {
            let value = BlockList::read_value_at(&self.blocks, txn, &coord).await?;
            Ok((coord, value))
        })
    }
}

impl<T: Clone + BlockList> TensorAccessor for DenseTensor<T> {
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
impl<T: Clone + BlockList, OT: Clone + BlockList> TensorBoolean<DenseTensor<OT>>
    for DenseTensor<T>
{
    type Combine = DenseTensor<BlockListCombine<T, OT>>;

    fn and(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        self.combine(other, Array::and, Number::and, NumberType::Bool)
    }

    fn or(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        self.combine(other, Array::or, Number::or, NumberType::Bool)
    }

    fn xor(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        self.combine(other, Array::xor, Number::xor, NumberType::Bool)
    }
}

#[async_trait]
impl<T: Clone + BlockList> TensorUnary for DenseTensor<T> {
    type Unary = DenseTensor<BlockListUnary<T>>;

    fn abs(&self) -> TCResult<Self::Unary> {
        let blocks = BlockListUnary {
            source: self.blocks.clone(),
            transform: Array::abs,
            value_transform: <Number as NumberInstance>::abs,
            dtype: NumberType::Bool,
        };

        Ok(DenseTensor { blocks })
    }

    async fn all(&self, txn: Txn) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(&txn).await?;

        while let Some(array) = blocks.next().await {
            if !array?.all() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(&self, txn: Txn) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(&txn).await?;
        while let Some(array) = blocks.next().await {
            if array?.any() {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn not(&self) -> TCResult<Self::Unary> {
        let blocks = BlockListUnary {
            source: self.blocks.clone(),
            transform: Array::not,
            value_transform: Number::not,
            dtype: NumberType::Bool,
        };

        Ok(DenseTensor { blocks })
    }
}

#[async_trait]
impl<T: Clone + BlockList, OT: Clone + BlockList> TensorCompare<DenseTensor<OT>>
    for DenseTensor<T>
{
    type Compare = DenseTensor<BlockListCombine<T, OT>>;
    type Dense = DenseTensor<BlockListCombine<T, OT>>;

    async fn eq(&self, other: &DenseTensor<OT>, _txn: Txn) -> TCResult<Self::Dense> {
        self.combine(
            other,
            Array::eq,
            <Number as NumberInstance>::eq,
            NumberType::Bool,
        )
    }

    fn gt(&self, other: &DenseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(
            other,
            Array::gt,
            <Number as NumberInstance>::gt,
            NumberType::Bool,
        )
    }

    async fn gte(&self, other: &DenseTensor<OT>, _txn: Txn) -> TCResult<Self::Dense> {
        self.combine(
            other,
            Array::gte,
            <Number as NumberInstance>::gte,
            NumberType::Bool,
        )
    }

    fn lt(&self, other: &DenseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(
            other,
            Array::lt,
            <Number as NumberInstance>::lt,
            NumberType::Bool,
        )
    }

    async fn lte(&self, other: &DenseTensor<OT>, _txn: Txn) -> TCResult<Self::Dense> {
        self.combine(
            other,
            Array::lte,
            <Number as NumberInstance>::lte,
            NumberType::Bool,
        )
    }

    fn ne(&self, other: &DenseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(
            other,
            Array::ne,
            <Number as NumberInstance>::ne,
            NumberType::Bool,
        )
    }
}

impl<T: Clone + BlockList, OT: Clone + BlockList> TensorMath<DenseTensor<OT>> for DenseTensor<T> {
    type Combine = DenseTensor<BlockListCombine<T, OT>>;

    fn add(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, Array::add, <Number as NumberInstance>::add, dtype)
    }

    fn multiply(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
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
impl<T: Clone + BlockList> TensorIO for DenseTensor<T> {
    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.blocks.read_value_at(txn, coord).await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        self.blocks.clone().write_value(txn_id, bounds, value).await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        self.blocks.write_value_at(txn_id, coord, value).await
    }
}

#[async_trait]
impl<T: Clone + BlockList, OT: Clone + BlockList> TensorDualIO<DenseTensor<OT>> for DenseTensor<T> {
    async fn mask(&self, _txn: &Txn, _other: DenseTensor<OT>) -> TCResult<()> {
        Err(error::not_implemented("DenseTensor::mask"))
    }

    async fn write(&self, txn: Txn, bounds: Bounds, other: DenseTensor<OT>) -> TCResult<()> {
        let slice = self.slice(bounds)?;
        let other = other
            .broadcast(slice.shape().clone())?
            .as_type(self.dtype())?;

        let coords = stream::iter(Bounds::all(slice.shape()).affected().map(TCResult::Ok));
        let values: TCTryStream<(Vec<u64>, Number)> =
            Box::pin(ValueReader::new(coords, &txn, &other.blocks));

        values
            .map_ok(|(coord, value)| slice.write_value_at(*txn.id(), coord, value))
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(())
    }
}

#[async_trait]
impl<T: Clone + BlockList> TensorDualIO<Tensor> for DenseTensor<T> {
    async fn mask(&self, txn: &Txn, other: Tensor) -> TCResult<()> {
        match other {
            Tensor::Sparse(sparse) => self.mask(txn, sparse.into_dense()).await,
            Tensor::Dense(dense) => self.mask(txn, dense).await,
        }
    }

    async fn write(&self, txn: Txn, bounds: Bounds, other: Tensor) -> TCResult<()> {
        match other {
            Tensor::Sparse(sparse) => self.write(txn, bounds, sparse.into_dense()).await,
            Tensor::Dense(dense) => self.write(txn, bounds, dense).await,
        }
    }
}

impl<T: Clone + BlockList> TensorReduce for DenseTensor<T> {
    type Reduce = DenseTensor<BlockListReduce<T>>;

    fn product(&self, _axis: usize) -> TCResult<Self::Reduce> {
        Err(error::not_implemented("DenseTensor::product"))
    }

    fn product_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let blocks = self.blocks.block_stream(&txn).await?;

            let mut block_products = blocks.map_ok(|array| array.product());

            let zero = self.dtype().zero();
            let mut product = self.dtype().one();
            while let Some(block_product) = block_products.try_next().await? {
                if block_product == zero {
                    return Ok(zero);
                }

                product = product * block_product;
            }

            Ok(product)
        })
    }

    fn sum(&self, _axis: usize) -> TCResult<Self::Reduce> {
        Err(error::not_implemented("DenseTensor::sum"))
    }

    fn sum_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let blocks = self.blocks.block_stream(&txn).await?;

            blocks
                .map_ok(|array| array.sum())
                .try_fold(self.dtype().zero(), |sum, block_sum| {
                    future::ready(Ok(sum + block_sum))
                })
                .await
        })
    }
}

impl<T: Clone + BlockList> TensorTransform for DenseTensor<T> {
    type Cast = DenseTensor<BlockListCast<T>>;
    type Broadcast = DenseTensor<BlockListBroadcast<T>>;
    type Expand = DenseTensor<BlockListExpand<T>>;
    type Slice = DenseTensor<BlockListSlice<T>>;
    type Transpose = DenseTensor<BlockListTranspose<T>>;

    fn as_type(&self, dtype: NumberType) -> TCResult<Self::Cast> {
        let blocks = BlockListCast {
            source: self.blocks.clone(),
            dtype,
        };

        Ok(DenseTensor { blocks })
    }

    fn broadcast(&self, shape: Shape) -> TCResult<Self::Broadcast> {
        let rebase = transform::Broadcast::new(self.shape().clone(), shape)?;
        let blocks = BlockListBroadcast {
            source: self.blocks.clone(),
            rebase,
        };

        Ok(DenseTensor { blocks })
    }

    fn expand_dims(&self, axis: usize) -> TCResult<Self::Expand> {
        let rebase = transform::Expand::new(self.shape().clone(), axis)?;
        let blocks = BlockListExpand {
            source: self.blocks.clone(),
            rebase,
        };

        Ok(DenseTensor { blocks })
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Self::Slice> {
        let rebase = transform::Slice::new(self.shape().clone(), bounds)?;
        let blocks = BlockListSlice {
            source: self.blocks.clone(),
            rebase,
        };

        Ok(DenseTensor { blocks })
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let rebase = transform::Transpose::new(self.shape().clone(), permutation)?;
        let blocks = BlockListTranspose {
            source: self.blocks.clone(),
            rebase,
        };

        Ok(DenseTensor { blocks })
    }
}

impl<T: Clone + BlockList> Route for DenseTensor<T> {
    fn route(
        &'_ self,
        method: MethodType,
        path: &'_ [PathSegment],
    ) -> Option<Box<dyn Handler + '_>> {
        super::handlers::route(self, method, path)
    }
}

#[async_trait]
impl<T: Clone + BlockList> Transact for DenseTensor<T> {
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

impl<T: Clone + BlockList> From<T> for DenseTensor<T> {
    fn from(blocks: T) -> DenseTensor<T> {
        DenseTensor { blocks }
    }
}

impl<T: Clone + BlockList> From<DenseTensor<T>> for Collection {
    fn from(dense: DenseTensor<T>) -> Collection {
        Collection::Tensor(Tensor::Dense(dense.into_dyn()))
    }
}

pub async fn dense_constant(
    txn: &Txn,
    shape: Shape,
    value: Number,
) -> TCResult<DenseTensor<BlockListFile>> {
    let blocks = BlockListFile::constant(txn, shape, value).await?;
    Ok(DenseTensor { blocks })
}

pub fn from_sparse<T: Clone + SparseAccess>(
    sparse: SparseTensor<T>,
) -> DenseTensor<BlockListSparse<T>> {
    let blocks = BlockListSparse::new(sparse);
    DenseTensor { blocks }
}
