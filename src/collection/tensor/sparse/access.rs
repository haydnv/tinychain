use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;

use crate::class::{Instance, TCBoxTryFuture, TCResult, TCTryStream};
use crate::error;
use crate::scalar::value::number::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::super::bounds::*;
use super::super::dense::{sort_coords, BlockList, DenseTensor};
use super::super::transform;
use super::{
    SparseCombine, SparseStream, SparseTable, SparseTensor, TensorAccessor, TensorIO,
    TensorTransform, ERR_NONBIJECTIVE_WRITE,
};

#[async_trait]
pub trait SparseAccess: TensorAccessor + Transact + 'static {
    fn copy<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, SparseTable> {
        Box::pin(async move {
            let accessor = SparseTable::create(&txn, self.shape().clone(), self.dtype()).await?;

            let txn_id = txn.id().clone();
            self.filled(txn)
                .await?
                .map_ok(|(coord, value)| accessor.write_value(txn_id, coord, value))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await?;

            Ok(accessor)
        })
    }

    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream>;

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>>;

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64>;

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream>;

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number>;

    fn read_value_owned<'a>(
        self: Arc<Self>,
        txn: Txn,
        coord: Vec<u64>,
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move { self.read_value(&txn, &coord).await })
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()>;
}

#[derive(Clone)]
pub struct SparseAccessorDyn {
    source: Arc<dyn SparseAccess>,
}

impl SparseAccessorDyn {
    pub fn new<T: SparseAccess>(source: T) -> Self {
        Self {
            source: Arc::new(source),
        }
    }
}

impl TensorAccessor for SparseAccessorDyn {
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
impl SparseAccess for SparseAccessorDyn {
    fn copy<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, SparseTable> {
        self.source.clone().copy(txn)
    }

    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        self.source.clone().filled(txn).await
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        self.source.clone().filled_at(txn, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.source.clone().filled_count(txn).await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        self.source.clone().filled_in(txn, bounds).await
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.source.read_value(txn, coord).await
    }

    fn read_value_owned<'a>(
        self: Arc<Self>,
        txn: Txn,
        coord: Vec<u64>,
    ) -> TCBoxTryFuture<'a, Number> {
        self.source.clone().read_value_owned(txn, coord)
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        self.source.write_value(txn_id, coord, value).await
    }
}

#[async_trait]
impl Transact for SparseAccessorDyn {
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
pub struct DenseAccessor<T: Clone + BlockList> {
    source: DenseTensor<T>,
}

impl<T: Clone + BlockList> DenseAccessor<T> {
    pub fn new(source: DenseTensor<T>) -> Self {
        Self { source }
    }
}

impl<T: Clone + BlockList> TensorAccessor for DenseAccessor<T> {
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
impl<T: Clone + BlockList> SparseAccess for DenseAccessor<T> {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let source = self.source.clone();
        let values = source.value_stream(txn).await?;

        let zero = self.dtype().zero();
        let filled = stream::iter(Bounds::all(self.shape()).affected())
            .zip(values)
            .map(|(coord, r)| r.map(|value| (coord, value)))
            .try_filter(move |(_, value)| future::ready(value != &zero));

        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at(
        self: Arc<Self>,
        _txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        let shape = self.shape();
        let filled_at = stream::iter(
            Bounds::all(&Shape::from(
                axes.iter().map(|x| shape[*x]).collect::<Vec<u64>>(),
            ))
            .affected(),
        )
        .map(Ok);

        let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
        Ok(filled_at)
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.source
            .value_stream(txn)
            .await?
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        match self.source.slice(bounds) {
            Ok(source) => {
                let slice = Arc::new(DenseAccessor { source });
                slice.filled(txn).await
            }
            Err(cause) => Err(cause),
        }
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.source.read_value(txn, coord).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        self.source.write_value(txn_id, coord.into(), value).await
    }
}

#[async_trait]
impl<T: Clone + BlockList> Transact for DenseAccessor<T> {
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
pub struct SparseBroadcast {
    source: Arc<dyn SparseAccess>,
    rebase: transform::Broadcast,
}

impl SparseBroadcast {
    pub fn new(source: Arc<dyn SparseAccess>, rebase: transform::Broadcast) -> Self {
        Self { source, rebase }
    }

    async fn broadcast_coords<S: Stream<Item = TCResult<Vec<u64>>> + Send + Unpin + 'static>(
        self: Arc<Self>,
        txn: Txn,
        coords: S,
        num_coords: u64,
    ) -> TCResult<SparseStream> {
        let coords = sort_coords(
            txn.subcontext_tmp().await?,
            coords,
            num_coords,
            self.shape(),
        )
        .await?;
        let coords = coords.and_then(move |coord| {
            self.clone()
                .read_value_owned(txn.clone(), coord.to_vec())
                .map_ok(|value| (coord, value))
        });
        let coords: SparseStream = Box::pin(coords);
        Ok(coords)
    }
}

impl TensorAccessor for SparseBroadcast {
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
impl SparseAccess for SparseBroadcast {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let rebase = self.rebase.clone();
        let filled = self
            .source
            .clone()
            .filled(txn.clone())
            .await?
            .map_ok(move |(coord, _)| stream::iter(rebase.map_coord(coord).map(TCResult::Ok)))
            .try_flatten();

        let num_coords = self.clone().filled_count(txn.clone()).await?;
        self.broadcast_coords(txn, filled, num_coords).await
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        let filled = self.source.clone().filled(txn).await?;
        let rebase = self.rebase.clone();
        filled
            .try_fold(0u64, |count, (coord, _)| {
                future::ready(Ok(count + rebase.map_bounds(coord.into()).size()))
            })
            .await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let source_bounds = self.rebase.invert_bounds(bounds);
        let source_filled_in1 = self
            .source
            .clone()
            .filled_in(txn.clone(), source_bounds.clone());
        let source_filled_in2 = self.source.clone().filled_in(txn.clone(), source_bounds);
        let (source_filled_in1, source_filled_in2) =
            try_join!(source_filled_in1, source_filled_in2)?;

        let rebase = self.rebase.clone();
        let filled_in = source_filled_in1
            .map_ok(move |(coord, _)| stream::iter(rebase.map_coord(coord).map(TCResult::Ok)))
            .try_flatten();

        let rebase = self.rebase.clone();
        let num_coords = source_filled_in2
            .try_fold(0u64, |count, (coord, _)| {
                future::ready(Ok(count + rebase.map_bounds(coord.into()).size()))
            })
            .await?;

        self.broadcast_coords(txn, filled_in, num_coords).await
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value(txn, &coord).await
    }

    async fn write_value(&self, _txn_id: TxnId, _coord: Vec<u64>, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

#[async_trait]
impl Transact for SparseBroadcast {
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
pub struct SparseCast {
    source: Arc<dyn SparseAccess>,
    dtype: NumberType,
}

impl SparseCast {
    pub fn new(source: Arc<dyn SparseAccess>, dtype: NumberType) -> Self {
        Self { source, dtype }
    }
}

impl TensorAccessor for SparseCast {
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
impl SparseAccess for SparseCast {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let dtype = self.dtype;
        let filled = self.source.clone().filled(txn).await?;
        let cast = filled.map_ok(move |(coord, value)| (coord, value.into_type(dtype)));
        let cast: SparseStream = Box::pin(cast);
        Ok(cast)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        self.source.clone().filled_at(txn, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.source.clone().filled_count(txn).await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let dtype = self.dtype;
        let source = self.source.clone().filled_in(txn, bounds).await?;
        let filled_in = source.map_ok(move |(coord, value)| (coord, value.into_type(dtype)));
        let filled_in: SparseStream = Box::pin(filled_in);
        Ok(filled_in)
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let dtype = self.dtype;
        self.source
            .read_value(txn, coord)
            .map_ok(move |value| value.into_type(dtype))
            .await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        self.source.write_value(txn_id, coord, value).await
    }
}

#[async_trait]
impl Transact for SparseCast {
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
pub struct SparseCombinator {
    left: Arc<dyn SparseAccess>,
    left_zero: Number,
    right: Arc<dyn SparseAccess>,
    right_zero: Number,
    combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
}

impl SparseCombinator {
    pub fn new(
        left: Arc<dyn SparseAccess>,
        right: Arc<dyn SparseAccess>,
        combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<SparseCombinator> {
        if left.shape() != right.shape() {
            return Err(error::internal(
                "Tried to combine SparseTensors with different shapes",
            ));
        }

        let left_zero = left.dtype().zero();
        let right_zero = right.dtype().zero();
        Ok(SparseCombinator {
            left,
            left_zero,
            right,
            right_zero,
            combinator,
            dtype,
        })
    }

    fn filled_inner(&self, left: SparseStream, right: SparseStream) -> SparseStream {
        let combinator = self.combinator;
        let left_zero = self.left_zero.clone();
        let right_zero = self.right_zero.clone();

        let combined =
            SparseCombine::new(self.shape(), left, right).try_filter_map(move |(coord, l, r)| {
                let l = l.unwrap_or_else(|| left_zero.clone());
                let r = r.unwrap_or_else(|| right_zero.clone());
                let value = combinator(l, r);
                let row = if value == value.class().zero() {
                    None
                } else {
                    Some((coord, value))
                };

                future::ready(Ok(row))
            });

        Box::pin(combined)
    }
}

impl TensorAccessor for SparseCombinator {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.left.ndim() + 1
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.left.shape()
    }

    fn size(&self) -> u64 {
        self.left.size()
    }
}

#[async_trait]
impl SparseAccess for SparseCombinator {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let left = self.left.clone().filled(txn.clone());
        let right = self.right.clone().filled(txn);
        let (left, right) = try_join!(left, right)?;
        Ok(self.filled_inner(left, right))
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        let count = self
            .filled(txn)
            .await?
            .fold(0u64, |count, _| future::ready(count + 1))
            .await;

        Ok(count)
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let left = self.left.clone().filled_in(txn.clone(), bounds.clone());
        let right = self.right.clone().filled_in(txn, bounds);
        let (left, right) = try_join!(left, right)?;
        Ok(self.filled_inner(left, right))
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let left = self.left.read_value(txn, coord);
        let right = self.right.read_value(txn, coord);
        let (left, right) = try_join!(left, right)?;
        let combinator = self.combinator;
        Ok(combinator(left, right))
    }

    async fn write_value(&self, _txn_id: TxnId, _coord: Vec<u64>, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

#[async_trait]
impl Transact for SparseCombinator {
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
pub struct SparseExpand {
    source: Arc<dyn SparseAccess>,
    rebase: transform::Expand,
}

impl SparseExpand {
    pub fn new(source: Arc<dyn SparseAccess>, rebase: transform::Expand) -> Self {
        Self { source, rebase }
    }
}

impl TensorAccessor for SparseExpand {
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
impl SparseAccess for SparseExpand {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let filled = self
            .source
            .clone()
            .filled(txn)
            .await?
            .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.source.clone().filled_count(txn).await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let bounds = self.rebase.invert_bounds(bounds);
        let filled_in = self
            .source
            .clone()
            .filled_in(txn, bounds)
            .await?
            .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

        let filled_in: SparseStream = Box::pin(filled_in);
        Ok(filled_in)
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value(txn, &coord).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value(txn_id, coord, value).await
    }
}

#[async_trait]
impl Transact for SparseExpand {
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

type Reductor = fn(&SparseTensor<SparseSlice>, Txn) -> TCBoxTryFuture<Number>;

#[derive(Clone)]
pub struct SparseReduce<T: Clone + SparseAccess> {
    source: SparseTensor<T>,
    rebase: transform::Reduce,
    reductor: Reductor,
}

impl<T: Clone + SparseAccess> SparseReduce<T> {
    pub fn new(
        source: SparseTensor<T>,
        axis: usize,
        reductor: Reductor,
    ) -> TCResult<SparseReduce<T>> {
        transform::Reduce::new(source.shape().clone(), axis).map(|rebase| SparseReduce {
            source,
            rebase,
            reductor,
        })
    }
}

impl<T: Clone + SparseAccess> TensorAccessor for SparseReduce<T> {
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
impl<T: Clone + SparseAccess> SparseAccess for SparseReduce<T> {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let reductor = self.reductor;
        let source = self.source.clone();

        let filled = self
            .clone()
            .filled_at(txn.clone(), (0..self.ndim()).collect())
            .await?
            .and_then(move |coord| {
                let txn = txn.clone();
                let source = source.clone();
                let source_bounds = self.rebase.invert_coord(&coord);
                Box::pin(
                    async move { Ok((coord, reductor(&source.slice(source_bounds)?, txn).await?)) },
                )
            });

        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        let reduce_axis = self.rebase.axis();

        if axes.is_empty() {
            let filled_at: TCTryStream<Vec<u64>> = Box::pin(stream::empty());
            return Ok(filled_at);
        } else if axes.iter().cloned().fold(axes[0], Ord::max) < reduce_axis {
            return self.source.clone().filled_at(txn, axes).await;
        }

        let source_axes: Vec<usize> = axes
            .iter()
            .cloned()
            .map(|x| if x < reduce_axis { x } else { x + 1 })
            .chain(iter::once(reduce_axis))
            .collect();

        let left = self
            .source
            .clone()
            .filled_at(txn.clone(), source_axes.to_vec())
            .await?;
        let mut right = self.source.clone().filled_at(txn, source_axes).await?;

        if right.next().await.is_none() {
            let filled_at: TCTryStream<Vec<u64>> = Box::pin(stream::empty());
            return Ok(filled_at);
        }

        let filled_at = left
            .zip(right)
            .map(|(lr, rr)| Ok((lr?, rr?)))
            .map_ok(|(mut l, mut r)| {
                l.pop();
                r.pop();
                (l, r)
            })
            .try_filter_map(|(l, r)| {
                let row = if l == r { None } else { Some(l) };
                future::ready(Ok(row))
            });

        let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
        Ok(filled_at)
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.filled(txn)
            .await?
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let (source_bounds, slice_reduce_axis) = self.rebase.invert_bounds(bounds);
        let slice = self.source.slice(source_bounds)?;
        Arc::new(SparseReduce::new(slice, slice_reduce_axis, self.reductor)?)
            .filled(txn)
            .await
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let source_bounds = self.rebase.invert_coord(coord);
        let reductor = self.reductor;
        reductor(&self.source.slice(source_bounds)?, txn.clone()).await
    }

    async fn write_value(&self, _txn_id: TxnId, _coord: Vec<u64>, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> Transact for SparseReduce<T> {
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
pub struct SparseReshape {
    source: Arc<dyn SparseAccess>,
    rebase: transform::Reshape,
}

impl SparseReshape {
    pub fn new(source: Arc<dyn SparseAccess>, rebase: transform::Reshape) -> Self {
        Self { source, rebase }
    }
}

impl TensorAccessor for SparseReshape {
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
impl SparseAccess for SparseReshape {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let rebase = self.rebase.clone();
        let filled = self
            .source
            .clone()
            .filled(txn)
            .await?
            .map_ok(move |(coord, value)| (rebase.map_coord(coord), value));

        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.source.clone().filled_count(txn).await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        if self.source.ndim() == 1 {
            let (start, end) = self.rebase.offsets(&bounds);

            let rebase = transform::Slice::new(
                self.source.shape().clone(),
                vec![AxisBounds::from(start..end)].into(),
            )?;

            let slice = Arc::new(SparseSlice {
                source: self.source.clone(),
                rebase,
            });

            let rebase = self.rebase.clone();
            let filled = slice
                .filled(txn)
                .await?
                .map_ok(move |(coord, value)| (rebase.map_coord(coord), value))
                .try_filter(move |(coord, _)| future::ready(bounds.contains_coord(coord)));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        } else {
            let rebase = transform::Reshape::new(
                self.source.shape().clone(),
                vec![self.source.size()].into(),
            )?;
            let flat = Arc::new(SparseReshape {
                source: self.source.clone(),
                rebase,
            });

            let rebase = transform::Reshape::new(flat.shape().clone(), self.shape().clone())?;
            let unflat = Arc::new(SparseReshape {
                source: flat,
                rebase,
            });
            unflat.filled_in(txn, bounds).await
        }
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value(txn, &coord).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value(txn_id, coord, value).await
    }
}

#[async_trait]
impl Transact for SparseReshape {
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
pub struct SparseSlice {
    source: Arc<dyn SparseAccess>,
    rebase: transform::Slice,
}

impl SparseSlice {
    pub fn new(source: Arc<dyn SparseAccess>, rebase: transform::Slice) -> Self {
        Self { source, rebase }
    }
}

impl TensorAccessor for SparseSlice {
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
impl SparseAccess for SparseSlice {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        debug!(
            "SparseSlice::filled, source bounds: {}",
            self.rebase.bounds()
        );

        let rebase = self.rebase.clone();
        let filled = self
            .source
            .clone()
            .filled_in(txn, rebase.bounds().clone())
            .await?
            .inspect_ok(|(coord, value)| debug!("source coord: {:?} = {}", coord, value))
            .map_ok(move |(coord, value)| (rebase.map_coord(coord), value));

        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        let count = self
            .filled(txn)
            .await?
            .fold(0u64, |count, _| future::ready(count + 1))
            .await;

        Ok(count)
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let bounds = self.rebase.invert_bounds(bounds);
        let filled_in = self
            .source
            .clone()
            .filled_in(txn, bounds)
            .await?
            .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

        let filled_in: SparseStream = Box::pin(filled_in);
        Ok(filled_in)
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value(txn, &coord).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value(txn_id, coord, value).await
    }
}

#[async_trait]
impl Transact for SparseSlice {
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
pub struct SparseTranspose {
    source: Arc<dyn SparseAccess>,
    rebase: transform::Transpose,
}

impl SparseTranspose {
    pub fn new(source: Arc<dyn SparseAccess>, rebase: transform::Transpose) -> Self {
        Self { source, rebase }
    }
}

impl TensorAccessor for SparseTranspose {
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
impl SparseAccess for SparseTranspose {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let ndim = self.ndim();
        let this = self.clone();
        let filled = self
            .filled_at(txn.clone(), (0..ndim).collect())
            .await?
            .and_then(move |coord| {
                this.clone()
                    .read_value_owned(txn.clone(), coord.to_vec())
                    .map_ok(|value| (coord, value))
            });

        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        // can't use group_axes here because it would lead to a circular dependency in self.filled
        let rebase = self.rebase.clone();
        let source_axes = rebase.invert_axes(&axes);
        let filled_at = self
            .source
            .clone()
            .filled_at(txn, source_axes.to_vec())
            .await?
            .map_ok(move |coord| rebase.map_coord_axes(coord, &source_axes));

        let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
        Ok(filled_at)
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.source.clone().filled_count(txn).await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let filled_in = self
            .source
            .clone()
            .filled_in(txn, self.rebase.invert_bounds(bounds))
            .await?
            .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

        let filled_in: SparseStream = Box::pin(filled_in);
        Ok(filled_in)
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let coord = self.rebase.invert_coord(coord);
        self.source.read_value(txn, &coord).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        self.source
            .write_value(txn_id, self.rebase.invert_coord(&coord), value)
            .await
    }
}

#[async_trait]
impl Transact for SparseTranspose {
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

fn group_axes<'a>(
    accessor: Arc<dyn SparseAccess>,
    txn: Txn,
    axes: Vec<usize>,
) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
    Box::pin(async move {
        if axes.len() > accessor.ndim() {
            let axes: Vec<String> = axes.iter().map(|x| x.to_string()).collect();
            return Err(error::bad_request("Axis out of bounds", axes.join(", ")));
        }

        let axes_clone = axes.to_vec();
        let map = move |(coord, _): (Vec<u64>, Number)| {
            axes_clone.iter().map(|x| coord[*x]).collect::<Vec<u64>>()
        };

        let sorted_axes: Vec<usize> = itertools::sorted(axes.to_vec()).collect::<Vec<usize>>();
        if axes == sorted_axes {
            let left = accessor
                .clone()
                .filled(txn.clone())
                .await?
                .map_ok(map.clone());
            let mut right = accessor.clone().filled(txn.clone()).await?.map_ok(map);

            if right.next().await.is_none() {
                let filled_at: TCTryStream<Vec<u64>> = Box::pin(stream::empty());
                return Ok(filled_at);
            }

            let filled_at = left
                .zip(right)
                .map(|(lr, rr)| Ok((lr?, rr?)))
                .try_filter_map(|(l, r)| {
                    if l == r {
                        future::ready(Ok(Some(l)))
                    } else {
                        future::ready(Ok(None))
                    }
                });

            let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        } else {
            let num_coords = accessor
                .clone()
                .filled_at(txn.clone(), sorted_axes.to_vec())
                .await?
                .try_fold(0, |count, _| future::ready(Ok(count + 1)))
                .await?;

            let coords = accessor
                .clone()
                .filled_at(txn.clone(), sorted_axes)
                .await?
                .map_ok(move |coord| axes.iter().map(|x| coord[*x]).collect::<Vec<u64>>());

            let filled_at = sort_coords(
                txn.subcontext_tmp().await?,
                coords,
                num_coords,
                accessor.shape(),
            )
            .await?;
            let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        }
    })
}
