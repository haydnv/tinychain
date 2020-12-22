use std::iter;
use std::pin::Pin;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;

use crate::class::Instance;
use crate::error;
use crate::general::{count_stream, TCBoxTryFuture, TCResult};
use crate::scalar::value::number::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::super::bounds::*;
use super::super::dense::{DenseAccess, DenseTensor};
use super::super::stream::*;
use super::super::transform;
use super::{
    SparseCombine, SparseStream, SparseTable, SparseTensor, TensorAccessor, TensorIO,
    TensorTransform, ERR_NONBIJECTIVE_WRITE,
};

pub type CoordStream<'a> = Pin<Box<dyn Stream<Item = TCResult<Vec<u64>>> + Send + Unpin + 'a>>;

#[async_trait]
pub trait SparseAccess: TensorAccessor + Transact + 'static {
    fn copy<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, SparseTable> {
        Box::pin(async move {
            let accessor = SparseTable::create(txn, self.shape().clone(), self.dtype()).await?;

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

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>>;

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'_>>;

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64>;

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>>;

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number>;

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()>;
}

fn read_value_at<'a, T: SparseAccess>(source: &'a T, txn: &'a Txn, coord: Vec<u64>) -> Read<'a> {
    Box::pin(async move {
        let value = source.read_value(txn, &coord).await?;
        Ok((coord, value))
    })
}

#[derive(Clone)]
pub struct SparseAccessorDyn {
    source: std::sync::Arc<dyn SparseAccess>,
}

impl SparseAccessorDyn {
    pub fn new<T: Clone + SparseAccess>(source: T) -> Self {
        Self {
            source: std::sync::Arc::new(source),
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
    fn copy<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, SparseTable> {
        self.source.copy(txn)
    }

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        self.source.filled(txn).await
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        self.source.filled_in(txn, bounds).await
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.source.read_value(txn, coord).await
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
pub struct DenseToSparse<T: Clone + DenseAccess> {
    source: DenseTensor<T>,
}

impl<T: Clone + DenseAccess> DenseToSparse<T> {
    pub fn new(source: DenseTensor<T>) -> Self {
        Self { source }
    }
}

impl<T: Clone + DenseAccess> TensorAccessor for DenseToSparse<T> {
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
impl<T: Clone + DenseAccess> SparseAccess for DenseToSparse<T> {
    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let values = self.source.value_stream(txn).await?;

        let zero = self.dtype().zero();
        let filled = stream::iter(Bounds::all(self.shape()).affected())
            .zip(values)
            .map(|(coord, r)| r.map(|value| (coord, value)))
            .try_filter(move |(_, value)| future::ready(value != &zero));

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(&'a self, _txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        let shape = self.shape();
        let filled_at = stream::iter(
            Bounds::all(&Shape::from(
                axes.iter().map(|x| shape[*x]).collect::<Vec<u64>>(),
            ))
            .affected(),
        )
        .map(Ok);

        Ok(Box::pin(filled_at))
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source
            .value_stream(txn)
            .await?
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn filled_in<'a>(&'a self, _txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        match self.source.slice(bounds) {
            Ok(_source) => Err(error::not_implemented("DenseAccessor::filled_in")),
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
impl<T: Clone + DenseAccess> Transact for DenseToSparse<T> {
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
pub struct SparseBroadcast<T: Clone + SparseAccess> {
    source: T,
    rebase: transform::Broadcast,
}

impl<T: Clone + SparseAccess> SparseBroadcast<T> {
    pub fn new(source: T, rebase: transform::Broadcast) -> Self {
        Self { source, rebase }
    }

    async fn broadcast_coords<'a, S: Stream<Item = TCResult<Vec<u64>>> + 'a + Send + Unpin + 'a>(
        &'a self,
        txn: &'a Txn,
        coords: S,
        num_coords: u64,
    ) -> TCResult<SparseStream<'a>> {
        let broadcast = sorted_values(txn, self, coords, num_coords).await?;
        Ok(Box::pin(broadcast))
    }
}

impl<T: Clone + SparseAccess> TensorAccessor for SparseBroadcast<T> {
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
impl<T: Clone + SparseAccess> SparseAccess for SparseBroadcast<T> {
    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let rebase = self.rebase.clone();

        let num_coords = self.source.filled_count(txn).await?;

        let filled = self
            .source
            .filled(txn)
            .await?
            .map_ok(move |(coord, _)| stream::iter(rebase.map_coord(coord).map(TCResult::Ok)))
            .try_flatten();

        self.broadcast_coords(txn, filled, num_coords).await
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        let filled = self.source.filled(txn).await?;
        let rebase = self.rebase.clone();
        filled
            .try_fold(0u64, |count, (coord, _)| {
                future::ready(Ok(count + rebase.map_bounds(coord.into()).size()))
            })
            .await
    }

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        let source_bounds = self.rebase.invert_bounds(bounds);
        let source_filled_in1 = self.source.filled_in(txn, source_bounds.clone());
        let source_filled_in2 = self.source.filled_in(txn, source_bounds);
        let (source_filled_in1, source_filled_in2) =
            try_join!(source_filled_in1, source_filled_in2)?;

        let rebase = self.rebase.clone();

        let num_coords = count_stream(source_filled_in1).await?;

        let filled_in = source_filled_in2
            .map_ok(move |(coord, _)| stream::iter(rebase.map_coord(coord).map(TCResult::Ok)))
            .try_flatten();

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

impl<T: Clone + SparseAccess> ReadValueAt for SparseBroadcast<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a> {
        read_value_at(self, txn, coord)
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> Transact for SparseBroadcast<T> {
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
pub struct SparseCast<T: Clone + SparseAccess> {
    source: T,
    dtype: NumberType,
}

impl<T: Clone + SparseAccess> SparseCast<T> {
    pub fn new(source: T, dtype: NumberType) -> Self {
        Self { source, dtype }
    }
}

impl<T: Clone + SparseAccess> TensorAccessor for SparseCast<T> {
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
impl<T: Clone + SparseAccess> SparseAccess for SparseCast<T> {
    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let dtype = self.dtype;

        let filled = self.source.filled(txn).await?;
        let cast = filled.map_ok(move |(coord, value)| (coord, value.into_type(dtype)));
        Ok(Box::pin(cast))
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        let dtype = self.dtype;
        let source = self.source.filled_in(txn, bounds).await?;
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
impl<T: Clone + SparseAccess> Transact for SparseCast<T> {
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
pub struct SparseCombinator<L: Clone + SparseAccess, R: Clone + SparseAccess> {
    left: L,
    left_zero: Number,
    right: R,
    right_zero: Number,
    combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
}

impl<L: Clone + SparseAccess, R: Clone + SparseAccess> SparseCombinator<L, R> {
    pub fn new(
        left: L,
        right: R,
        combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<Self> {
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

    fn filled_inner<'a>(
        &'a self,
        left: SparseStream<'a>,
        right: SparseStream<'a>,
    ) -> SparseStream<'a> {
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

impl<L: Clone + SparseAccess, R: Clone + SparseAccess> TensorAccessor for SparseCombinator<L, R> {
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
impl<L: Clone + SparseAccess, R: Clone + SparseAccess> SparseAccess for SparseCombinator<L, R> {
    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let left = self.left.filled(txn);
        let right = self.right.filled(txn);
        let (left, right) = try_join!(left, right)?;
        Ok(self.filled_inner(left, right))
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        let count = self
            .filled(txn)
            .await?
            .fold(0u64, |count, _| future::ready(count + 1))
            .await;

        Ok(count)
    }

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        let left = self.left.filled_in(txn, bounds.clone());
        let right = self.right.filled_in(txn, bounds);
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
impl<L: Clone + SparseAccess, R: Clone + SparseAccess> Transact for SparseCombinator<L, R> {
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
pub struct SparseExpand<T: Clone + SparseAccess> {
    source: T,
    rebase: transform::Expand,
}

impl<T: Clone + SparseAccess> SparseExpand<T> {
    pub fn new(source: T, rebase: transform::Expand) -> Self {
        Self { source, rebase }
    }
}

impl<T: Clone + SparseAccess> TensorAccessor for SparseExpand<T> {
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
impl<T: Clone + SparseAccess> SparseAccess for SparseExpand<T> {
    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let filled = self
            .source
            .filled(txn)
            .await?
            .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        let bounds = self.rebase.invert_bounds(bounds);
        let filled_in = self
            .source
            .filled_in(txn, bounds)
            .await?
            .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

        Ok(Box::pin(filled_in))
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
impl<T: Clone + SparseAccess> Transact for SparseExpand<T> {
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

type Reductor = fn(&SparseTensor<SparseAccessorDyn>, Txn) -> TCBoxTryFuture<Number>;

#[derive(Clone)]
pub struct SparseReduce<T: Clone + SparseAccess> {
    source: SparseTensor<T>,
    rebase: transform::Reduce,
    reductor: Reductor,
}

impl<'a, T: Clone + SparseAccess> SparseReduce<T> {
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
    async fn filled<'b>(&'b self, txn: &'b Txn) -> TCResult<SparseStream<'b>> {
        let reductor = self.reductor;
        let source = &self.source;

        let filled = self.filled_at(txn, (0..self.ndim()).collect()).await?;

        let filled = filled.and_then(move |coord| {
            let source_bounds = self.rebase.invert_coord(&coord);
            let txn = txn.clone();
            Box::pin(async move {
                let slice = source.slice(source_bounds)?;
                Ok((coord, reductor(&slice.into_dyn(), txn).await?))
            })
        });

        Ok(Box::pin(filled))
    }

    async fn filled_at<'b>(&'b self, txn: &'b Txn, axes: Vec<usize>) -> TCResult<CoordStream<'b>> {
        let reduce_axis = self.rebase.axis();

        if axes.is_empty() {
            let filled_at: CoordStream<'_> = Box::pin(stream::empty());
            return Ok(filled_at);
        } else if axes.iter().cloned().fold(axes[0], Ord::max) < reduce_axis {
            return self.source.filled_at(txn, axes).await;
        }

        let source_axes: Vec<usize> = axes
            .iter()
            .cloned()
            .map(|x| if x < reduce_axis { x } else { x + 1 })
            .chain(iter::once(reduce_axis))
            .collect();

        let left = self.source.filled_at(txn, source_axes.to_vec()).await?;
        let mut right = self.source.filled_at(txn, source_axes).await?;

        if right.next().await.is_none() {
            let filled_at: CoordStream<'_> = Box::pin(stream::empty());
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

        let filled_at: CoordStream<'_> = Box::pin(filled_at);
        Ok(filled_at)
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.filled(txn)
            .await?
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn filled_in<'b>(&'b self, _txn: &'b Txn, _bounds: Bounds) -> TCResult<SparseStream<'b>> {
        Err(error::not_implemented("SparseReduce::filled_in"))
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let source_bounds = self.rebase.invert_coord(coord);
        let reductor = self.reductor;
        let slice = self.source.slice(source_bounds)?;
        reductor(&slice.into_dyn(), txn.clone()).await
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
pub struct SparseSlice {
    source: SparseAccessorDyn,
    rebase: transform::Slice,
}

impl SparseSlice {
    pub fn new(source: SparseAccessorDyn, rebase: transform::Slice) -> Self {
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
    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        debug!(
            "SparseSlice::filled, source bounds: {}",
            self.rebase.bounds()
        );

        let rebase = self.rebase.clone();
        let filled = self
            .source
            .filled_in(txn, rebase.bounds().clone())
            .await?
            .inspect_ok(|(coord, value)| debug!("source coord: {:?} = {}", coord, value))
            .map_ok(move |(coord, value)| (rebase.map_coord(coord), value));

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        group_axes(self, txn, axes).await
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        let count = self
            .filled(txn)
            .await?
            .fold(0u64, |count, _| future::ready(count + 1))
            .await;

        Ok(count)
    }

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        let bounds = self.rebase.invert_bounds(bounds);
        let filled_in = self
            .source
            .filled_in(txn, bounds)
            .await?
            .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

        Ok(Box::pin(filled_in))
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
pub struct SparseTranspose<T: Clone + SparseAccess> {
    source: T,
    rebase: transform::Transpose,
}

impl<T: Clone + SparseAccess> SparseTranspose<T> {
    pub fn new(source: T, rebase: transform::Transpose) -> Self {
        Self { source, rebase }
    }
}

impl<T: Clone + SparseAccess> TensorAccessor for SparseTranspose<T> {
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
impl<T: Clone + SparseAccess> SparseAccess for SparseTranspose<T> {
    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let ndim = self.ndim();
        let coords = self.filled_at(txn, (0..ndim).collect()).await?;

        Ok(Box::pin(ValueReader::new(coords, txn, self)))
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        // can't use group_axes here because it would lead to a circular dependency in self.filled
        let rebase = self.rebase.clone();
        let source_axes = rebase.invert_axes(&axes);
        let filled_at = self
            .source
            .filled_at(txn, source_axes.to_vec())
            .await?
            .map_ok(move |coord| rebase.map_coord_axes(coord, &source_axes));

        Ok(Box::pin(filled_at))
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        let filled_in = self
            .source
            .filled_in(txn, self.rebase.invert_bounds(bounds))
            .await?
            .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

        Ok(Box::pin(filled_in))
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

impl<T: Clone + SparseAccess> ReadValueAt for SparseTranspose<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a> {
        read_value_at(self, txn, coord)
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> Transact for SparseTranspose<T> {
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

fn group_axes<'a, T: Clone + SparseAccess>(
    accessor: &'a T,
    txn: &'a Txn,
    axes: Vec<usize>,
) -> TCBoxTryFuture<'a, CoordStream<'a>> {
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
            let left = accessor.filled(txn).await?.map_ok(map.clone());
            let mut right = accessor.filled(txn).await?.map_ok(map);

            if right.next().await.is_none() {
                let filled_at: CoordStream<'_> = Box::pin(stream::empty());
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

            let filled_at: CoordStream<'_> = Box::pin(filled_at);
            Ok(filled_at)
        } else {
            let (coords1, coords2) = try_join!(
                accessor.filled_at(txn, sorted_axes.to_vec()),
                accessor.filled_at(txn, sorted_axes)
            )?;

            let num_coords = count_stream(coords1).await?;
            let coords =
                coords2.map_ok(move |coord| axes.iter().map(|x| coord[*x]).collect::<Vec<u64>>());

            let filled_at: CoordStream<'a> =
                sorted_coords(txn, accessor.shape(), coords, num_coords)
                    .map_ok(Box::pin)
                    .await?;
            Ok(filled_at)
        }
    })
}

#[derive(Clone)]
pub struct SparseUnary {
    source: SparseAccessorDyn,
    transform: fn(Number) -> Number,
    dtype: NumberType,
}

impl SparseUnary {
    pub fn new(
        source: SparseAccessorDyn,
        transform: fn(Number) -> Number,
        dtype: NumberType,
    ) -> Self {
        Self {
            source,
            transform,
            dtype,
        }
    }
}

impl TensorAccessor for SparseUnary {
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
impl SparseAccess for SparseUnary {
    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let transform = self.transform;
        let filled = self.source.filled(txn).await?;
        let cast = filled.map_ok(move |(coord, value)| (coord, transform(value)));
        Ok(Box::pin(cast))
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn filled_in<'a>(&'a self, txn: &'a Txn, bounds: Bounds) -> TCResult<SparseStream<'a>> {
        let transform = self.transform;
        let source = self.source.filled_in(txn, bounds).await?;
        let filled_in = source.map_ok(move |(coord, value)| (coord, transform(value)));
        Ok(Box::pin(filled_in))
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let dtype = self.dtype;
        self.source
            .read_value(txn, coord)
            .map_ok(move |value| value.into_type(dtype))
            .await
    }

    async fn write_value(&self, _txn_id: TxnId, _coord: Vec<u64>, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

#[async_trait]
impl Transact for SparseUnary {
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
