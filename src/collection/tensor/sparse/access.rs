use std::iter;
use std::pin::Pin;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;

use crate::class::Instance;
use crate::collection::stream::GroupStream;
use crate::error;
use crate::general::{count_stream, TCBoxTryFuture, TCResult};
use crate::scalar::value::number::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::super::bounds::*;
use super::super::dense::{DenseAccess, DenseTensor};
use super::super::stream::*;
use super::super::transform;
use super::super::Coord;
use super::{
    SparseCombine, SparseStream, SparseTable, SparseTensor, TensorAccessor, TensorIO,
    TensorTransform, ERR_NONBIJECTIVE_WRITE,
};
use crate::collection::tensor::dense::DenseAccessor;

pub type CoordStream<'a> = Pin<Box<dyn Stream<Item = TCResult<Coord>> + Send + Unpin + 'a>>;

#[async_trait]
pub trait SparseAccess: ReadValueAt + TensorAccessor + Transact + 'static {
    fn accessor(self) -> SparseAccessor;

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

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

#[derive(Clone)]
pub enum SparseAccessor {
    Broadcast(Box<SparseBroadcast<SparseAccessor>>),
    Cast(Box<SparseCast<SparseAccessor>>),
    Combine(Box<SparseCombinator<SparseAccessor, SparseAccessor>>),
    Dense(Box<DenseToSparse<DenseAccessor>>),
    Expand(Box<SparseExpand<SparseAccessor>>),
    Slice(Box<SparseSlice>),
    Reduce(Box<SparseReduce<SparseAccessor>>),
    Table(SparseTable),
    Transpose(Box<SparseTranspose<SparseAccessor>>),
    Unary(Box<SparseUnary>),
}

impl TensorAccessor for SparseAccessor {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Broadcast(broadcast) => broadcast.dtype(),
            Self::Cast(cast) => cast.dtype(),
            Self::Combine(combinator) => combinator.dtype(),
            Self::Dense(dense) => dense.dtype(),
            Self::Expand(expand) => expand.dtype(),
            Self::Slice(slice) => slice.dtype(),
            Self::Reduce(reduce) => reduce.dtype(),
            Self::Table(table) => table.dtype(),
            Self::Transpose(transpose) => transpose.dtype(),
            Self::Unary(unary) => unary.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Broadcast(broadcast) => broadcast.ndim(),
            Self::Cast(cast) => cast.ndim(),
            Self::Combine(combinator) => combinator.ndim(),
            Self::Dense(dense) => dense.ndim(),
            Self::Expand(expand) => expand.ndim(),
            Self::Slice(slice) => slice.ndim(),
            Self::Reduce(reduce) => reduce.ndim(),
            Self::Table(table) => table.ndim(),
            Self::Transpose(transpose) => transpose.ndim(),
            Self::Unary(unary) => unary.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Broadcast(broadcast) => broadcast.shape(),
            Self::Cast(cast) => cast.shape(),
            Self::Combine(combinator) => combinator.shape(),
            Self::Dense(dense) => dense.shape(),
            Self::Expand(expand) => expand.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::Reduce(reduce) => reduce.shape(),
            Self::Table(table) => table.shape(),
            Self::Transpose(transpose) => transpose.shape(),
            Self::Unary(unary) => unary.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Broadcast(broadcast) => broadcast.size(),
            Self::Cast(cast) => cast.size(),
            Self::Combine(combinator) => combinator.size(),
            Self::Dense(dense) => dense.size(),
            Self::Expand(expand) => expand.size(),
            Self::Slice(slice) => slice.size(),
            Self::Reduce(reduce) => reduce.size(),
            Self::Table(table) => table.size(),
            Self::Transpose(transpose) => transpose.size(),
            Self::Unary(unary) => unary.size(),
        }
    }
}

#[async_trait]
impl SparseAccess for SparseAccessor {
    fn accessor(self) -> SparseAccessor {
        self
    }

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.filled(txn).await,
            Self::Cast(cast) => cast.filled(txn).await,
            Self::Combine(combinator) => combinator.filled(txn).await,
            Self::Dense(dense) => dense.filled(txn).await,
            Self::Expand(expand) => expand.filled(txn).await,
            Self::Slice(slice) => slice.filled(txn).await,
            Self::Reduce(reduce) => reduce.filled(txn).await,
            Self::Table(table) => table.filled(txn).await,
            Self::Transpose(transpose) => transpose.filled(txn).await,
            Self::Unary(unary) => unary.filled(txn).await,
        }
    }

    async fn filled_at<'a>(
        &'a self,
        _txn: &'a Txn,
        _axes: Vec<usize>,
    ) -> TCResult<CoordStream<'a>> {
        unimplemented!()
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        match self {
            Self::Broadcast(broadcast) => broadcast.filled_count(txn).await,
            Self::Cast(cast) => cast.filled_count(txn).await,
            Self::Combine(combinator) => combinator.filled_count(txn).await,
            Self::Dense(dense) => dense.filled_count(txn).await,
            Self::Expand(expand) => expand.filled_count(txn).await,
            Self::Slice(slice) => slice.filled_count(txn).await,
            Self::Reduce(reduce) => reduce.filled_count(txn).await,
            Self::Table(table) => table.filled_count(txn).await,
            Self::Transpose(transpose) => transpose.filled_count(txn).await,
            Self::Unary(unary) => unary.filled_count(txn).await,
        }
    }

    async fn filled_in<'a>(&'a self, _txn: &'a Txn, _bounds: Bounds) -> TCResult<SparseStream<'a>> {
        unimplemented!()
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        match self {
            Self::Broadcast(broadcast) => broadcast.write_value(txn_id, coord, value).await,
            Self::Cast(cast) => cast.write_value(txn_id, coord, value).await,
            Self::Combine(combinator) => combinator.write_value(txn_id, coord, value).await,
            Self::Dense(dense) => dense.write_value(txn_id, coord, value).await,
            Self::Expand(expand) => expand.write_value(txn_id, coord, value).await,
            Self::Slice(slice) => slice.write_value(txn_id, coord, value).await,
            Self::Reduce(reduce) => reduce.write_value(txn_id, coord, value).await,
            Self::Table(table) => table.write_value(txn_id, coord, value).await,
            Self::Transpose(transpose) => transpose.write_value(txn_id, coord, value).await,
            Self::Unary(unary) => unary.write_value(txn_id, coord, value).await,
        }
    }
}

impl ReadValueAt for SparseAccessor {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        match self {
            Self::Broadcast(broadcast) => broadcast.read_value_at(txn, coord),
            Self::Cast(cast) => cast.read_value_at(txn, coord),
            Self::Combine(combinator) => combinator.read_value_at(txn, coord),
            Self::Dense(dense) => dense.read_value_at(txn, coord),
            Self::Expand(expand) => expand.read_value_at(txn, coord),
            Self::Slice(slice) => slice.read_value_at(txn, coord),
            Self::Reduce(reduce) => reduce.read_value_at(txn, coord),
            Self::Table(table) => table.read_value_at(txn, coord),
            Self::Transpose(transpose) => transpose.read_value_at(txn, coord),
            Self::Unary(unary) => unary.read_value_at(txn, coord),
        }
    }
}

#[async_trait]
impl Transact for SparseAccessor {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Broadcast(broadcast) => broadcast.commit(txn_id).await,
            Self::Cast(cast) => cast.commit(txn_id).await,
            Self::Combine(combinator) => combinator.commit(txn_id).await,
            Self::Dense(dense) => dense.commit(txn_id).await,
            Self::Expand(expand) => expand.commit(txn_id).await,
            Self::Slice(slice) => slice.commit(txn_id).await,
            Self::Reduce(reduce) => reduce.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Transpose(transpose) => transpose.commit(txn_id).await,
            Self::Unary(unary) => unary.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Broadcast(broadcast) => broadcast.rollback(txn_id).await,
            Self::Cast(cast) => cast.rollback(txn_id).await,
            Self::Combine(combinator) => combinator.rollback(txn_id).await,
            Self::Dense(dense) => dense.rollback(txn_id).await,
            Self::Expand(expand) => expand.rollback(txn_id).await,
            Self::Slice(slice) => slice.rollback(txn_id).await,
            Self::Reduce(reduce) => reduce.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Transpose(transpose) => transpose.rollback(txn_id).await,
            Self::Unary(unary) => unary.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Broadcast(broadcast) => broadcast.finalize(txn_id).await,
            Self::Cast(cast) => cast.finalize(txn_id).await,
            Self::Combine(combinator) => combinator.finalize(txn_id).await,
            Self::Dense(dense) => dense.finalize(txn_id).await,
            Self::Expand(expand) => expand.finalize(txn_id).await,
            Self::Slice(slice) => slice.finalize(txn_id).await,
            Self::Reduce(reduce) => reduce.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::Transpose(transpose) => transpose.finalize(txn_id).await,
            Self::Unary(unary) => unary.finalize(txn_id).await,
        }
    }
}

impl From<SparseTable> for SparseAccessor {
    fn from(table: SparseTable) -> Self {
        Self::Table(table)
    }
}

impl<T: Clone + DenseAccess> From<DenseTensor<T>> for SparseAccessor {
    fn from(dense: DenseTensor<T>) -> Self {
        let source = dense.into_inner().accessor().into();
        Self::Dense(Box::new(DenseToSparse { source }))
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
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Dense(Box::new(DenseToSparse {
            source: self.source.into_inner().accessor().into(),
        }))
    }

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
        // TODO: skip zero-valued coordinates
        let shape = self.shape();
        let filled_at = stream::iter(
            Bounds::all(&Shape::from(
                axes.iter().map(|x| shape[*x]).collect::<Coord>(),
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

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.source.write_value(txn_id, coord.into(), value).await
    }
}

impl<T: Clone + DenseAccess> ReadValueAt for DenseToSparse<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        self.source.read_value_at(txn, coord)
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
pub struct SparseBroadcast<T> {
    source: T,
    rebase: transform::Broadcast,
}

impl<T: Clone + SparseAccess> SparseBroadcast<T> {
    pub fn new(source: T, rebase: transform::Broadcast) -> Self {
        Self { source, rebase }
    }

    async fn broadcast_coords<'a, S: Stream<Item = TCResult<Coord>> + 'a + Send + Unpin + 'a>(
        &'a self,
        txn: &'a Txn,
        coords: S,
        num_coords: u64,
    ) -> TCResult<SparseStream<'a>> {
        let broadcast = sorted_values(txn, self, coords, num_coords).await?;
        Ok(Box::pin(broadcast))
    }
}

impl<T: SparseAccess> TensorAccessor for SparseBroadcast<T> {
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
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Broadcast(Box::new(SparseBroadcast {
            source: self.source.accessor(),
            rebase: self.rebase,
        }))
    }

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

    async fn write_value(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<T: SparseAccess> ReadValueAt for SparseBroadcast<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));
        Box::pin(read)
    }
}

#[async_trait]
impl<T: SparseAccess> Transact for SparseBroadcast<T> {
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
pub struct SparseCast<T> {
    source: T,
    dtype: NumberType,
}

impl<T> SparseCast<T> {
    pub fn new(source: T, dtype: NumberType) -> Self {
        Self { source, dtype }
    }
}

impl<T: SparseAccess> TensorAccessor for SparseCast<T> {
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
impl<T: SparseAccess> SparseAccess for SparseCast<T> {
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Cast(Box::new(SparseCast {
            source: self.source.accessor(),
            dtype: self.dtype,
        }))
    }

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

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.source.write_value(txn_id, coord, value).await
    }
}

impl<T: SparseAccess> ReadValueAt for SparseCast<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let dtype = self.dtype;
        let read = self
            .source
            .read_value_at(txn, coord)
            .map_ok(move |(coord, value)| (coord, value.into_type(dtype)));

        Box::pin(read)
    }
}

#[async_trait]
impl<T: SparseAccess> Transact for SparseCast<T> {
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
pub struct SparseCombinator<L, R> {
    left: L,
    right: R,
    combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
}

impl<L: SparseAccess, R: SparseAccess> SparseCombinator<L, R> {
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

        Ok(SparseCombinator {
            left,
            right,
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
        let left_zero = self.left.dtype().zero();
        let right_zero = self.right.dtype().zero();

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

impl<L: SparseAccess, R: SparseAccess> TensorAccessor for SparseCombinator<L, R> {
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
impl<L: SparseAccess, R: SparseAccess> SparseAccess for SparseCombinator<L, R> {
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Combine(Box::new(SparseCombinator {
            left: self.left.accessor(),
            right: self.right.accessor(),
            combinator: self.combinator,
            dtype: self.dtype,
        }))
    }

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

    async fn write_value(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<L: SparseAccess, R: SparseAccess> ReadValueAt for SparseCombinator<L, R> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let left = self.left.read_value_at(txn, coord.to_vec());
            let right = self.right.read_value_at(txn, coord);
            let ((coord, left), (_, right)) = try_join!(left, right)?;
            let value = (self.combinator)(left, right);
            Ok((coord, value))
        })
    }
}

#[async_trait]
impl<L: SparseAccess, R: SparseAccess> Transact for SparseCombinator<L, R> {
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
pub struct SparseExpand<T> {
    source: T,
    rebase: transform::Expand,
}

impl<T> SparseExpand<T> {
    pub fn new(source: T, rebase: transform::Expand) -> Self {
        Self { source, rebase }
    }
}

impl<T: SparseAccess> TensorAccessor for SparseExpand<T> {
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
impl<T: SparseAccess> SparseAccess for SparseExpand<T> {
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Expand(Box::new(SparseExpand {
            source: self.source.accessor(),
            rebase: self.rebase,
        }))
    }

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

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value(txn_id, coord, value).await
    }
}

impl<T: SparseAccess> ReadValueAt for SparseExpand<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));
        Box::pin(read)
    }
}

#[async_trait]
impl<T: SparseAccess> Transact for SparseExpand<T> {
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

type Reductor = fn(&SparseTensor<SparseAccessor>, Txn) -> TCBoxTryFuture<Number>;

#[derive(Clone)]
pub struct SparseReduce<T: Clone + SparseAccess> {
    source: SparseTensor<T>, // TODO: replace with just T
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
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Reduce(Box::new(SparseReduce {
            source: self.source.into_inner().accessor().into(),
            rebase: self.rebase,
            reductor: self.reductor,
        }))
    }

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let reductor = self.reductor;
        let source = &self.source;

        let filled = self.filled_at(txn, (0..self.ndim()).collect()).await?;

        let filled = filled.and_then(move |coord| {
            let source_bounds = self.rebase.invert_coord(&coord);
            let txn = txn.clone();
            Box::pin(async move {
                let slice = source.slice(source_bounds)?;
                Ok((
                    coord,
                    reductor(&slice.into_inner().accessor().into(), txn).await?,
                ))
            })
        });

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        let reduce_axis = self.rebase.axis();

        if axes.is_empty() {
            let filled_at: CoordStream<'_> = Box::pin(stream::empty());
            return Ok(filled_at);
        } else if axes.to_vec().into_iter().fold(axes[0], Ord::max) < reduce_axis {
            return self.source.filled_at(txn, axes).await;
        }

        let source_axes: Vec<usize> = axes
            .into_iter()
            .map(|x| if x < reduce_axis { x } else { x + 1 })
            .chain(iter::once(reduce_axis))
            .collect();

        let filled_at = self
            .source
            .filled_at(txn, source_axes)
            .map_ok(GroupStream::from)
            .await?;
        let filled_at: CoordStream<'a> = Box::pin(filled_at);
        Ok(filled_at)
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.filled(txn)
            .await?
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn filled_in<'a>(&'a self, _txn: &'a Txn, _bounds: Bounds) -> TCResult<SparseStream<'a>> {
        Err(error::not_implemented("SparseReduce::filled_in"))
    }

    async fn write_value(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<T: Clone + SparseAccess> ReadValueAt for SparseReduce<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let source_bounds = self.rebase.invert_coord(&coord);
            let reductor = self.reductor;
            let slice = self.source.slice(source_bounds)?;
            let value = reductor(&slice.into_inner().accessor().into(), txn.clone()).await?;
            Ok((coord, value))
        })
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
    source: SparseAccessor,
    rebase: transform::Slice,
}

impl SparseSlice {
    pub fn new(source: SparseAccessor, rebase: transform::Slice) -> Self {
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
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Slice(Box::new(self))
    }

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

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value(txn_id, coord, value).await
    }
}

impl ReadValueAt for SparseSlice {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));
        Box::pin(read)
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
pub struct SparseTranspose<T> {
    source: T,
    rebase: transform::Transpose,
}

impl<T> SparseTranspose<T> {
    pub fn new(source: T, rebase: transform::Transpose) -> Self {
        Self { source, rebase }
    }
}

impl<T: SparseAccess> TensorAccessor for SparseTranspose<T> {
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
impl<T: SparseAccess> SparseAccess for SparseTranspose<T> {
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Transpose(Box::new(SparseTranspose {
            source: self.source.accessor(),
            rebase: self.rebase,
        }))
    }

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let ndim = self.ndim();
        let coords = self.filled_at(txn, (0..ndim).collect()).await?;

        Ok(Box::pin(ValueReader::new(coords, txn, self)))
    }

    async fn filled_at<'a>(&'a self, txn: &'a Txn, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        group_axes(self, txn, axes).await
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

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.source
            .write_value(txn_id, self.rebase.invert_coord(&coord), value)
            .await
    }
}

impl<T: SparseAccess> ReadValueAt for SparseTranspose<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));
        Box::pin(read)
    }
}

#[async_trait]
impl<T: SparseAccess> Transact for SparseTranspose<T> {
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
pub struct SparseUnary {
    source: SparseAccessor,
    transform: fn(Number) -> Number,
    dtype: NumberType,
}

impl SparseUnary {
    pub fn new(source: SparseAccessor, transform: fn(Number) -> Number, dtype: NumberType) -> Self {
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
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Unary(Box::new(self))
    }

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

    async fn write_value(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl ReadValueAt for SparseUnary {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let dtype = self.dtype;
        let read = self
            .source
            .read_value_at(txn, coord)
            .map_ok(move |(coord, value)| (coord, value.into_type(dtype)));

        Box::pin(read)
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

fn group_axes<'a, T: SparseAccess>(
    accessor: &'a T,
    txn: &'a Txn,
    axes: Vec<usize>,
) -> TCBoxTryFuture<'a, CoordStream<'a>> {
    Box::pin(async move {
        if axes.len() > accessor.ndim() {
            let axes: Vec<String> = axes.iter().map(|x| x.to_string()).collect();
            return Err(error::bad_request("Axis out of bounds", axes.join(", ")));
        }

        let sorted_axes: Vec<usize> = itertools::sorted(axes.to_vec()).collect::<Vec<usize>>();
        if axes == sorted_axes {
            let filled = accessor.filled(txn).await?;
            let filled_coords =
                filled.map_ok(move |(coord, _)| axes.iter().map(|x| coord[*x]).collect::<Coord>());

            let filled_at: CoordStream<'a> = Box::pin(GroupStream::from(filled_coords));
            Ok(filled_at)
        } else {
            let (coords_to_count, coords) = try_join!(
                accessor.filled_at(txn, sorted_axes.to_vec()),
                accessor.filled_at(txn, sorted_axes)
            )?;

            let num_coords = count_stream(coords_to_count).await?;
            let coords =
                coords.map_ok(move |coord| axes.iter().map(|x| coord[*x]).collect::<Coord>());

            let filled_at: CoordStream<'a> =
                sorted_coords(txn, accessor.shape(), coords, num_coords)
                    .map_ok(Box::pin)
                    .await?;

            Ok(filled_at)
        }
    })
}
