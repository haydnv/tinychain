use std::pin::Pin;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;

use crate::class::Instance;
use crate::collection::stream::GroupStream;
use crate::error;
use crate::general::{TCBoxTryFuture, TCResult};
use crate::scalar::value::number::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::super::bounds::*;
use super::super::dense::{DenseAccess, DenseAccessor, DenseTensor};
use super::super::stream::*;
use super::super::transform::{self, Rebase};
use super::super::Coord;

use super::combine::SparseCombine;
use super::{
    SparseStream, SparseTable, SparseTableSlice, SparseTensor, TensorAccess, TensorTransform,
    ERR_NONBIJECTIVE_WRITE,
};

pub type CoordStream<'a> = Pin<Box<dyn Stream<Item = TCResult<Coord>> + Send + Unpin + 'a>>;

#[async_trait]
pub trait SparseAccess: ReadValueAt + TensorAccess + Transact + 'static {
    type Slice: Clone + SparseAccess;
    type Transpose: Clone + SparseAccess;

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

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64>;

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice>;

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose>;

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

#[derive(Clone)]
pub enum SparseAccessor {
    Broadcast(Box<SparseBroadcast<SparseAccessor>>),
    Cast(Box<SparseCast<SparseAccessor>>),
    Combine(Box<SparseCombinator<SparseAccessor, SparseAccessor>>),
    Dense(Box<DenseToSparse<DenseAccessor>>),
    Expand(Box<SparseExpand<SparseAccessor>>),
    Reduce(Box<SparseReduce<SparseAccessor>>),
    Table(SparseTable),
    TableSlice(SparseTableSlice),
    Transpose(Box<SparseTranspose<SparseAccessor>>),
    Unary(Box<SparseUnary>),
}

impl TensorAccess for SparseAccessor {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Broadcast(broadcast) => broadcast.dtype(),
            Self::Cast(cast) => cast.dtype(),
            Self::Combine(combinator) => combinator.dtype(),
            Self::Dense(dense) => dense.dtype(),
            Self::Expand(expand) => expand.dtype(),
            Self::Reduce(reduce) => reduce.dtype(),
            Self::Table(table) => table.dtype(),
            Self::TableSlice(slice) => slice.dtype(),
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
            Self::Reduce(reduce) => reduce.ndim(),
            Self::Table(table) => table.ndim(),
            Self::TableSlice(slice) => slice.ndim(),
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
            Self::Reduce(reduce) => reduce.shape(),
            Self::Table(table) => table.shape(),
            Self::TableSlice(slice) => slice.shape(),
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
            Self::Reduce(reduce) => reduce.size(),
            Self::Table(table) => table.size(),
            Self::TableSlice(slice) => slice.size(),
            Self::Transpose(transpose) => transpose.size(),
            Self::Unary(unary) => unary.size(),
        }
    }
}

#[async_trait]
impl SparseAccess for SparseAccessor {
    type Slice = Self;
    type Transpose = Self;

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
            Self::Reduce(reduce) => reduce.filled(txn).await,
            Self::Table(table) => table.filled(txn).await,
            Self::TableSlice(slice) => slice.filled(txn).await,
            Self::Transpose(transpose) => transpose.filled(txn).await,
            Self::Unary(unary) => unary.filled(txn).await,
        }
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        match self {
            Self::Broadcast(broadcast) => broadcast.filled_count(txn).await,
            Self::Cast(cast) => cast.filled_count(txn).await,
            Self::Combine(combinator) => combinator.filled_count(txn).await,
            Self::Dense(dense) => dense.filled_count(txn).await,
            Self::Expand(expand) => expand.filled_count(txn).await,
            Self::Reduce(reduce) => reduce.filled_count(txn).await,
            Self::Table(table) => table.filled_count(txn).await,
            Self::TableSlice(slice) => slice.filled_count(txn).await,
            Self::Transpose(transpose) => transpose.filled_count(txn).await,
            Self::Unary(unary) => unary.filled_count(txn).await,
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::Broadcast(broadcast) => broadcast.slice(bounds).map(SparseAccess::accessor),
            Self::Cast(cast) => cast.slice(bounds).map(SparseAccess::accessor),
            Self::Combine(combinator) => combinator.slice(bounds).map(SparseAccess::accessor),
            Self::Dense(dense) => dense.slice(bounds).map(SparseAccess::accessor),
            Self::Expand(expand) => expand.slice(bounds).map(SparseAccess::accessor),
            Self::Reduce(reduce) => reduce.slice(bounds).map(SparseAccess::accessor),
            Self::Table(table) => table.slice(bounds).map(SparseAccess::accessor),
            Self::TableSlice(slice) => slice.slice(bounds).map(SparseAccess::accessor),
            Self::Transpose(transpose) => transpose.slice(bounds).map(SparseAccess::accessor),
            Self::Unary(unary) => unary.slice(bounds).map(SparseAccess::accessor),
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        match self {
            Self::Broadcast(broadcast) => {
                broadcast.transpose(permutation).map(SparseAccess::accessor)
            }
            Self::Cast(cast) => cast.transpose(permutation).map(SparseAccess::accessor),
            Self::Combine(combinator) => combinator
                .transpose(permutation)
                .map(SparseAccess::accessor),
            Self::Dense(dense) => dense.transpose(permutation).map(SparseAccess::accessor),
            Self::Expand(expand) => expand.transpose(permutation).map(SparseAccess::accessor),
            Self::Reduce(reduce) => reduce.transpose(permutation).map(SparseAccess::accessor),
            Self::Table(table) => table.transpose(permutation).map(SparseAccess::accessor),
            Self::TableSlice(slice) => slice.transpose(permutation).map(SparseAccess::accessor),
            Self::Transpose(transpose) => {
                transpose.transpose(permutation).map(SparseAccess::accessor)
            }
            Self::Unary(unary) => unary.transpose(permutation).map(SparseAccess::accessor),
        }
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        match self {
            Self::Broadcast(broadcast) => broadcast.write_value(txn_id, coord, value).await,
            Self::Cast(cast) => cast.write_value(txn_id, coord, value).await,
            Self::Combine(combinator) => combinator.write_value(txn_id, coord, value).await,
            Self::Dense(dense) => dense.write_value(txn_id, coord, value).await,
            Self::Expand(expand) => expand.write_value(txn_id, coord, value).await,
            Self::Reduce(reduce) => reduce.write_value(txn_id, coord, value).await,
            Self::Table(table) => table.write_value(txn_id, coord, value).await,
            Self::TableSlice(slice) => slice.write_value(txn_id, coord, value).await,
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
            Self::Reduce(reduce) => reduce.read_value_at(txn, coord),
            Self::Table(table) => table.read_value_at(txn, coord),
            Self::TableSlice(slice) => slice.read_value_at(txn, coord),
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
            Self::Reduce(reduce) => reduce.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::TableSlice(slice) => slice.commit(txn_id).await,
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
            Self::Reduce(reduce) => reduce.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::TableSlice(slice) => slice.rollback(txn_id).await,
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
            Self::Reduce(reduce) => reduce.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::TableSlice(slice) => slice.finalize(txn_id).await,
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
pub struct DenseToSparse<T> {
    source: T,
}

impl<T> DenseToSparse<T> {
    pub fn new(source: T) -> Self {
        Self { source }
    }
}

impl<T: DenseAccess> TensorAccess for DenseToSparse<T> {
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
impl<T: DenseAccess> SparseAccess for DenseToSparse<T> {
    type Slice = DenseToSparse<<T as DenseAccess>::Slice>;
    type Transpose = DenseToSparse<<T as DenseAccess>::Transpose>;

    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Dense(Box::new(DenseToSparse {
            source: self.source.accessor(),
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

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source
            .value_stream(txn)
            .await?
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.source.slice(bounds).map(DenseToSparse::from)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        self.source.transpose(permutation).map(DenseToSparse::from)
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.source.write_value(txn_id, coord.into(), value).await
    }
}

impl<T: DenseAccess> ReadValueAt for DenseToSparse<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        self.source.read_value_at(txn, coord)
    }
}

#[async_trait]
impl<T: DenseAccess> Transact for DenseToSparse<T> {
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

impl<T: Clone + DenseAccess> From<DenseTensor<T>> for DenseToSparse<T> {
    fn from(dense: DenseTensor<T>) -> Self {
        dense.into_inner().into()
    }
}

impl<T: DenseAccess> From<T> for DenseToSparse<T> {
    fn from(source: T) -> DenseToSparse<T> {
        Self { source }
    }
}

#[derive(Clone)]
pub struct SparseBroadcast<T> {
    source: T,
    rebase: transform::Broadcast,
}

impl<T: Clone + SparseAccess> SparseBroadcast<T> {
    pub fn new(source: T, shape: Shape) -> TCResult<Self> {
        let rebase = transform::Broadcast::new(source.shape().clone(), shape)?;
        Ok(Self { source, rebase })
    }

    // TODO: require values in the input stream to avoid the redundant lookup
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

impl<T: SparseAccess> TensorAccess for SparseBroadcast<T> {
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
    type Slice = SparseBroadcast<<T as SparseAccess>::Slice>;
    type Transpose = SparseTranspose<Self>;

    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Broadcast(Box::new(SparseBroadcast {
            source: self.source.accessor(),
            rebase: self.rebase,
        }))
    }

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let rebase = &self.rebase;
        let num_coords = self.source.filled_count(txn).await?;
        let filled = self.source.filled(txn).await?;

        let filled = filled
            .map_ok(move |(coord, _)| {
                stream::iter(rebase.map_coord(coord).affected().map(TCResult::Ok))
            })
            .try_flatten();

        self.broadcast_coords(txn, filled, num_coords).await
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        let rebase = &self.rebase;
        let filled = self.source.filled(txn).await?;
        filled
            .try_fold(0u64, move |count, (coord, _)| {
                future::ready(Ok(count + rebase.map_bounds(coord.into()).size()))
            })
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.shape().validate_bounds(&bounds)?;

        let shape = bounds.to_shape();
        let source_bounds = self.rebase.invert_bounds(bounds);
        let source = self.source.slice(source_bounds)?;
        SparseBroadcast::new(source, shape)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        SparseTranspose::new(self, permutation)
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

impl<T: SparseAccess> TensorAccess for SparseCast<T> {
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
    type Slice = SparseCast<<T as SparseAccess>::Slice>;
    type Transpose = SparseCast<<T as SparseAccess>::Transpose>;

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

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let source = self.source.slice(bounds)?;
        Ok(SparseCast {
            source,
            dtype: self.dtype,
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let source = self.source.transpose(permutation)?;
        Ok(SparseCast {
            source,
            dtype: self.dtype,
        })
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

impl<L: SparseAccess, R: SparseAccess> TensorAccess for SparseCombinator<L, R> {
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
    type Slice = SparseCombinator<<L as SparseAccess>::Slice, <R as SparseAccess>::Slice>;
    type Transpose =
        SparseCombinator<<L as SparseAccess>::Transpose, <R as SparseAccess>::Transpose>;

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

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        let count = self
            .filled(txn)
            .await?
            .fold(0u64, |count, _| future::ready(count + 1))
            .await;

        Ok(count)
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let left = self.left.slice(bounds.clone())?;
        let right = self.right.slice(bounds)?;
        assert_eq!(left.shape(), right.shape());

        Ok(SparseCombinator {
            left,
            right,
            combinator: self.combinator,
            dtype: self.dtype,
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let left = self.left.transpose(permutation.clone())?;
        let right = self.right.transpose(permutation)?;
        assert_eq!(left.shape(), right.shape());

        Ok(SparseCombinator {
            left,
            right,
            combinator: self.combinator,
            dtype: self.dtype,
        })
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

impl<T: SparseAccess> SparseExpand<T> {
    pub fn new(source: T, axis: usize) -> TCResult<Self> {
        let rebase = transform::Expand::new(source.shape().clone(), axis)?;
        Ok(Self { source, rebase })
    }
}

impl<T: SparseAccess> TensorAccess for SparseExpand<T> {
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
    type Slice = <T as SparseAccess>::Slice;
    type Transpose = SparseExpand<<T as SparseAccess>::Transpose>;

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

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.shape().validate_bounds(&bounds)?;

        let source_bounds = self.rebase.invert_bounds(bounds);
        self.source.slice(source_bounds)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let expand_axis = if let Some(permutation) = &permutation {
            if permutation.len() != self.ndim() {
                return Err(error::unsupported(format!(
                    "Invalid permutation for tensor of shape {}: {:?}",
                    self.shape(),
                    permutation
                )));
            }

            permutation[self.rebase.expand_axis()]
        } else {
            self.ndim() - self.rebase.expand_axis()
        };

        let source = self.source.transpose(permutation)?;
        SparseExpand::new(source, expand_axis)
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

    async fn filled_at<'a>(&'a self, txn: &'a Txn) -> TCResult<CoordStream<'a>> {
        let reduce_axis = self.rebase.axis();

        let source = self.source.filled(txn).await?;

        let filled_at = source.map_ok(move |(mut coord, _)| {
            coord.remove(reduce_axis);
            coord
        });

        let filled_at: CoordStream<'a> = Box::pin(GroupStream::from(filled_at));
        Ok(filled_at)
    }
}

impl<T: Clone + SparseAccess> TensorAccess for SparseReduce<T> {
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
    type Slice = SparseReduce<<T as SparseAccess>::Slice>;
    type Transpose = SparseTranspose<Self>;

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

        let filled = self.filled_at(txn).await?;

        let filled = filled.and_then(move |coord| {
            let source_bounds = self.rebase.invert_coord(&coord);
            let txn = txn.clone();
            Box::pin(async move {
                let slice = source.slice(source_bounds)?;
                let slice = slice.into_inner().accessor().into();
                let value = reductor(&slice, txn).await?;
                Ok((coord, value))
            })
        });

        Ok(Box::pin(filled))
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.filled(txn)
            .await?
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.shape().validate_bounds(&bounds)?;

        let reduce_axis = self.rebase.reduce_axis(&bounds);
        let source_bounds = self.rebase.invert_bounds(bounds);
        let source = self.source.slice(source_bounds)?;
        SparseReduce::new(source.into_inner().into(), reduce_axis, self.reductor)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        SparseTranspose::new(self, permutation)
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
pub struct SparseTranspose<T> {
    source: T,
    rebase: transform::Transpose,
}

impl<T: SparseAccess> SparseTranspose<T> {
    pub fn new(source: T, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        let rebase = transform::Transpose::new(source.shape().clone(), permutation)?;
        Ok(Self { source, rebase })
    }
}

impl<T: SparseAccess> TensorAccess for SparseTranspose<T> {
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
    type Slice = <<T as SparseAccess>::Slice as SparseAccess>::Transpose;
    type Transpose = <T as SparseAccess>::Transpose;

    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Transpose(Box::new(SparseTranspose {
            source: self.source.accessor(),
            rebase: self.rebase,
        }))
    }

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let num_coords = self.filled_count(txn).await?;
        let coords = self.source.filled(txn).await?;
        let coords = coords.map_ok(|(coord, _)| coord); // TODO: forward the values as well
        let filled = sorted_values(txn, self, coords, num_coords).await?;
        Ok(Box::pin(filled))
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.shape().validate_bounds(&bounds)?;

        let slice_permutation = self.rebase.invert_permutation(&bounds);
        let source_bounds = self.rebase.invert_bounds(bounds);
        let source = self.source.slice(source_bounds)?;
        source.transpose(Some(slice_permutation))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let permutation = permutation.map(|axes| self.rebase.invert_axes(&axes));
        self.source.transpose(permutation)
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
    source: SparseAccessor, // TODO: can this be a type parameter?
    dtype: NumberType,
    transform: fn(Number) -> Number,
}

impl SparseUnary {
    pub fn new(source: SparseAccessor, transform: fn(Number) -> Number, dtype: NumberType) -> Self {
        Self {
            source,
            dtype,
            transform,
        }
    }
}

impl TensorAccess for SparseUnary {
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
    type Slice = Self;
    type Transpose = Self;

    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Unary(Box::new(self))
    }

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let transform = self.transform;
        let filled = self.source.filled(txn).await?;
        let cast = filled.map_ok(move |(coord, value)| (coord, transform(value)));
        Ok(Box::pin(cast))
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let source = self.source.slice(bounds)?;
        Ok(SparseUnary {
            source: source.accessor(),
            dtype: self.dtype,
            transform: self.transform,
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let source = self.source.transpose(permutation)?;
        Ok(SparseUnary {
            source: source.accessor(),
            dtype: self.dtype,
            transform: self.transform,
        })
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
