use std::fmt;

use afarray::{Array, CoordBlocks, CoordMerge, Coords};
use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;
use safecast::AsType;

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::{FloatInstance, Number, NumberClass, NumberInstance, NumberType};
use tcgeneric::{TCBoxTryFuture, TCBoxTryStream};

use crate::dense::{DenseAccess, DenseAccessor, DenseTensor, PER_BLOCK};
use crate::stream::{sorted_coords, sorted_values, Read, ReadValueAt};
use crate::{
    coord_bounds, transform, AxisBounds, Bounds, Coord, Phantom, Shape, TensorAccess, TensorType,
    TensorUnary, ERR_INF, ERR_NAN,
};

use super::combine::{coord_to_offset, SparseCombine};
use super::table::{SparseTable, SparseTableSlice};
use super::{SparseRow, SparseStream, SparseTensor};

/// Access methods for [`SparseTensor`] data
#[async_trait]
pub trait SparseAccess<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>>:
    Clone + fmt::Display + ReadValueAt<D, Txn = T> + TensorAccess + Send + Sync + 'static
{
    /// The type of a slice of this accessor
    type Slice: SparseAccess<FD, FS, D, T>;

    /// The type of a transpose of this accessor
    type Transpose: SparseAccess<FD, FS, D, T>;

    /// Return this accessor as a [`SparseAccessor`].
    fn accessor(self) -> SparseAccessor<FD, FS, D, T>;

    /// Return this [`SparseTensor`]'s contents as an ordered stream of ([`Coord`], [`Number`]) pairs.
    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>>;

    /// Return an ordered stream of unique [`Coord`]s on the given axes with nonzero values.
    ///
    /// Note: is it *not* safe to assume that all returned coordinates are filled,
    /// but it *is* safe to assume that all coordinates *not* returned are *not* filled.
    ///
    /// TODO: add `sort` parameter to make sorting results optional.
    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>>;

    /// Return the number of nonzero values in this [`SparseTensor`].
    async fn filled_count(self, txn: T) -> TCResult<u64>;

    /// Return a slice of this accessor with the given [`Bounds`].
    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice>;

    /// Return this accessor as transposed according to the given `permutation`.
    ///
    /// If no permutation is given, this accessor's axes will be reversed.
    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose>;
}

/// Write methods for [`SparseTensor`] data
#[async_trait]
pub trait SparseWrite<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>>:
    SparseAccess<FD, FS, D, T>
{
    /// Write the given `value` at the given `coord` of this [`SparseTensor`].
    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

/// A generic [`SparseAccess`] type
#[derive(Clone)]
pub enum SparseAccessor<FD, FS, D, T> {
    Broadcast(Box<SparseBroadcast<FD, FS, D, T, Self>>),
    Cast(Box<SparseCast<FD, FS, D, T, Self>>),
    Combine(Box<SparseCombinator<FD, FS, D, T, Self, Self>>),
    CombineConst(Box<SparseConstCombinator<FD, FS, D, T, Self>>),
    CombineLeft(Box<SparseLeftCombinator<FD, FS, D, T, Self, Self>>),
    Dense(Box<DenseToSparse<FD, FS, D, T, DenseAccessor<FD, FS, D, T>>>),
    Expand(Box<SparseExpand<FD, FS, D, T, Self>>),
    Slice(SparseTableSlice<FD, FS, D, T>),
    Reduce(Box<SparseReduce<FD, FS, D, T>>),
    Table(SparseTable<FD, FS, D, T>),
    Transpose(Box<SparseTranspose<FD, FS, D, T, Self>>),
    Unary(Box<SparseUnary<FD, FS, D, T>>),
}

impl<FD, FS, D, T> TensorAccess for SparseAccessor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    fn dtype(&self) -> NumberType {
        match self {
            Self::Broadcast(broadcast) => broadcast.dtype(),
            Self::Cast(cast) => cast.dtype(),
            Self::Combine(combine) => combine.dtype(),
            Self::CombineConst(combine) => combine.dtype(),
            Self::CombineLeft(combine) => combine.dtype(),
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
            Self::Combine(combine) => combine.ndim(),
            Self::CombineConst(combine) => combine.ndim(),
            Self::CombineLeft(combine) => combine.ndim(),
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
            Self::Combine(combine) => combine.shape(),
            Self::CombineConst(combine) => combine.shape(),
            Self::CombineLeft(combine) => combine.shape(),
            Self::Dense(dense) => dense.shape(),
            Self::Expand(expand) => expand.shape(),
            Self::Reduce(reduce) => reduce.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::Table(table) => table.shape(),
            Self::Transpose(transpose) => transpose.shape(),
            Self::Unary(unary) => unary.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Broadcast(broadcast) => broadcast.size(),
            Self::Cast(cast) => cast.size(),
            Self::Combine(combine) => combine.size(),
            Self::CombineConst(combine) => combine.size(),
            Self::CombineLeft(combine) => combine.size(),
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
impl<FD, FS, D, T> SparseAccess<FD, FS, D, T> for SparseAccessor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Slice = Self;
    type Transpose = Self;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        self
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.filled(txn).await,
            Self::Cast(cast) => cast.filled(txn).await,
            Self::Combine(combine) => combine.filled(txn).await,
            Self::CombineConst(combine) => combine.filled(txn).await,
            Self::CombineLeft(combine) => combine.filled(txn).await,
            Self::Dense(dense) => dense.filled(txn).await,
            Self::Expand(expand) => expand.filled(txn).await,
            Self::Reduce(reduce) => reduce.filled(txn).await,
            Self::Slice(slice) => slice.filled(txn).await,
            Self::Table(table) => table.filled(txn).await,
            Self::Transpose(transpose) => transpose.filled(txn).await,
            Self::Unary(unary) => unary.filled(txn).await,
        }
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.filled_at(txn, axes).await,
            Self::Cast(cast) => cast.filled_at(txn, axes).await,
            Self::Combine(combine) => combine.filled_at(txn, axes).await,
            Self::CombineConst(combine) => combine.filled_at(txn, axes).await,
            Self::CombineLeft(combine) => combine.filled_at(txn, axes).await,
            Self::Dense(dense) => dense.filled_at(txn, axes).await,
            Self::Expand(expand) => expand.filled_at(txn, axes).await,
            Self::Reduce(reduce) => reduce.filled_at(txn, axes).await,
            Self::Slice(slice) => slice.filled_at(txn, axes).await,
            Self::Table(table) => table.filled_at(txn, axes).await,
            Self::Transpose(transpose) => transpose.filled_at(txn, axes).await,
            Self::Unary(unary) => unary.filled_at(txn, axes).await,
        }
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        match self {
            Self::Broadcast(broadcast) => broadcast.filled_count(txn).await,
            Self::Cast(cast) => cast.filled_count(txn).await,
            Self::Combine(combine) => combine.filled_count(txn).await,
            Self::CombineConst(combine) => combine.filled_count(txn).await,
            Self::CombineLeft(combine) => combine.filled_count(txn).await,
            Self::Dense(dense) => dense.filled_count(txn).await,
            Self::Expand(expand) => expand.filled_count(txn).await,
            Self::Reduce(reduce) => reduce.filled_count(txn).await,
            Self::Slice(slice) => slice.filled_count(txn).await,
            Self::Table(table) => table.filled_count(txn).await,
            Self::Transpose(transpose) => transpose.filled_count(txn).await,
            Self::Unary(unary) => unary.filled_count(txn).await,
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::Broadcast(broadcast) => broadcast.slice(bounds).map(SparseAccess::accessor),
            Self::Cast(cast) => cast.slice(bounds).map(SparseAccess::accessor),
            Self::Combine(combinator) => combinator.slice(bounds).map(SparseAccess::accessor),
            Self::CombineConst(combinator) => combinator.slice(bounds).map(SparseAccess::accessor),
            Self::CombineLeft(combinator) => combinator.slice(bounds).map(SparseAccess::accessor),
            Self::Dense(dense) => dense.slice(bounds).map(SparseAccess::accessor),
            Self::Expand(expand) => expand.slice(bounds).map(SparseAccess::accessor),
            Self::Reduce(reduce) => reduce.slice(bounds).map(SparseAccess::accessor),
            Self::Slice(slice) => slice.slice(bounds).map(SparseAccess::accessor),
            Self::Table(table) => table.slice(bounds).map(SparseAccess::accessor),
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

            Self::CombineConst(combinator) => combinator
                .transpose(permutation)
                .map(SparseAccess::accessor),

            Self::CombineLeft(combinator) => combinator
                .transpose(permutation)
                .map(SparseAccess::accessor),

            Self::Dense(dense) => dense.transpose(permutation).map(SparseAccess::accessor),

            Self::Expand(expand) => expand.transpose(permutation).map(SparseAccess::accessor),

            Self::Reduce(reduce) => reduce.transpose(permutation).map(SparseAccess::accessor),

            Self::Table(table) => table.transpose(permutation).map(SparseAccess::accessor),

            Self::Slice(slice) => slice.transpose(permutation).map(SparseAccess::accessor),

            Self::Transpose(transpose) => {
                transpose.transpose(permutation).map(SparseAccess::accessor)
            }

            Self::Unary(unary) => unary.transpose(permutation).map(SparseAccess::accessor),
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> SparseWrite<FD, FS, D, T> for SparseAccessor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        match self {
            Self::Table(table) => table.write_value(txn_id, coord, value).await,
            _ => Err(TCError::unsupported("cannot write to a Tensor view")),
        }
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseAccessor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        match self {
            Self::Broadcast(broadcast) => broadcast.read_value_at(txn, coord),
            Self::Cast(cast) => cast.read_value_at(txn, coord),
            Self::Combine(combine) => combine.read_value_at(txn, coord),
            Self::CombineConst(combine) => combine.read_value_at(txn, coord),
            Self::CombineLeft(combine) => combine.read_value_at(txn, coord),
            Self::Dense(dense) => dense.read_value_at(txn, coord),
            Self::Expand(expand) => expand.read_value_at(txn, coord),
            Self::Reduce(reduce) => reduce.read_value_at(txn, coord),
            Self::Slice(slice) => slice.read_value_at(txn, coord),
            Self::Table(table) => table.read_value_at(txn, coord),
            Self::Transpose(transpose) => transpose.read_value_at(txn, coord),
            Self::Unary(unary) => unary.read_value_at(txn, coord),
        }
    }
}

impl<FD, FS, D, T> fmt::Display for SparseAccessor<FD, FS, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Broadcast(broadcast) => fmt::Display::fmt(broadcast, f),
            Self::Cast(cast) => fmt::Display::fmt(cast, f),
            Self::Combine(combinator) => fmt::Display::fmt(combinator, f),
            Self::CombineConst(combinator) => fmt::Display::fmt(combinator, f),
            Self::CombineLeft(combinator) => fmt::Display::fmt(combinator, f),
            Self::Dense(dense) => fmt::Display::fmt(dense, f),
            Self::Expand(expand) => fmt::Display::fmt(expand, f),
            Self::Reduce(reduce) => fmt::Display::fmt(reduce, f),
            Self::Slice(slice) => fmt::Display::fmt(slice, f),
            Self::Table(table) => fmt::Display::fmt(table, f),
            Self::Transpose(transpose) => fmt::Display::fmt(transpose, f),
            Self::Unary(unary) => fmt::Display::fmt(unary, f),
        }
    }
}

#[derive(Clone)]
pub struct DenseToSparse<FD, FS, D, T, B> {
    source: B,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> TensorAccess for DenseToSparse<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
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
impl<FD, FS, D, T, B> SparseAccess<FD, FS, D, T> for DenseToSparse<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = DenseToSparse<FD, FS, D, T, B::Slice>;
    type Transpose = DenseToSparse<FD, FS, D, T, B::Transpose>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Dense(Box::new(DenseToSparse {
            source: self.source.accessor(),
            phantom: self.phantom,
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let zero = self.dtype().zero();
        let bounds = Bounds::all(self.shape());

        let values = self.source.value_stream(txn).await?;
        let filled = stream::iter(bounds.affected())
            .zip(values)
            .map(|(coord, r)| r.map(|value| (coord, value)))
            .try_filter(move |(_, value)| future::ready(value != &zero));

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        if axes.is_empty() {
            return Ok(Box::pin(stream::empty()));
        }

        let bounds = Bounds::all(self.source.shape());
        let ndim = axes.len();

        let affected = {
            let shape = self.source.shape();
            let shape = axes.iter().map(|x| shape[*x]).collect();
            Bounds::all(&shape).affected()
        };

        let source = self.source;
        let filled_at = stream::iter(affected)
            .map(move |coord| {
                let mut bounds = bounds.clone();
                for (x, i) in axes.iter().zip(&coord) {
                    bounds[*x] = AxisBounds::At(*i);
                }

                (coord, bounds)
            })
            .then(move |(coord, bounds)| {
                let slice = source
                    .clone()
                    .slice(bounds)
                    .map(DenseTensor::from)
                    .map(|slice| (coord, slice));

                future::ready(slice)
            })
            .map_ok(move |(coord, slice)| slice.any(txn.clone()).map_ok(|any| (coord, any)))
            .try_buffered(num_cpus::get())
            .try_filter_map(|(coord, any)| {
                let coord = if any { Some(coord) } else { None };
                future::ready(Ok(coord))
            });

        let filled_at = Box::pin(filled_at);
        let filled_at = CoordBlocks::new(filled_at, ndim, PER_BLOCK);
        Ok(Box::pin(filled_at))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let zero = self.dtype().zero();
        let values = self.source.value_stream(txn).await?;

        values
            .try_filter(move |value| future::ready(value != &zero))
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.source.slice(bounds).map(DenseToSparse::from)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        self.source.transpose(permutation).map(DenseToSparse::from)
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for DenseToSparse<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        self.source.read_value_at(txn, coord)
    }
}

impl<FD, FS, D, T, B> fmt::Display for DenseToSparse<FD, FS, D, T, B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sparse representation of a dense Tensor")
    }
}

impl<FD, FS, D, T, B> From<B> for DenseToSparse<FD, FS, D, T, B> {
    fn from(source: B) -> Self {
        Self {
            source,
            phantom: Phantom::default(),
        }
    }
}

#[derive(Clone)]
pub struct SparseBroadcast<FD, FS, D, T, A> {
    source: A,
    rebase: transform::Broadcast,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseBroadcast<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    pub fn new(source: A, shape: Shape) -> TCResult<Self> {
        debug!("SparseBroadcast::new {} into {}", source.shape(), shape);
        let rebase = transform::Broadcast::new(source.shape().clone(), shape)?;
        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseBroadcast<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
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
        self.source.size()
    }
}

#[async_trait]
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseBroadcast<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Slice = SparseBroadcast<FD, FS, D, T, A::Slice>;
    type Transpose = SparseTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Broadcast(Box::new(SparseBroadcast {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        debug!("SparseBroadcast::filled");

        let source_axes: Vec<usize> = (0..self.source.ndim()).collect();
        let ndim = self.ndim();

        let rebase = self.rebase.clone();

        let filled_at = self
            .source
            .clone()
            .filled_at(txn.clone(), source_axes)
            .await?;

        let coords = filled_at
            .map_ok(move |coords| stream::iter(coords.to_vec()).map(TCResult::Ok))
            .try_flatten()
            .map_ok(move |coord| rebase.map_coord(coord))
            .map_ok(move |bounds| stream::iter(bounds.affected().map(TCResult::Ok)))
            .try_flatten();

        let coords = CoordBlocks::new(coords, ndim, PER_BLOCK);
        let broadcast = sorted_values::<FD, FS, T, D, _, _>(txn, self, coords).await?;
        Ok(Box::pin(broadcast))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        debug!(
            "SparseBroadcast::filled_at {:?} (source is {})",
            axes, self.source
        );

        if axes.is_empty() {
            return Ok(Box::pin(stream::empty()));
        }

        self.shape().validate_axes(&axes)?;

        let shape = Shape::from({
            let shape = self.shape();
            axes.iter().map(|x| shape[*x]).collect::<Vec<u64>>()
        });

        let ndim = self.ndim();
        let rebase = self.rebase;
        let source_axes = (0..self.source.ndim()).collect();

        let filled_at = self.source.filled_at(txn.clone(), source_axes).await?;
        let filled_at = filled_at
            .map_ok(|coords| stream::iter(coords.to_vec()).map(TCResult::Ok))
            .try_flatten()
            .map_ok(move |coord| {
                debug!("broadcast source coord {:?}", coord);
                rebase.map_coord(coord)
            })
            // TODO: can this happen in `Coords`? maybe a new stream generator type?
            .map_ok(move |bounds| stream::iter(bounds.affected().map(TCResult::Ok)))
            .try_flatten();

        let filled_at =
            CoordBlocks::new(filled_at, ndim, PER_BLOCK).map_ok(move |coords| coords.get(&axes));

        let filled_at = sorted_coords::<FD, FS, D, T, _>(&txn, shape, filled_at).await?;

        Ok(Box::pin(filled_at))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let rebase = self.rebase;
        let filled = self.source.filled(txn).await?;

        filled
            .try_fold(0u64, move |count, (coord, _)| {
                future::ready(Ok(count + rebase.map_bounds(coord.into()).size()))
            })
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        debug!("SparseBroadcast::slice {} from {}", bounds, self.shape());
        self.shape().validate_bounds(&bounds)?;

        let shape = bounds.to_shape(self.shape())?;
        let source_bounds = self.rebase.invert_bounds(bounds);

        debug!(
            "SparseBroadcast::slice source bounds are {} (from {})",
            source_bounds,
            self.source.shape()
        );

        let source = self.source.slice(source_bounds)?;
        SparseBroadcast::new(source, shape)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        SparseTranspose::new(self, permutation)
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseBroadcast<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            self.shape().validate_coord(&coord)?;

            let source_coord = self.rebase.invert_coord(&coord);
            self.source
                .read_value_at(txn, source_coord)
                .map_ok(|(_, val)| (coord, val))
                .await
        })
    }
}

impl<FD, FS, D, T, A> fmt::Display for SparseBroadcast<FD, FS, D, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a broadcasted sparse Tensor")
    }
}

#[derive(Clone)]
pub struct SparseCast<FD, FS, D, T, A> {
    source: A,
    dtype: NumberType,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseCast<FD, FS, D, T, A> {
    pub fn new(source: A, dtype: NumberType) -> Self {
        Self {
            source,
            dtype,
            phantom: Phantom::default(),
        }
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseCast<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
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
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseCast<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Slice = SparseCast<FD, FS, D, T, A::Slice>;
    type Transpose = SparseCast<FD, FS, D, T, A::Transpose>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Cast(Box::new(SparseCast::new(
            self.source.accessor(),
            self.dtype,
        )))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let dtype = self.dtype;

        let filled = self.source.filled(txn).await?;
        let cast = filled.map_ok(move |(coord, value)| (coord, value.into_type(dtype)));
        Ok(Box::pin(cast))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        Ok(SparseCast {
            source: self.source.slice(bounds)?,
            dtype: self.dtype,
            phantom: self.phantom,
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let source = self.source.transpose(permutation)?;
        Ok(SparseCast {
            source,
            dtype: self.dtype,
            phantom: self.phantom,
        })
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseCast<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        let dtype = self.dtype;
        let read = self
            .source
            .read_value_at(txn, coord)
            .map_ok(move |(coord, value)| (coord, value.into_type(dtype)));

        Box::pin(read)
    }
}

impl<FD, FS, D, T, A> fmt::Display for SparseCast<FD, FS, D, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a casted sparse Tensor")
    }
}

#[derive(Clone)]
pub struct SparseCombinator<FD, FS, D, T, L, R> {
    left: L,
    right: R,
    combinator: fn(Number, Number) -> Number,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, L, R> SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    pub fn new(left: L, right: R, combinator: fn(Number, Number) -> Number) -> TCResult<Self> {
        if left.shape() != right.shape() {
            return Err(TCError::unsupported(
                "tried to combine SparseTensors with different shapes",
            ));
        }

        Ok(SparseCombinator {
            left,
            right,
            combinator,
            phantom: Phantom::default(),
        })
    }

    pub async fn filled_inner<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let left = self.left.clone().filled(txn.clone());
        let right = self.right.clone().filled(txn);
        let (left, right) = try_join!(left, right)?;

        let coord_bounds = coord_bounds(self.shape());
        let combinator = self.combinator;
        let left_zero = self.left.dtype().zero();
        let right_zero = self.right.dtype().zero();

        let offset = move |row: &SparseRow| coord_to_offset(&row.0, &coord_bounds);
        let combined = SparseCombine::new(left, right, offset)
            .map_ok(move |(l, r)| match (l, r) {
                (Some((l_coord, l)), Some((r_coord, r))) => {
                    debug_assert_eq!(l_coord, r_coord);
                    (l_coord, combinator(l, r))
                }
                (Some((l_coord, l)), None) => (l_coord, combinator(l, right_zero)),
                (None, Some((r_coord, r))) => (r_coord, combinator(left_zero, r)),
                (None, None) => {
                    panic!("expected a coordinate and value from one sparse tensor stream")
                }
            })
            .map(|result| {
                result.and_then(|(coord, n)| {
                    if n.is_infinite() {
                        Err(TCError::unsupported(ERR_INF))
                    } else if n.is_nan() {
                        Err(TCError::unsupported(ERR_NAN))
                    } else {
                        Ok((coord, n))
                    }
                })
            });

        Ok(Box::pin(combined))
    }
}

impl<FD, FS, D, T, L, R> TensorAccess for SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        (self.combinator)(self.left.dtype().zero(), self.right.dtype().zero()).class()
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
impl<FD, FS, D, T, L, R> SparseAccess<FD, FS, D, T> for SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Slice = SparseCombinator<FD, FS, D, T, L::Slice, R::Slice>;
    type Transpose = SparseCombinator<FD, FS, D, T, L::Transpose, R::Transpose>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Combine(Box::new(SparseCombinator {
            left: self.left.accessor(),
            right: self.right.accessor(),
            combinator: self.combinator,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let zero = self.dtype().zero();
        let filled_inner = self.filled_inner(txn).await?;
        let filled = filled_inner.try_filter(move |(_, value)| future::ready(*value != zero));

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        self.shape().validate_axes(&axes)?;

        if axes.is_empty() {
            return Ok(Box::pin(stream::empty()));
        }

        let shape = {
            let shape = self.shape();
            axes.iter().map(|x| shape[*x]).collect()
        };

        let (left, right) = try_join!(
            self.left.filled_at(txn.clone(), axes.clone()),
            self.right.filled_at(txn, axes)
        )?;

        let filled_at = CoordMerge::new(left, right, shape, PER_BLOCK);
        Ok(Box::pin(filled_at))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let filled = self.filled(txn).await?;

        filled
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        debug!("SparseCombinator::slice {}", bounds);

        let left = self.left.slice(bounds.clone())?;
        let right = self.right.slice(bounds)?;
        assert_eq!(left.shape(), right.shape());

        Ok(SparseCombinator {
            left,
            right,
            combinator: self.combinator,
            phantom: self.phantom,
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
            phantom: self.phantom,
        })
    }
}

impl<FD, FS, D, T, L, R> ReadValueAt<D> for SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let left = self.left.read_value_at(txn.clone(), coord.to_vec());
            let right = self.right.read_value_at(txn, coord);
            let ((coord, left), (_, right)) = try_join!(left, right)?;
            let value = (self.combinator)(left, right);
            Ok((coord, value))
        })
    }
}

impl<FD, FS, D, T, L, R> fmt::Display for SparseCombinator<FD, FS, D, T, L, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("the result of combining two sparse Tensors")
    }
}

#[derive(Clone)]
pub struct SparseConstCombinator<FD, FS, D, T, A> {
    source: A,
    other: Number,
    combinator: fn(Number, Number) -> Number,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseConstCombinator<FD, FS, D, T, A> {
    pub fn new(source: A, other: Number, combinator: fn(Number, Number) -> Number) -> Self {
        Self {
            source,
            other,
            combinator,
            phantom: Phantom::default(),
        }
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseConstCombinator<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        Ord::max(self.source.dtype(), self.other.class())
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
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseConstCombinator<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Slice = SparseConstCombinator<FD, FS, D, T, A::Slice>;
    type Transpose = SparseConstCombinator<FD, FS, D, T, A::Transpose>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        let this = SparseConstCombinator {
            source: self.source.accessor(),
            other: self.other,
            combinator: self.combinator,
            phantom: self.phantom,
        };

        SparseAccessor::CombineConst(Box::new(this))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let combinator = self.combinator;
        let other = self.other;
        let filled = self.source.filled(txn).await?;
        Ok(Box::pin(filled.map_ok(move |(coord, value)| {
            (coord, combinator(value, other))
        })))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let slice = self.source.slice(bounds)?;
        Ok(SparseConstCombinator::new(
            slice,
            self.other,
            self.combinator,
        ))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let transpose = self.source.transpose(permutation)?;
        Ok(SparseConstCombinator::new(
            transpose,
            self.other,
            self.combinator,
        ))
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseConstCombinator<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let combinator = self.combinator;
            let other = self.other;

            self.source
                .read_value_at(txn, coord)
                .map_ok(|(coord, val)| (coord, combinator(val, other)))
                .await
        })
    }
}

impl<FD, FS, D, T, A> fmt::Display for SparseConstCombinator<FD, FS, D, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("sparse constant combinator")
    }
}

#[derive(Clone)]
pub struct SparseLeftCombinator<FD, FS, D, T, L, R> {
    left: L,
    right: R,
    combinator: fn(Number, Number) -> Number,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, L, R> SparseLeftCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    pub fn new(left: L, right: R, combinator: fn(Number, Number) -> Number) -> TCResult<Self> {
        if left.shape() != right.shape() {
            return Err(TCError::unsupported(
                "tried to combine SparseTensors with different shapes",
            ));
        }

        Ok(SparseLeftCombinator {
            left,
            right,
            combinator,
            phantom: Phantom::default(),
        })
    }

    pub async fn filled_inner<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        debug!(
            "SparseLeftCombinator::filled_inner ({}, {})",
            self.left, self.right
        );

        let left = self.left.filled(txn.clone()).await?;

        let combinator = self.combinator;
        let right = self.right;

        let filled = left
            .and_then(move |(coord, left_value)| {
                right
                    .clone()
                    .read_value_at(txn.clone(), coord)
                    .map_ok(move |(coord, right_value)| (coord, left_value, right_value))
            })
            .map_ok(move |(coord, left, right)| (coord, combinator(left, right)));

        Ok(Box::pin(filled))
    }
}

impl<FD, FS, D, T, L, R> TensorAccess for SparseLeftCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        (self.combinator)(self.left.dtype().zero(), self.right.dtype().zero()).class()
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
impl<FD, FS, D, T, L, R> SparseAccess<FD, FS, D, T> for SparseLeftCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Slice = SparseLeftCombinator<FD, FS, D, T, L::Slice, R::Slice>;
    type Transpose = SparseLeftCombinator<FD, FS, D, T, L::Transpose, R::Transpose>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::CombineLeft(Box::new(SparseLeftCombinator {
            left: self.left.accessor(),
            right: self.right.accessor(),
            combinator: self.combinator,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let zero = self.dtype().zero();
        let filled_inner = self.filled_inner(txn).await?;
        let filled = filled_inner.try_filter(move |(_, value)| future::ready(value != &zero));
        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        self.left.filled_at(txn, axes).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let filled = self.filled(txn).await?;

        filled
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        debug!("SparseLeftCombinator::slice left {} {}", self.left, bounds);
        let left = self.left.slice(bounds.clone())?;

        debug!(
            "SparseLeftCombinator::slice right {} {}",
            self.right, bounds
        );
        let right = self.right.slice(bounds)?;

        assert_eq!(left.shape(), right.shape());

        Ok(SparseLeftCombinator {
            left,
            right,
            combinator: self.combinator,
            phantom: self.phantom,
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let left = self.left.transpose(permutation.clone())?;
        let right = self.right.transpose(permutation)?;
        assert_eq!(left.shape(), right.shape());

        Ok(SparseLeftCombinator {
            left,
            right,
            combinator: self.combinator,
            phantom: self.phantom,
        })
    }
}

impl<FD, FS, D, T, L, R> ReadValueAt<D> for SparseLeftCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let left_zero = self.left.dtype().zero();
            let zero = self.dtype().zero();

            let (coord, left) = self.left.read_value_at(txn.clone(), coord).await?;
            let (coord, value) = if left == left_zero {
                (coord, zero)
            } else {
                let (coord, right) = self.right.read_value_at(txn, coord).await?;
                let value = (self.combinator)(left, right);
                (coord, value)
            };

            Ok((coord, value))
        })
    }
}

impl<FD, FS, D, T, L, R> fmt::Display for SparseLeftCombinator<FD, FS, D, T, L, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("the result of combining two sparse Tensors, ignoring the right Tensor where the left Tensor is zero")
    }
}

#[derive(Clone)]
pub struct SparseExpand<FD, FS, D, T, A> {
    source: A,
    rebase: transform::Expand,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseExpand<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    pub fn new(source: A, axis: usize) -> TCResult<Self> {
        let rebase = transform::Expand::new(source.shape().clone(), axis)?;
        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseExpand<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
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
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseExpand<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Slice = SparseAccessor<FD, FS, D, T>;
    type Transpose = SparseExpand<FD, FS, D, T, A::Transpose>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Expand(Box::new(SparseExpand {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rebase = self.rebase;
        let filled = self.source.filled(txn).await?;
        let filled = filled.map_ok(move |(coord, value)| (rebase.map_coord(coord), value));
        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        debug!("SparseExpand::filled_at {:?}", axes);

        self.shape().validate_axes(&axes)?;

        if axes.is_empty() {
            return Ok(Box::pin(stream::empty()));
        }

        let mut i = 0;
        let expand = loop {
            if i >= axes.len() {
                break None;
            } else if axes[i] == self.rebase.expand_axis() {
                break Some(i);
            } else {
                i += 1;
            }
        };

        debug!(
            "SparseExpand::filled_at {:?} will expand axis {:?}",
            axes, expand
        );

        let source_axes = self.rebase.invert_axes(axes);
        let source = self.source.filled_at(txn, source_axes).await?;
        if let Some(x) = expand {
            let filled_at = source.map_ok(move |coords| coords.expand_dim(x));
            Ok(Box::pin(filled_at))
        } else {
            Ok(source)
        }
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    fn slice(self, mut bounds: Bounds) -> TCResult<Self::Slice> {
        debug!("SparseExpand {} slice {}", self.shape(), bounds);
        self.shape().validate_bounds(&bounds)?;
        bounds.normalize(self.shape());

        if bounds == Bounds::all(self.shape()) {
            return Ok(self.accessor());
        }

        let source_bounds = self.rebase.invert_bounds(bounds.clone());
        let source_slice_shape = source_bounds.to_shape(self.source.shape())?;

        let expand_axis = self.rebase.expand_axis();
        let source_expand_axis = if bounds[expand_axis].is_index() {
            // in this case the expanded dimension is elided
            None
        } else if source_slice_shape.len() == 1 {
            // in this case only the expanded dimension is present in the slice
            None
        } else {
            let num_elided = bounds[..expand_axis]
                .iter()
                .filter(|bound| bound.is_index())
                .count();

            Some(expand_axis - num_elided)
        };

        debug!(
            "SparseExpand slice source {} with bounds {} (removed axis {}, expanding axis {:?})",
            self.source.shape(),
            source_bounds,
            expand_axis,
            source_expand_axis,
        );

        let slice = self.source.slice(source_bounds)?;
        debug_assert_eq!(&source_slice_shape, slice.shape());

        if let Some(source_expand_axis) = source_expand_axis {
            SparseExpand::new(slice, source_expand_axis).map(|expansion| expansion.accessor())
        } else {
            Ok(slice.accessor())
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let expand_axis = if let Some(permutation) = &permutation {
            if permutation.len() != self.ndim() {
                return Err(TCError::unsupported(format!(
                    "invalid permutation for tensor of shape {}: {:?}",
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
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseExpand<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));

        Box::pin(read)
    }
}

impl<FD, FS, D, T, A> fmt::Display for SparseExpand<FD, FS, D, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sparse Tensor expansion")
    }
}

type Reductor<FD, FS, D, T> =
    fn(&SparseTensor<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>, T) -> TCBoxTryFuture<Number>;

#[derive(Clone)]
pub struct SparseReduce<FD, FS, D, T> {
    source: SparseAccessor<FD, FS, D, T>,
    rebase: transform::Reduce,
    reductor: Reductor<FD, FS, D, T>,
}

impl<FD, FS, D, T> SparseReduce<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    pub fn new(
        source: SparseAccessor<FD, FS, D, T>,
        axis: usize,
        reductor: Reductor<FD, FS, D, T>,
    ) -> TCResult<Self> {
        transform::Reduce::new(source.shape().clone(), axis).map(|rebase| SparseReduce {
            source,
            rebase,
            reductor,
        })
    }
}

impl<FD, FS, D, T> TensorAccess for SparseReduce<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
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
impl<FD, FS, D, T> SparseAccess<FD, FS, D, T> for SparseReduce<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Slice = SparseReduce<FD, FS, D, T>;
    type Transpose = SparseTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Reduce(Box::new(self))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        debug!("SparseReduce::filled");

        let filled_at = self
            .clone()
            .filled_at(txn.clone(), (0..self.ndim()).collect())
            .await?;

        let zero = self.dtype().zero();
        let rebase = self.rebase;
        let reductor = self.reductor;
        let source = self.source;

        let filled = filled_at
            .map_ok(|coords| stream::iter(coords.to_vec()).map(TCResult::Ok))
            .try_flatten()
            .map_ok(move |coord| {
                let source_bounds = rebase.invert_coord(&coord);
                debug!("reduce {:?} (bounds {})", coord, source_bounds);
                let source = source.clone();
                let txn = txn.clone();
                Box::pin(async move {
                    let slice = source.slice(source_bounds)?;
                    let value = reductor(&slice.into(), txn).await?;
                    Ok((coord, value))
                })
            })
            .try_buffered(num_cpus::get())
            .try_filter(move |(_coord, value)| future::ready(value != &zero));

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        debug!("SparseReduce::filled_at {:?}", axes);

        let source_axes = self.rebase.invert_axes(axes);
        self.source.filled_at(txn, source_axes).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let axes = (0..self.ndim()).collect();
        let filled = self.filled_at(txn, axes).await?;

        filled
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        debug!("SparseReduce::slice {}", bounds);

        self.shape().validate_bounds(&bounds)?;

        let reduce_axis = self.rebase.reduce_axis(&bounds);
        let source_bounds = self.rebase.invert_bounds(bounds);
        debug!(
            "SparseReduce::slice source is {}, bounds are {}, source axis to reduce is {}",
            self.source, source_bounds, reduce_axis
        );

        let source = self.source.slice(source_bounds)?;
        let reduced = SparseReduce::new(source.into(), reduce_axis, self.reductor)?;
        if reduced.ndim() == 0 {
            Err(TCError::unsupported(
                "cannot return a zero-dimensional slice from a reduced Tensor",
            ))
        } else {
            Ok(reduced)
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        SparseTranspose::new(self, permutation)
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseReduce<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let source_bounds = self.rebase.invert_coord(&coord);
            let reductor = self.reductor;
            let slice = self.source.slice(source_bounds)?;
            let value = reductor(&slice.into(), txn).await?;
            Ok((coord, value))
        })
    }
}

impl<FD, FS, D, T> fmt::Display for SparseReduce<FD, FS, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sparse Tensor reduction")
    }
}

#[derive(Clone)]
pub struct SparseTranspose<FD, FS, D, T, A> {
    source: A,
    rebase: transform::Transpose,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseTranspose<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    pub fn new(source: A, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        transform::Transpose::new(source.shape().clone(), permutation).map(|rebase| Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseTranspose<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
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
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseTranspose<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
    A::Slice: SparseAccess<FD, FS, D, T>,
{
    type Slice = <A::Slice as SparseAccess<FD, FS, D, T>>::Transpose;
    type Transpose = A::Transpose;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Transpose(Box::new(SparseTranspose {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: self.phantom,
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rebase = self.rebase.clone();
        let source_axes = self.rebase.invert_axes((0..self.ndim()).collect());
        let filled_at = self
            .source
            .clone()
            .filled_at(txn.clone(), source_axes)
            .await?;

        let coords = filled_at.map_ok(move |coords| rebase.map_coords(coords));
        let filled = sorted_values::<FD, FS, T, D, _, _>(txn, self, coords).await?;
        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        debug!(
            "SparseTranspose::filled_at {:?} (source is {}, shape is {})",
            axes,
            self.source,
            self.shape()
        );

        if axes.is_empty() {
            return Ok(Box::pin(stream::empty()));
        }

        let shape = self.shape();
        let shape = axes.iter().map(|x| shape[*x]).collect();
        let source_axes = self.rebase.invert_axes(axes);
        let permutation = self.rebase.map_axes(&source_axes);
        let source = self.source.filled_at(txn.clone(), source_axes).await?;
        let filled_at = source.map_ok(move |coords| coords.transpose(Some(&permutation)));
        let filled_at = sorted_coords::<FD, FS, D, T, _>(&txn, shape, filled_at).await?;
        Ok(Box::pin(filled_at))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        debug!("SparseTranspose::slice {}", bounds);
        self.shape().validate_bounds(&bounds)?;

        let slice_permutation = self.rebase.invert_permutation(&bounds);
        debug!(
            "slice permutation {}{} is {:?}",
            self.shape(),
            bounds,
            slice_permutation
        );

        let source_bounds = self.rebase.invert_bounds(&bounds);
        let source = self.source.slice(source_bounds)?;
        source.transpose(Some(slice_permutation))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let permutation = permutation.map(|axes| self.rebase.invert_axes(axes));
        self.source.transpose(permutation)
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseTranspose<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));

        Box::pin(read)
    }
}

impl<FD, FS, D, T, A> fmt::Display for SparseTranspose<FD, FS, D, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a transposed sparse Tensor")
    }
}

#[derive(Clone)]
pub struct SparseUnary<FD, FS, D, T> {
    source: SparseAccessor<FD, FS, D, T>, // TODO: can this be a type parameter A: SparseAccess?
    dtype: NumberType,
    transform: fn(Number) -> Number,
}

impl<FD, FS, D, T> SparseUnary<FD, FS, D, T> {
    pub fn new(
        source: SparseAccessor<FD, FS, D, T>,
        transform: fn(Number) -> Number,
        dtype: NumberType,
    ) -> Self {
        Self {
            source,
            dtype,
            transform,
        }
    }
}

impl<FD, FS, D, T> TensorAccess for SparseUnary<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
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
impl<FD, FS, D, T> SparseAccess<FD, FS, D, T> for SparseUnary<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Slice = Self;
    type Transpose = Self;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Unary(Box::new(self))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let transform = self.transform;
        let filled = self.source.filled(txn).await?;
        let cast = filled.map_ok(move |(coord, value)| (coord, transform(value)));
        Ok(Box::pin(cast))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
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
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseUnary<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        let dtype = self.dtype;
        let read = self
            .source
            .read_value_at(txn, coord)
            .map_ok(move |(coord, value)| (coord, value.into_type(dtype)));

        Box::pin(read)
    }
}

impl<FD, FS, D, T> fmt::Display for SparseUnary<FD, FS, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a unary operation on a sparse Tensor")
    }
}
