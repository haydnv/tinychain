use std::fmt;

use afarray::{Array, CoordBlocks, CoordMerge, CoordUnique, Coords};
use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::try_join;
use log::{debug, trace};

use tc_error::*;
use tc_table::{Node, NodeId};
use tc_transact::fs::{Dir, DirCreateFile, File};
use tc_transact::{Transaction, TxnId};
use tc_value::{FloatInstance, Number, NumberClass, NumberInstance, NumberType};
use tcgeneric::{TCBoxTryFuture, TCBoxTryStream, Tuple};

use crate::dense::{DenseAccess, DenseAccessor, DenseTensor, PER_BLOCK};
use crate::stream::{sorted_coords, sorted_values, Read, ReadValueAt};
use crate::{
    coord_bounds, needs_transpose, transform, AxisBounds, Bounds, Coord, Phantom, Shape,
    TensorAccess, TensorUnary, ERR_INF, ERR_NAN,
};

use super::combine::{coord_to_offset, SparseCombine};
use super::table::{SparseTable, SparseTableSlice};
use super::{SparseRow, SparseStream, SparseTensor};

/// Access methods for [`SparseTensor`] data
#[async_trait]
pub trait SparseAccess<FD, FS, D, T>:
    ReadValueAt<D, Txn = T> + TensorAccess + Clone + fmt::Display + Send + Sync + 'static
where
    D: Dir,
    T: Transaction<D>,
{
    /// The type of a slice of this accessor
    type Slice: SparseAccess<FD, FS, D, T>;

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

    /// Return `true` if this [`SparseTensor`] contains no elements.
    async fn is_empty(self, txn: T) -> TCResult<bool>;

    /// Return a slice of this accessor with the given [`Bounds`].
    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice>;

    /// Return this accessor as transposed according to the given `permutation`.
    ///
    /// If no permutation is given, this accessor's axes will be reversed.
    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>>;
}

/// Write methods for [`SparseTensor`] data
#[async_trait]
pub trait SparseWrite: Send + Sync {
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
    Flip(Box<SparseFlip<FD, FS, D, T, Self>>),
    Slice(SparseTableSlice<FD, FS, D, T>),
    Reduce(Box<SparseReduce<FD, FS, D, T>>),
    Reshape(Box<SparseReshape<FD, FS, D, T, Self>>),
    Table(SparseTable<FD, FS, D, T>),
    Transpose(Box<SparseTranspose<FD, FS, D, T, Self>>),
    Unary(Box<SparseUnary<FD, FS, D, T>>),
}

macro_rules! dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Broadcast($var) => $call,
            Self::Cast($var) => $call,
            Self::Combine($var) => $call,
            Self::CombineConst($var) => $call,
            Self::CombineLeft($var) => $call,
            Self::Dense($var) => $call,
            Self::Expand($var) => $call,
            Self::Flip($var) => $call,
            Self::Slice($var) => $call,
            Self::Reduce($var) => $call,
            Self::Reshape($var) => $call,
            Self::Table($var) => $call,
            Self::Transpose($var) => $call,
            Self::Unary($var) => $call,
        }
    };
}

impl<FD, FS, D, T> TensorAccess for SparseAccessor<FD, FS, D, T> {
    fn dtype(&self) -> NumberType {
        dispatch!(self, this, this.dtype())
    }

    fn ndim(&self) -> usize {
        dispatch!(self, this, this.ndim())
    }

    fn shape(&self) -> &Shape {
        dispatch!(self, this, this.shape())
    }

    fn size(&self) -> u64 {
        dispatch!(self, this, this.size())
    }
}

#[async_trait]
impl<FD, FS, D, T> SparseAccess<FD, FS, D, T> for SparseAccessor<FD, FS, D, T>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = Self;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        self
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        dispatch!(self, this, this.filled(txn).await)
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        dispatch!(self, this, this.filled_at(txn, axes).await)
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        dispatch!(self, this, this.filled_count(txn).await)
    }

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        dispatch!(self, this, this.is_empty(txn).await)
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        dispatch!(self, this, this.slice(bounds).map(SparseAccess::accessor))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        dispatch!(
            self,
            this,
            this.transpose(permutation).map(SparseAccess::accessor)
        )
    }
}

#[async_trait]
impl<FD, FS, D, T> SparseWrite for SparseAccessor<FD, FS, D, T>
where
    FD: Send + Sync,
    FS: Send + Sync,
    D: Send + Sync,
    T: Send + Sync,
    SparseTable<FD, FS, D, T>: SparseWrite,
{
    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        match self {
            Self::Table(table) => table.write_value(txn_id, coord, value).await,
            _ => Err(bad_request!("cannot write to a Tensor view")),
        }
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseAccessor<FD, FS, D, T>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    D::Write: DirCreateFile<FD>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        dispatch!(self, this, this.read_value_at(txn, coord))
    }
}

impl<FD, FS, D, T> fmt::Display for SparseAccessor<FD, FS, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        dispatch!(self, this, fmt::Display::fmt(this, f))
    }
}

#[derive(Clone)]
pub struct DenseToSparse<FD, FS, D, T, B> {
    source: B,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> TensorAccess for DenseToSparse<FD, FS, D, T, B>
where
    B: TensorAccess,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = DenseToSparse<FD, FS, D, T, B::Slice>;

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
        debug!("DenseToSparse::filled_at {:?}", axes);

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

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        DenseTensor::from(self.source)
            .any(txn)
            .map_ok(|any| !any)
            .await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.source.slice(bounds).map(DenseToSparse::from)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("DenseToSparse::transpose {:?}", permutation);

        if needs_transpose(&permutation) {
            self.source
                .transpose(permutation)
                .map(DenseToSparse::from)
                .map(SparseAccess::accessor)
        } else {
            Ok(self.accessor())
        }
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for DenseToSparse<FD, FS, D, T, B>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
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

    #[allow(dead_code)]
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseBroadcast<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
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
    A: TensorAccess,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseAccessor<FD, FS, D, T>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Broadcast(Box::new(SparseBroadcast {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        debug!("SparseBroadcast::filled with source {}", self.source);

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
            "SparseBroadcast::filled_at {:?} (source is {}, shape is {})",
            axes,
            self.source,
            self.rebase.shape()
        );

        if axes.is_empty() {
            return Ok(Box::pin(stream::empty()));
        }

        // TODO: optimize for the case when none of the `axes` have been broadcast

        self.shape().validate_axes(&axes)?;

        let shape = Shape::from({
            let shape = self.shape();
            axes.iter().map(|x| shape[*x]).collect::<Vec<u64>>()
        });

        let ndim = axes.len();
        let rebase = self.rebase;
        let source_axes = (0..self.source.ndim()).collect();

        let filled_at = self.source.filled_at(txn.clone(), source_axes).await?;
        let filled_at = filled_at
            .map_ok(|coords| stream::iter(coords.to_vec()).map(TCResult::Ok))
            .try_flatten()
            .map_ok(move |coord| {
                trace!("broadcast source coord {:?}", coord);

                let bounds = rebase.map_coord(coord);
                let bounds: Bounds = bounds
                    .into_iter()
                    .enumerate()
                    .filter_map(|(x, bound)| if axes.contains(&x) { Some(bound) } else { None })
                    .collect();

                assert_eq!(bounds.len(), ndim);
                bounds
            })
            // TODO: can this happen in `Coords`? maybe a new stream generator type?
            .inspect_ok(|bounds: &Bounds| trace!("broadcast coord mapped to bounds {}", bounds))
            .map_ok(move |bounds| stream::iter(bounds.affected().map(TCResult::Ok)))
            .try_flatten();

        let filled_at = CoordBlocks::new(filled_at, ndim, PER_BLOCK);

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

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        self.source.is_empty(txn).await
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

        if source.shape() == &shape {
            Ok(source.accessor())
        } else {
            SparseBroadcast::new(source, shape).map(SparseAccess::accessor)
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseBroadcast::transpose {:?}", permutation);

        if needs_transpose(&permutation) {
            SparseTranspose::new(self, permutation).map(SparseAccess::accessor)
        } else {
            Ok(self.accessor())
        }
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseBroadcast<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
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
    A: TensorAccess,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseCast<FD, FS, D, T, A::Slice>;

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
        debug!("SparseCast::filled_at {:?}", axes);

        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        self.source.is_empty(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        Ok(SparseCast {
            source: self.source.slice(bounds)?,
            dtype: self.dtype,
            phantom: self.phantom,
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseCast::transpose {:?}", permutation);

        if needs_transpose(&permutation) {
            let source = self.source.transpose(permutation)?;
            let cast = SparseCast {
                source,
                dtype: self.dtype,
                phantom: self.phantom,
            };

            Ok(cast.accessor())
        } else {
            Ok(self.accessor())
        }
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseCast<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    pub fn new(left: L, right: R, combinator: fn(Number, Number) -> Number) -> TCResult<Self> {
        if left.shape() != right.shape() {
            return Err(bad_request!(
                "tried to combine SparseTensors with different shapes"
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
                        Err(bad_request!("{}", ERR_INF))
                    } else if n.is_nan() {
                        Err(bad_request!("{}", ERR_NAN))
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
    L: TensorAccess,
    R: TensorAccess,
{
    fn dtype(&self) -> NumberType {
        (self.combinator)(self.left.dtype().one(), self.right.dtype().one()).class()
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseCombinator<FD, FS, D, T, L::Slice, R::Slice>;

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
        let filled = filled_inner
            .try_filter(move |(_, value)| future::ready(*value != zero))
            .inspect_ok(|(coord, value)| {
                trace!("SparseCombinator filled at {:?}: {}", coord, value)
            });

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        debug!("SparseCombinator::filled_at {:?}", axes);

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

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        let mut filled = self.filled(txn).await?;
        filled.try_next().map_ok(|row| row.is_none()).await
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

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseCombinator::transpose {:?}", permutation);

        if !needs_transpose(&permutation) {
            return Ok(self.accessor());
        }

        let left = self.left.transpose(permutation.clone())?;
        let right = self.right.transpose(permutation)?;
        assert_eq!(left.shape(), right.shape());

        let transpose = SparseCombinator {
            left,
            right,
            combinator: self.combinator,
            phantom: self.phantom,
        };

        Ok(transpose.accessor())
    }
}

impl<FD, FS, D, T, L, R> ReadValueAt<D> for SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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
    A: TensorAccess,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseConstCombinator<FD, FS, D, T, A::Slice>;

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
        let zero = self.dtype().zero();
        let combinator = self.combinator;
        let other = self.other;
        let filled = self.source.filled(txn).await?;
        let filled = filled
            .map_ok(move |(coord, value)| (coord, combinator(value, other)))
            .try_filter(move |(_coord, value)| future::ready(value != &zero));

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        debug!("SparseConstCombinator::filled_at {:?}", axes);
        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        let mut filled = self.filled(txn).await?;
        filled.try_next().map_ok(|row| row.is_none()).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let slice = self.source.slice(bounds)?;
        Ok(SparseConstCombinator::new(
            slice,
            self.other,
            self.combinator,
        ))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseConstCombinator::transpose {:?}", permutation);

        if !needs_transpose(&permutation) {
            return Ok(self.accessor());
        }

        let source = self.source.transpose(permutation)?;
        let transpose = SparseConstCombinator::new(source, self.other, self.combinator);

        Ok(transpose.accessor())
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseConstCombinator<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    pub fn new(left: L, right: R, combinator: fn(Number, Number) -> Number) -> TCResult<Self> {
        if left.shape() != right.shape() {
            return Err(bad_request!(
                "tried to combine SparseTensors with different shapes"
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
            .inspect_ok(|(coord, left, right)| {
                trace!("left combine {} with {} at {:?}", left, right, coord)
            })
            .map_ok(move |(coord, left, right)| (coord, combinator(left, right)));

        Ok(Box::pin(filled))
    }
}

impl<FD, FS, D, T, L, R> TensorAccess for SparseLeftCombinator<FD, FS, D, T, L, R>
where
    L: TensorAccess,
    R: TensorAccess,
{
    fn dtype(&self) -> NumberType {
        (self.combinator)(self.left.dtype().one(), self.right.dtype().one()).class()
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseLeftCombinator<FD, FS, D, T, L::Slice, R::Slice>;

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
        let filled = filled_inner
            .try_filter(move |(_, value)| future::ready(value != &zero))
            .inspect_ok(|(coord, value)| {
                trace!("SparseLeftCombinator filled at {:?}: {}", coord, value)
            });

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        let ndim = self.ndim();
        let shape: Shape = axes.iter().map(|x| self.shape()[*x]).collect();
        let filled = self.filled(txn.clone()).await?;
        let filled = filled.map_ok(|(coord, _value)| coord);

        let coords = CoordBlocks::new(filled, ndim, PER_BLOCK);
        let filled_at = coords.map_ok(move |coords| coords.get(&axes));
        let unique = sorted_coords::<FD, FS, D, T, _>(&txn, shape.clone(), filled_at).await?;

        Ok(Box::pin(CoordUnique::new(
            unique,
            shape.to_vec(),
            PER_BLOCK,
        )))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let filled = self.filled(txn).await?;

        filled
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        let mut filled = self.filled(txn).await?;
        filled.try_next().map_ok(|row| row.is_none()).await
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

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseLeftCombinator::transpose {:?}", permutation);

        if !needs_transpose(&permutation) {
            return Ok(self.accessor());
        }

        let left = self.left.transpose(permutation.clone())?;
        let right = self.right.transpose(permutation)?;
        assert_eq!(left.shape(), right.shape());

        let transpose = SparseLeftCombinator {
            left,
            right,
            combinator: self.combinator,
            phantom: self.phantom,
        };

        Ok(transpose.accessor())
    }
}

impl<FD, FS, D, T, L, R> ReadValueAt<D> for SparseLeftCombinator<FD, FS, D, T, L, R>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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

    #[allow(dead_code)]
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseExpand<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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
    A: TensorAccess,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseAccessor<FD, FS, D, T>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Expand(Box::new(SparseExpand {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        debug!("SparseExpand::filled");

        let rebase = self.rebase;
        let filled = self.source.filled(txn).await?;
        let filled = filled
            .map_ok(move |(coord, value)| (rebase.map_coord(coord), value))
            .inspect_ok(|(coord, value)| trace!("SparseExpand::filled at {:?}: {}", coord, value));

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

        let mut filled = self.source.clone().filled(txn.clone()).await?;
        while let Some((coord, value)) = filled.try_next().await? {
            trace!("source filled at {:?}: {}", coord, value)
        }

        let source_axes = self.rebase.invert_axes(axes);
        if source_axes.is_empty() {
            return if self.source.is_empty(txn).await? {
                Ok(Box::pin(stream::empty()))
            } else {
                // the expanded axis has a size of one, so only coordinate 0 can possibly be filled
                let filled_at = Coords::from_iter([vec![0]], 1);
                Ok(Box::pin(stream::once(future::ready(Ok(filled_at)))))
            };
        }

        debug!(
            "SparseExpand::filled_at will expand axis {:?} of source {} axes {:?}",
            expand, self.source, source_axes
        );

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

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        self.source.is_empty(txn).await
    }

    fn slice(self, mut bounds: Bounds) -> TCResult<Self::Slice> {
        debug!(
            "SparseExpand {} axis {} slice {}",
            self.shape(),
            self.rebase.expand_axis(),
            bounds
        );

        self.shape().validate_bounds(&bounds)?;
        bounds.normalize(self.shape());
        let ndim = bounds.ndim();

        let expand_axis = self.rebase.invert_axis(&bounds);
        let bounds = self.rebase.invert_bounds(bounds);
        let source = self.source.slice(bounds)?;

        if ndim == source.ndim() {
            Ok(source.accessor())
        } else if let Some(axis) = expand_axis {
            SparseExpand::new(source, axis).map(SparseAccess::accessor)
        } else {
            Ok(source.accessor())
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseExpand::transpose {:?}", permutation);

        if !needs_transpose(&permutation) {
            return Ok(self.accessor());
        }

        let permutation =
            permutation.unwrap_or_else(|| (0..self.ndim()).into_iter().rev().collect());

        assert_eq!(permutation.len(), self.ndim());

        let mut expand_axis = None;
        for i in 0..permutation.len() {
            if permutation[i] == self.rebase.expand_axis() {
                expand_axis = Some(i);
            }
        }
        let expand_axis = expand_axis.expect("expand axis");

        let permutation = self.rebase.invert_axes(permutation);
        let source = self.source.transpose(Some(permutation))?;
        SparseExpand::new(source, expand_axis).map(SparseAccess::accessor)
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseExpand<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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

#[derive(Clone)]
pub struct SparseFlip<FD, FS, D, T, A> {
    source: A,
    rebase: transform::Flip,

    #[allow(dead_code)]
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseFlip<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    pub fn new(source: A, axis: usize) -> TCResult<Self> {
        let rebase = transform::Flip::new(source.shape().clone(), axis)?;

        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseFlip<FD, FS, D, T, A>
where
    A: TensorAccess,
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
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseFlip<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseAccessor<FD, FS, D, T>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Flip(Box::new(SparseFlip {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rebase = self.rebase;
        let filled = self.source.filled(txn).await?;
        let filled = filled.map_ok(move |(coord, value)| (rebase.flip_coord(coord), value));
        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        debug!("SparseExpand::filled_at {:?}", axes);

        self.shape().validate_axes(&axes)?;

        if axes.is_empty() {
            return Ok(Box::pin(stream::empty()));
        }

        let shape = self.shape().clone();
        let rebase = self.rebase;

        let coords = self.source.filled_at(txn.clone(), axes).await?;
        let coords = coords.map_ok(move |coords| rebase.flip_coords(coords));
        let coords = sorted_coords::<FD, FS, D, T, _>(&txn, shape, coords).await?;
        Ok(Box::pin(coords))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        self.source.is_empty(txn).await
    }

    fn slice(self, mut bounds: Bounds) -> TCResult<Self::Slice> {
        debug!("SparseFlip {} slice {}", self.shape(), bounds);

        self.shape().validate_bounds(&bounds)?;
        bounds.normalize(self.shape());

        if bounds == Bounds::all(self.shape()) {
            return Ok(self.accessor());
        }

        if let Some(flip_axis) = self.rebase.invert_axis(&bounds) {
            let source_bounds = self.rebase.flip_bounds(bounds);
            let slice = self.source.slice(source_bounds)?;
            SparseFlip::new(slice, flip_axis).map(|slice| slice.accessor())
        } else {
            let source_bounds = self.rebase.flip_bounds(bounds);
            self.source
                .slice(source_bounds)
                .map(|slice| slice.accessor())
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseFlip::transpose {:?}", permutation);

        if !needs_transpose(&permutation) {
            return Ok(self.accessor());
        }

        let flip_axis = if let Some(permutation) = &permutation {
            if permutation.len() == self.ndim() {
                permutation[self.rebase.axis()]
            } else {
                return Err(bad_request!(
                    "invalid permutation: {}",
                    permutation.iter().collect::<Tuple<&usize>>(),
                ));
            }
        } else {
            self.ndim() - self.rebase.axis()
        };

        let source = self.source.transpose(permutation)?;
        SparseFlip::new(source, flip_axis).map(SparseAccess::accessor)
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseFlip<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.flip_coord(coord.clone());
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));

        Box::pin(read)
    }
}

impl<FD, FS, D, T, A> fmt::Display for SparseFlip<FD, FS, D, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a flipped sparse Tensor")
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
{
    pub fn new(
        source: SparseAccessor<FD, FS, D, T>,
        axis: usize,
        keepdims: bool,
        reductor: Reductor<FD, FS, D, T>,
    ) -> TCResult<Self> {
        transform::Reduce::new(source.shape().clone(), axis, keepdims).map(|rebase| SparseReduce {
            source,
            rebase,
            reductor,
        })
    }
}

impl<FD, FS, D, T> TensorAccess for SparseReduce<FD, FS, D, T> {
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseReduce<FD, FS, D, T>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Reduce(Box::new(self))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        debug!("SparseReduce::filled");

        let filled_at = self
            .source
            .clone()
            .filled_at(
                txn.clone(),
                self.rebase
                    .invert_axes((0..self.ndim()).into_iter().collect()),
            )
            .await?;

        let ndim = self.ndim();
        let zero = self.dtype().zero();
        let rebase = self.rebase;
        let reductor = self.reductor;
        let source = self.source;

        let filled = filled_at
            .map_ok(|coords| stream::iter(coords.to_vec()).map(TCResult::Ok))
            .try_flatten()
            .map_ok(move |coord| {
                debug_assert_eq!(coord.len(), ndim);

                let source_bounds = rebase.invert_coord(&coord);
                trace!("reduce {:?} (bounds {})", coord, source_bounds);

                let source = source.clone();
                let txn = txn.clone();
                Box::pin(async move {
                    let slice = source.slice(source_bounds)?;
                    let value = reductor(&slice.into(), txn).await?;
                    Ok((coord, value))
                })
            })
            .try_buffered(num_cpus::get())
            .try_filter(move |(_coord, value)| future::ready(value != &zero))
            .inspect_ok(|(coord, value)| trace!("SparseReduce filled at {:?}: {}", coord, value));

        Ok(Box::pin(filled))
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<TCBoxTryStream<'a, Coords>> {
        let ndim = self.ndim();
        let shape: Vec<u64> = axes.iter().map(|x| self.shape()[*x]).collect();

        let filled = self.filled(txn.clone()).await?;
        let filled = filled.map_ok(|(coord, _value)| coord);
        let filled =
            CoordBlocks::new(filled, ndim, PER_BLOCK).map_ok(move |coords| coords.get(&axes));

        let sorted = sorted_coords::<FD, FS, D, T, _>(&txn, shape.to_vec().into(), filled).await?;

        let unique = CoordUnique::new(sorted, shape, PER_BLOCK);

        Ok(Box::pin(unique))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let axes = (0..self.ndim()).collect();
        let filled = self.filled_at(txn, axes).await?;

        filled
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        let mut filled = self.filled(txn).await?;
        filled.try_next().map_ok(|row| row.is_none()).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        debug!("SparseReduce::slice {} from {:?}", bounds, self.shape());

        self.shape().validate_bounds(&bounds)?;

        let reduce_axis = self.rebase.invert_axis(&bounds);
        let keepdims = self.ndim() == self.source.ndim();
        let source_bounds = self.rebase.invert_bounds(bounds);
        debug!(
            "SparseReduce::slice source is {}, bounds are {}, source axis to reduce is {}",
            self.source, source_bounds, reduce_axis
        );

        let source = self.source.slice(source_bounds)?;
        let reduced = SparseReduce::new(source.into(), reduce_axis, keepdims, self.reductor)?;
        debug_assert!(reduced.ndim() > 0);

        if reduced.ndim() == 0 {
            Err(bad_request!(
                "cannot return a zero-dimensional slice from a reduced Tensor"
            ))
        } else {
            Ok(reduced)
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseReduce::transpose {:?}", permutation);

        if needs_transpose(&permutation) {
            SparseTranspose::new(self, permutation).map(SparseAccess::accessor)
        } else {
            Ok(self.accessor())
        }
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseReduce<FD, FS, D, T>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    D::Write: DirCreateFile<FD>,
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
pub struct SparseReshape<FD, FS, D, T, A> {
    source: A,
    rebase: transform::Reshape,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseReshape<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    pub fn new(source: A, shape: Shape) -> TCResult<Self> {
        let rebase = transform::Reshape::new(source.shape().clone(), shape, "reshape")?;
        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseReshape<FD, FS, D, T, A>
where
    A: TensorAccess,
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
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseReshape<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseTable<FD, FS, D, T>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        let reshape = SparseReshape {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: self.phantom,
        };

        SparseAccessor::Reshape(Box::new(reshape))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        debug!("SparseReshape::filled");

        let rebase = self.rebase;
        let source = self.source.filled(txn).await?;
        Ok(Box::pin(source.map_ok(move |(coord, value)| {
            (rebase.map_coord(coord), value)
        })))
    }

    async fn filled_at<'a>(
        self,
        _txn: T,
        _axes: Vec<usize>,
    ) -> TCResult<TCBoxTryStream<'a, Coords>> {
        Err(bad_request!(
            "cannot slice a reshaped Tensor; make a copy first"
        ))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let axes = (0..self.ndim()).collect();
        let filled = self.filled_at(txn, axes).await?;

        filled
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        self.source.is_empty(txn).await
    }

    fn slice(self, _bounds: Bounds) -> TCResult<Self::Slice> {
        Err(bad_request!(
            "cannot slice a reshaped Tensor; make a copy first"
        ))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseReshape::transpose {:?}", permutation);

        if needs_transpose(&permutation) {
            SparseTranspose::new(self, permutation).map(SparseAccess::accessor)
        } else {
            Ok(self.accessor())
        }
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseReshape<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        self.source
            .read_value_at(txn, self.rebase.invert_coord(coord))
    }
}

impl<FD, FS, D, T, A> fmt::Display for SparseReshape<FD, FS, D, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a reshaped sparse Tensor")
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    pub fn new(source: A, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        if !needs_transpose(&permutation) {
            return Err(bad_request!(
                "sparse transpose got no axes to transpose: {}",
                permutation
                    .expect("permutation")
                    .iter()
                    .collect::<Tuple<&usize>>(),
            ));
        }

        transform::Transpose::new(source.shape().clone(), permutation).map(|rebase| Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseTranspose<FD, FS, D, T, A>
where
    A: TensorAccess,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    A::Slice: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = SparseAccessor<FD, FS, D, T>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Transpose(Box::new(SparseTranspose {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: self.phantom,
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        debug!("SparseTranspose::filled with source {}", self.source);

        let rebase = self.rebase.clone();
        let source_axes = (0..self.ndim()).collect();
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

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        self.source.is_empty(txn).await
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

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseTranspose::transpose {:?}", permutation);

        if !needs_transpose(&permutation) {
            return Ok(self.accessor());
        }

        let source_permutation = permutation.map(|axes| self.rebase.invert_axes(axes));

        if needs_transpose(&source_permutation) {
            self.source
                .transpose(source_permutation)
                .map(|access| access.accessor())
        } else {
            Ok(self.source.accessor())
        }
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseTranspose<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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

impl<FD, FS, D, T> TensorAccess for SparseUnary<FD, FS, D, T> {
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    D::Write: DirCreateFile<FD>,
{
    type Slice = Self;

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
        debug!("SparseUnary::filled_at {:?}", axes);
        self.source.filled_at(txn, axes).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn is_empty(self, txn: T) -> TCResult<bool> {
        self.source.is_empty(txn).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let source = self.source.slice(bounds)?;
        Ok(SparseUnary {
            source: source.accessor(),
            dtype: self.dtype,
            transform: self.transform,
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<SparseAccessor<FD, FS, D, T>> {
        debug!("SparseUnary::transpose {:?}", permutation);

        if needs_transpose(&permutation) {
            let source = self.source.transpose(permutation)?;
            let transpose = SparseUnary {
                source: source.accessor(),
                dtype: self.dtype,
                transform: self.transform,
            };

            Ok(transpose.accessor())
        } else {
            Ok(self.accessor())
        }
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseUnary<FD, FS, D, T>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    D::Write: DirCreateFile<FD>,
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
