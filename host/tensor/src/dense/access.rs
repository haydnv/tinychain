use std::convert::TryFrom;

use afarray::{Array, ArrayExt, Coords, Offsets};
use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;

use tc_btree::*;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::{Number, NumberClass, NumberInstance, NumberType};
use tcgeneric::{TCBoxTryFuture, TCStream, TCTryStream};

use crate::sparse::{SparseAccess, SparseAccessor};
use crate::stream::{Read, ReadValueAt};
use crate::transform;
use crate::{Bounds, Coord, Phantom, Shape, TensorAccess, TensorType, ERR_NONBIJECTIVE_WRITE};

use super::file::{BlockListFile, BlockListFileSlice};
use super::stream::SparseValueStream;
use super::{DenseTensor, PER_BLOCK};

const ERR_NON_SEQUENTIAL_WRITE: &str =
    "non-sequential writes to a dense Tensor are not efficient--\
consider transposing or broadcasting the value instead";

/// Common [`DenseTensor`] access methods
#[async_trait]
pub trait DenseAccess<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>>:
    Clone + ReadValueAt<D, Txn = T> + TensorAccess + Send + Sync + Sized + 'static
{
    /// The type returned by `slice`
    type Slice: DenseAccess<FD, FS, D, T>;

    /// The type returned by `transpose`
    type Transpose: DenseAccess<FD, FS, D, T>;

    /// Return a [`DenseAccessor`] enum which contains this accessor.
    fn accessor(self) -> DenseAccessor<FD, FS, D, T>;

    /// Return a stream of the [`Array`]s which this [`DenseTensor`] comprises.
    fn block_stream<'a>(self, txn: Self::Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let blocks = self.value_stream(txn).await?;
            let blocks = blocks
                .chunks(PER_BLOCK)
                .map(|values| values.into_iter().collect::<TCResult<Vec<Number>>>())
                .map_ok(Array::from);

            let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    /// Return a stream of the elements of this [`DenseTensor`].
    fn value_stream<'a>(self, txn: Self::Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let values = self.block_stream(txn).await?;

            let values = values
                .map_ok(|array| array.to_vec())
                .map_ok(|values| values.into_iter().map(Ok))
                .map_ok(futures::stream::iter)
                .try_flatten();

            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    /// Return a slice of this [`DenseTensor`].
    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice>;

    /// Return a transpose of this [`DenseTensor`].
    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose>;

    /// Return an Array with the values at the given coordinates.
    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array>;

    /// Overwrite this accessor's contents with those of the given accessor.
    async fn write<V: DenseAccess<FD, FS, D, T>>(&self, txn: Self::Txn, value: V) -> TCResult<()>;

    /// Write a value to the slice of this [`DenseTensor`] with the given [`Bounds`].
    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()>;
}

/// A generic enum which can contain any [`DenseAccess`] impl
#[derive(Clone)]
pub enum DenseAccessor<FD, FS, D, T> {
    Broadcast(Box<BlockListBroadcast<FD, FS, D, T, Self>>),
    Cast(Box<BlockListCast<FD, FS, D, T, Self>>),
    Combine(Box<BlockListCombine<FD, FS, D, T, Self, Self>>),
    Expand(Box<BlockListExpand<FD, FS, D, T, Self>>),
    File(BlockListFile<FD, FS, D, T>),
    Reduce(Box<BlockListReduce<FD, FS, D, T, Self>>),
    Slice(BlockListFileSlice<FD, FS, D, T>),
    Sparse(BlockListSparse<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>),
    Transpose(Box<BlockListTranspose<FD, FS, D, T, Self>>),
    Unary(Box<BlockListUnary<FD, FS, D, T, Self>>),
}

impl<FD, FS, D, T> TensorAccess for DenseAccessor<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    fn dtype(&self) -> NumberType {
        match self {
            Self::Broadcast(broadcast) => broadcast.dtype(),
            Self::Cast(cast) => cast.dtype(),
            Self::Combine(combine) => combine.dtype(),
            Self::Expand(expansion) => expansion.dtype(),
            Self::File(file) => file.dtype(),
            Self::Reduce(reduced) => reduced.dtype(),
            Self::Slice(slice) => slice.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
            Self::Transpose(transpose) => transpose.dtype(),
            Self::Unary(unary) => unary.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Broadcast(broadcast) => broadcast.ndim(),
            Self::Cast(cast) => cast.ndim(),
            Self::Combine(combine) => combine.ndim(),
            Self::Expand(expansion) => expansion.ndim(),
            Self::File(file) => file.ndim(),
            Self::Reduce(reduced) => reduced.ndim(),
            Self::Slice(slice) => slice.ndim(),
            Self::Sparse(sparse) => sparse.ndim(),
            Self::Transpose(transpose) => transpose.ndim(),
            Self::Unary(unary) => unary.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Broadcast(broadcast) => broadcast.shape(),
            Self::Cast(cast) => cast.shape(),
            Self::Combine(combine) => combine.shape(),
            Self::Expand(expansion) => expansion.shape(),
            Self::File(file) => file.shape(),
            Self::Reduce(reduced) => reduced.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::Sparse(sparse) => sparse.shape(),
            Self::Transpose(transpose) => transpose.shape(),
            Self::Unary(unary) => unary.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Broadcast(broadcast) => broadcast.size(),
            Self::Cast(cast) => cast.size(),
            Self::Combine(combine) => combine.size(),
            Self::Expand(expansion) => expansion.size(),
            Self::File(file) => file.size(),
            Self::Reduce(reduced) => reduced.size(),
            Self::Slice(slice) => slice.size(),
            Self::Sparse(sparse) => sparse.size(),
            Self::Transpose(transpose) => transpose.size(),
            Self::Unary(unary) => unary.size(),
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> DenseAccess<FD, FS, D, T> for DenseAccessor<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Slice = Self;
    type Transpose = Self;

    fn accessor(self) -> Self {
        self
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        match self {
            Self::File(file) => file.block_stream(txn),
            Self::Slice(slice) => slice.block_stream(txn),
            Self::Broadcast(broadcast) => broadcast.block_stream(txn),
            Self::Cast(cast) => cast.block_stream(txn),
            Self::Combine(combine) => combine.block_stream(txn),
            Self::Expand(expansion) => expansion.block_stream(txn),
            Self::Reduce(reduced) => reduced.block_stream(txn),
            Self::Sparse(sparse) => sparse.block_stream(txn),
            Self::Transpose(transpose) => transpose.block_stream(txn),
            Self::Unary(unary) => unary.block_stream(txn),
        }
    }

    fn value_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        match self {
            Self::File(file) => file.value_stream(txn),
            Self::Slice(slice) => slice.value_stream(txn),
            Self::Broadcast(broadcast) => broadcast.value_stream(txn),
            Self::Cast(cast) => cast.value_stream(txn),
            Self::Combine(combine) => combine.value_stream(txn),
            Self::Expand(expansion) => expansion.value_stream(txn),
            Self::Reduce(reduced) => reduced.value_stream(txn),
            Self::Sparse(sparse) => sparse.value_stream(txn),
            Self::Transpose(transpose) => transpose.value_stream(txn),
            Self::Unary(unary) => unary.value_stream(txn),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::File(file) => file.slice(bounds).map(Self::Slice),
            Self::Slice(slice) => slice.slice(bounds).map(Self::Slice),
            Self::Broadcast(broadcast) => broadcast.slice(bounds).map(|slice| slice.accessor()),
            Self::Cast(cast) => cast.slice(bounds).map(|slice| slice.accessor()),
            Self::Combine(combine) => combine.slice(bounds).map(|slice| slice.accessor()),
            Self::Expand(expansion) => expansion.slice(bounds).map(|slice| slice.accessor()),
            Self::Reduce(reduced) => reduced.slice(bounds).map(|slice| slice.accessor()),
            Self::Sparse(sparse) => sparse.slice(bounds).map(|slice| slice.accessor()),
            Self::Transpose(transpose) => transpose.slice(bounds).map(|slice| slice.accessor()),
            Self::Unary(unary) => unary.slice(bounds).map(|slice| slice.accessor()),
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        match self {
            Self::File(file) => file
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Slice(slice) => slice
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Broadcast(broadcast) => broadcast
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Cast(cast) => cast
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Combine(combine) => combine
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Expand(expansion) => expansion
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Reduce(reduced) => reduced
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Sparse(sparse) => sparse
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Transpose(transpose) => transpose
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Unary(unary) => unary
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
        }
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        match self {
            Self::File(file) => file.read_values(txn, coords).await,
            Self::Slice(slice) => slice.read_values(txn, coords).await,
            Self::Broadcast(broadcast) => broadcast.read_values(txn, coords).await,
            Self::Cast(cast) => cast.read_values(txn, coords).await,
            Self::Combine(combine) => combine.read_values(txn, coords).await,
            Self::Expand(expansion) => expansion.read_values(txn, coords).await,
            Self::Reduce(reduced) => reduced.read_values(txn, coords).await,
            Self::Sparse(sparse) => sparse.read_values(txn, coords).await,
            Self::Transpose(transpose) => transpose.read_values(txn, coords).await,
            Self::Unary(unary) => unary.read_values(txn, coords).await,
        }
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(&self, txn: Self::Txn, value: V) -> TCResult<()> {
        match self {
            Self::File(file) => file.write(txn, value).await,
            Self::Slice(slice) => slice.write(txn, value).await,
            Self::Broadcast(broadcast) => broadcast.write(txn, value).await,
            Self::Cast(cast) => cast.write(txn, value).await,
            Self::Combine(combine) => combine.write(txn, value).await,
            Self::Expand(expansion) => expansion.write(txn, value).await,
            Self::Reduce(reduced) => reduced.write(txn, value).await,
            Self::Sparse(sparse) => sparse.write(txn, value).await,
            Self::Transpose(transpose) => transpose.write(txn, value).await,
            Self::Unary(unary) => unary.write(txn, value).await,
        }
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        match self {
            Self::File(file) => file.write_value(txn_id, bounds, number).await,
            Self::Slice(slice) => slice.write_value(txn_id, bounds, number).await,
            Self::Broadcast(broadcast) => broadcast.write_value(txn_id, bounds, number).await,
            Self::Cast(cast) => cast.write_value(txn_id, bounds, number).await,
            Self::Combine(combine) => combine.write_value(txn_id, bounds, number).await,
            Self::Expand(expansion) => expansion.write_value(txn_id, bounds, number).await,
            Self::Reduce(reduced) => reduced.write_value(txn_id, bounds, number).await,
            Self::Sparse(sparse) => sparse.write_value(txn_id, bounds, number).await,
            Self::Transpose(transpose) => transpose.write_value(txn_id, bounds, number).await,
            Self::Unary(unary) => unary.write_value(txn_id, bounds, number).await,
        }
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for DenseAccessor<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        match self {
            Self::File(file) => file.read_value_at(txn, coord),
            Self::Slice(slice) => slice.read_value_at(txn, coord),
            Self::Broadcast(broadcast) => broadcast.read_value_at(txn, coord),
            Self::Cast(cast) => cast.read_value_at(txn, coord),
            Self::Combine(combine) => combine.read_value_at(txn, coord),
            Self::Expand(expansion) => expansion.read_value_at(txn, coord),
            Self::Reduce(reduced) => reduced.read_value_at(txn, coord),
            Self::Sparse(sparse) => sparse.read_value_at(txn, coord),
            Self::Transpose(transpose) => transpose.read_value_at(txn, coord),
            Self::Unary(unary) => unary.read_value_at(txn, coord),
        }
    }
}

impl<FD, FS, D, T> From<BlockListFile<FD, FS, D, T>> for DenseAccessor<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
{
    fn from(file: BlockListFile<FD, FS, D, T>) -> Self {
        Self::File(file)
    }
}

#[derive(Clone)]
pub struct BlockListCombine<FD, FS, D, T, L, R> {
    left: L,
    right: R,
    combinator: fn(&Array, &Array) -> Array,
    value_combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, L, R> BlockListCombine<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: DenseAccess<FD, FS, D, T>,
    R: DenseAccess<FD, FS, D, T>,
{
    pub fn new(
        left: L,
        right: R,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<Self> {
        if left.shape() != right.shape() {
            return Err(TCError::bad_request(
                format!("cannot combine shape {} with shape", left.shape()),
                right.shape(),
            ));
        }

        Ok(BlockListCombine {
            left,
            right,
            combinator,
            value_combinator,
            dtype,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, L, R> TensorAccess for BlockListCombine<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: DenseAccess<FD, FS, D, T>,
    R: DenseAccess<FD, FS, D, T>,
{
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
impl<FD, FS, D, T, L, R> DenseAccess<FD, FS, D, T> for BlockListCombine<FD, FS, D, T, L, R>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    L: DenseAccess<FD, FS, D, T>,
    R: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = BlockListCombine<FD, FS, D, T, L::Slice, R::Slice>;
    type Transpose = BlockListCombine<FD, FS, D, T, L::Transpose, R::Transpose>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        let left = self.left.accessor();
        let right = self.right.accessor();
        let combine = BlockListCombine {
            left,
            right,
            combinator: self.combinator,
            value_combinator: self.value_combinator,
            dtype: self.dtype,
            phantom: self.phantom,
        };

        DenseAccessor::Combine(Box::new(combine))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        debug!("BlockListCombine::block_stream");

        Box::pin(async move {
            let left = self.left.block_stream(txn.clone());
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

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let left = self.left.slice(bounds.clone())?;
        let right = self.right.slice(bounds)?;

        BlockListCombine::new(
            left,
            right,
            self.combinator,
            self.value_combinator,
            self.dtype,
        )
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        debug!(
            "BlockListCombine::transpose {} {} {:?}",
            self.left.shape(),
            self.right.shape(),
            permutation
        );

        let left = self.left.transpose(permutation.clone())?;
        let right = self.right.transpose(permutation)?;

        BlockListCombine::new(
            left,
            right,
            self.combinator,
            self.value_combinator,
            self.dtype,
        )
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let (left, right) = try_join!(
            self.left.read_values(txn.clone(), coords.clone()),
            self.right.read_values(txn, coords)
        )?;

        Ok((self.combinator)(&left, &right))
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(
        &self,
        _txn: Self::Txn,
        _value: V,
    ) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<FD, FS, D, T, L, R> ReadValueAt<D> for BlockListCombine<FD, FS, D, T, L, R>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    L: DenseAccess<FD, FS, D, T>,
    R: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let left = self.left.read_value_at(txn.clone(), coord.to_vec());
            let right = self.right.read_value_at(txn, coord);
            let ((coord, left), (_, right)) = try_join!(left, right)?;
            let value = (self.value_combinator)(left, right);
            Ok((coord, value))
        })
    }
}

#[derive(Clone)]
pub struct BlockListBroadcast<FD, FS, D, T, B> {
    source: B,
    rebase: transform::Broadcast,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> BlockListBroadcast<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    pub fn new(source: B, shape: Shape) -> TCResult<Self> {
        let rebase = transform::Broadcast::new(source.shape().clone(), shape)?;
        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, B> TensorAccess for BlockListBroadcast<FD, FS, D, T, B>
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
impl<FD, FS, D, T, B> DenseAccess<FD, FS, D, T> for BlockListBroadcast<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = BlockListBroadcast<FD, FS, D, T, B::Slice>;
    type Transpose = BlockListTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        let source = self.source.accessor();
        let broadcast = BlockListBroadcast {
            source,
            rebase: self.rebase,
            phantom: Phantom::default(),
        };

        DenseAccessor::Broadcast(Box::new(broadcast))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        let shape = self.shape().clone();
        let size = self.size();
        let rebase = self.rebase;
        let source = self.source;

        let blocks = stream::iter((0..size).step_by(PER_BLOCK))
            .map(move |start| {
                let end = match start + PER_BLOCK as u64 {
                    end if end > size => size,
                    end => end,
                };

                ArrayExt::range(start, end)
            })
            .map(move |offsets| Coords::from_offsets(offsets, &shape))
            .map(move |coords| rebase.invert_coords(&coords))
            .map(move |coords| source.clone().read_values(txn.clone(), coords))
            .buffered(num_cpus::get());

        let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
        Box::pin(future::ready(Ok(blocks)))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.shape().validate_bounds(&bounds)?;

        let shape = bounds.to_shape(self.shape())?;
        let bounds = self.rebase.invert_bounds(bounds);
        let source = self.source.slice(bounds)?;
        BlockListBroadcast::new(source, shape)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        debug!(
            "BlockListTranspose {:?} (shape is {})",
            permutation,
            self.shape()
        );
        BlockListTranspose::new(self, permutation)
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let coords = self.rebase.invert_coords(&coords);
        self.source.read_values(txn, coords).await
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(
        &self,
        _txn: Self::Txn,
        _value: V,
    ) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for BlockListBroadcast<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));

        Box::pin(read)
    }
}

#[derive(Clone)]
pub struct BlockListCast<FD, FS, D, T, B> {
    source: B,
    dtype: NumberType,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> BlockListCast<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    pub fn new(source: B, dtype: NumberType) -> Self {
        Self {
            source,
            dtype,
            phantom: Phantom::default(),
        }
    }
}

impl<FD, FS, D, T, B> TensorAccess for BlockListCast<FD, FS, D, T, B>
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

    fn shape(&self) -> &Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<FD, FS, D, T, B> DenseAccess<FD, FS, D, T> for BlockListCast<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = BlockListCast<FD, FS, D, T, B::Slice>;
    type Transpose = BlockListCast<FD, FS, D, T, B::Transpose>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        let cast = BlockListCast::new(self.source.accessor(), self.dtype);
        DenseAccessor::Cast(Box::new(cast))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let dtype = self.dtype;
            let blocks: TCStream<'a, TCResult<Array>> = self.source.block_stream(txn).await?;
            let cast = blocks.map_ok(move |array| array.cast_into(dtype));
            let cast: TCTryStream<'a, Array> = Box::pin(cast);
            Ok(cast)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let slice = self.source.slice(bounds)?;
        Ok(BlockListCast::new(slice, self.dtype))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let transpose = self.source.transpose(permutation)?;
        Ok(BlockListCast::new(transpose, self.dtype))
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let dtype = self.dtype;

        self.source
            .read_values(txn, coords)
            .map_ok(|values| values.cast_into(dtype))
            .await
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(&self, txn: Self::Txn, value: V) -> TCResult<()> {
        self.source.write(txn, value).await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        self.source.write_value(txn_id, bounds, number).await
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for BlockListCast<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        let dtype = self.dtype;
        let read = self
            .source
            .read_value_at(txn, coord)
            .map_ok(move |(coord, value)| (coord, value.into_type(dtype)));

        Box::pin(read)
    }
}

#[derive(Clone)]
pub struct BlockListExpand<FD, FS, D, T, B> {
    source: B,
    rebase: transform::Expand,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> BlockListExpand<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    pub fn new(source: B, axis: usize) -> TCResult<Self> {
        let rebase = transform::Expand::new(source.shape().clone(), axis)?;
        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, B> TensorAccess for BlockListExpand<FD, FS, D, T, B>
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
impl<FD, FS, D, T, B> DenseAccess<FD, FS, D, T> for BlockListExpand<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = B::Slice;
    type Transpose = B::Transpose;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        let expand = BlockListExpand {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        };

        DenseAccessor::Expand(Box::new(expand))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        self.source.block_stream(txn)
    }

    fn value_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        self.source.value_stream(txn)
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.slice(bounds) // TODO: expand the result
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let permutation = permutation.map(|axes| self.rebase.invert_axes(axes));
        self.source.transpose(permutation) // TODO: expand the result
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let coords = self.rebase.invert_coords(&coords);
        self.source.read_values(txn, coords).await
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(&self, txn: Self::Txn, value: V) -> TCResult<()> {
        self.source.write(txn, value).await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.write_value(txn_id, bounds, number).await
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for BlockListExpand<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, value)| (coord, value));

        Box::pin(read)
    }
}

// TODO: &Txn, not Txn
type Reductor<FD, FS, D, T> =
    fn(&DenseTensor<FD, FS, D, T, DenseAccessor<FD, FS, D, T>>, T) -> TCBoxTryFuture<Number>;

#[derive(Clone)]
pub struct BlockListReduce<FD, FS, D, T, B> {
    source: B,
    rebase: transform::Reduce,
    reductor: Reductor<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> BlockListReduce<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    pub fn new(source: B, axis: usize, reductor: Reductor<FD, FS, D, T>) -> TCResult<Self> {
        let rebase = transform::Reduce::new(source.shape().clone(), axis)?;

        Ok(BlockListReduce {
            source,
            rebase,
            reductor,
        })
    }
}

impl<FD, FS, D, T, B> TensorAccess for BlockListReduce<FD, FS, D, T, B>
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
impl<FD, FS, D, T, B> DenseAccess<FD, FS, D, T> for BlockListReduce<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = BlockListReduce<FD, FS, D, T, <B as DenseAccess<FD, FS, D, T>>::Slice>;
    type Transpose = BlockListTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        let reduce = BlockListReduce {
            source: self.source.accessor(),
            rebase: self.rebase,
            reductor: self.reductor,
        };

        DenseAccessor::Reduce(Box::new(reduce))
    }

    fn value_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let values = stream::iter(Bounds::all(self.shape()).affected())
                .map(move |coord| {
                    let txn = txn.clone();
                    let source = self.source.clone();
                    let reductor = self.reductor;
                    let source_bounds = self.rebase.invert_coord(&coord);
                    Box::pin(async move {
                        let slice = source.slice(source_bounds)?;
                        reductor(&slice.accessor().into(), txn.clone()).await
                    })
                })
                .buffered(num_cpus::get());

            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let reduce_axis = self.rebase.reduce_axis(&bounds);
        let source_bounds = self.rebase.invert_bounds(bounds);
        let slice = self.source.slice(source_bounds)?;
        BlockListReduce::new(slice, reduce_axis, self.reductor)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        debug!(
            "BlockListReduce::transpose {} {:?}",
            self.shape(),
            permutation
        );
        BlockListTranspose::new(self, permutation)
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let coords = coords.into_vec();
        let values: Vec<Number> = stream::iter(coords)
            .map(move |coord| self.clone().read_value_at(txn.clone(), coord))
            .buffered(num_cpus::get())
            .map_ok(|(_coord, value)| value)
            .try_collect()
            .await?;

        Ok(Array::from(values))
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(
        &self,
        _txn: Self::Txn,
        _value: V,
    ) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for BlockListReduce<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let reductor = self.reductor;
            let source_bounds = self.rebase.invert_coord(&coord);
            let slice = self.source.slice(source_bounds)?;
            let value = reductor(&slice.accessor().into(), txn.clone()).await?;

            Ok((coord, value))
        })
    }
}

#[derive(Clone)]
pub struct BlockListTranspose<FD, FS, D, T, B> {
    source: B,
    rebase: transform::Transpose,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> BlockListTranspose<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    pub fn new(source: B, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        let rebase = transform::Transpose::new(source.shape().clone(), permutation)?;
        Ok(BlockListTranspose {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, B> TensorAccess for BlockListTranspose<FD, FS, D, T, B>
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
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<FD, FS, D, T, B> DenseAccess<FD, FS, D, T> for BlockListTranspose<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = BlockListTranspose<FD, FS, D, T, B::Slice>;
    type Transpose = Self;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        let accessor = BlockListTranspose {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        };

        DenseAccessor::Transpose(Box::new(accessor))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let size = self.size();
            let shape = self.shape().clone();

            let per_block = PER_BLOCK as u64;
            let blocks = stream::iter((0..size).step_by(PER_BLOCK))
                .map(move |start| {
                    let end = start + per_block;
                    if end > size {
                        (start, size)
                    } else {
                        (start, end)
                    }
                })
                .map(|(start, end)| Offsets::range(start, end))
                .map(move |offsets| Coords::from_offsets(offsets, &shape))
                .map(move |coords| self.clone().read_values(txn.clone(), coords))
                .buffered(num_cpus::get());

            let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn slice(self, _bounds: Bounds) -> TCResult<Self::Slice> {
        Err(TCError::not_implemented("BlockListTranspose::slice"))
    }

    fn transpose(self, _permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        Err(TCError::not_implemented("BlockListTranspose::transpose"))
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let coords = self.rebase.invert_coords(&coords);
        self.source.read_values(txn, coords).await
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(
        &self,
        _txn: Self::Txn,
        _value: V,
    ) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NON_SEQUENTIAL_WRITE))
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        let bounds = self.rebase.invert_bounds(&bounds);
        self.source.write_value(txn_id, bounds, number).await
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for BlockListTranspose<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));

        Box::pin(read)
    }
}

#[derive(Clone)]
pub struct BlockListSparse<FD, FS, D, T, A> {
    source: A,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> TensorAccess for BlockListSparse<FD, FS, D, T, A>
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
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<FD, FS, D, T, A> DenseAccess<FD, FS, D, T> for BlockListSparse<FD, FS, D, T, A>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = BlockListSparse<FD, FS, D, T, A::Slice>;
    type Transpose = BlockListSparse<FD, FS, D, T, A::Transpose>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        let source = self.source.accessor();
        DenseAccessor::Sparse(BlockListSparse {
            source,
            phantom: self.phantom,
        })
    }

    fn value_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        debug!("BlockListSparse::value_stream");

        Box::pin(async move {
            let bounds = Bounds::all(self.shape());
            let zero = self.dtype().zero();
            let filled = self.source.filled(txn).await?;
            let values = SparseValueStream::new(filled, bounds, zero).await?;
            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let slice = self.source.slice(bounds)?;
        Ok(slice.into())
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let transpose = self.source.transpose(permutation)?;
        Ok(transpose.into())
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let coords = coords.into_vec();
        let source = self.source.clone();
        let values: Vec<Number> = stream::iter(coords)
            .map(move |coord| source.clone().read_value_at(txn.clone(), coord))
            .buffered(num_cpus::get())
            .map_ok(|(_coord, value)| value)
            .try_collect()
            .await?;

        Ok(Array::from(values))
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(&self, txn: Self::Txn, value: V) -> TCResult<()> {
        if value.shape() != self.shape() {
            return Err(TCError::unsupported(format!(
                "cannot write a value of shape {} to a tensor of shape {}",
                value.shape(),
                self.shape()
            )));
        }

        let txn_id = *txn.id();
        let bounds = Bounds::all(self.shape());
        let values = value.value_stream(txn).await?;
        let source = &self.source;
        stream::iter(bounds.affected())
            .zip(values)
            .map(|(coord, r)| r.map(|n| (coord, n)))
            .map_ok(|(coord, n)| source.write_value(txn_id, coord, n))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        stream::iter(bounds.affected())
            .map(|coord| self.source.write_value(txn_id, coord, number))
            .buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for BlockListSparse<FD, FS, D, T, A>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        self.source.read_value_at(txn, coord)
    }
}

impl<FD, FS, D, T, A> From<A> for BlockListSparse<FD, FS, D, T, A> {
    fn from(source: A) -> Self {
        BlockListSparse {
            source,
            phantom: Phantom::default(),
        }
    }
}

#[derive(Clone)]
pub struct BlockListUnary<FD, FS, D, T, B> {
    source: B,
    transform: fn(&Array) -> Array,
    value_transform: fn(Number) -> Number,
    dtype: NumberType,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> BlockListUnary<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    pub fn new(
        source: B,
        transform: fn(&Array) -> Array,
        value_transform: fn(Number) -> Number,
        dtype: NumberType,
    ) -> Self {
        Self {
            source,
            transform,
            value_transform,
            dtype,
            phantom: Phantom::default(),
        }
    }
}

impl<FD, FS, D, T, B> TensorAccess for BlockListUnary<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
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
impl<FD, FS, D, T, B> DenseAccess<FD, FS, D, T> for BlockListUnary<FD, FS, D, T, B>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Slice = BlockListUnary<FD, FS, D, T, B::Slice>;
    type Transpose = BlockListUnary<FD, FS, D, T, B::Transpose>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        let unary = BlockListUnary::new(
            self.source.accessor(),
            self.transform,
            self.value_transform,
            self.dtype,
        );

        DenseAccessor::Unary(Box::new(unary))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let transform = self.transform;
            let blocks = self.source.block_stream(txn).await?;
            let blocks: TCTryStream<'a, Array> =
                Box::pin(blocks.map_ok(move |array| transform(&array)));

            Ok(blocks)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let source = self.source.slice(bounds)?;
        Ok(BlockListUnary {
            source,
            transform: self.transform,
            value_transform: self.value_transform,
            dtype: self.dtype,
            phantom: Phantom::default(),
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let source = self.source.transpose(permutation)?;
        Ok(BlockListUnary {
            source,
            transform: self.transform,
            value_transform: self.value_transform,
            dtype: self.dtype,
            phantom: Phantom::default(),
        })
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let transform = self.transform;

        self.source
            .read_values(txn, coords)
            .map_ok(move |values| (transform)(&values))
            .await
    }

    async fn write<V: DenseAccess<FD, FS, D, T>>(
        &self,
        _txn: Self::Txn,
        _value: V,
    ) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for BlockListUnary<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let transform = self.value_transform;
            self.source
                .read_value_at(txn, coord)
                .map_ok(|(coord, value)| (coord, transform(value)))
                .await
        })
    }
}
