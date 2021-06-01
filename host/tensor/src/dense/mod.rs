use std::convert::TryFrom;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use afarray::{Array, ArrayInstance};
use arrayfire as af;
use async_trait::async_trait;
use destream::{de, en, EncodeSeq};
use futures::future::{self, TryFutureExt};
use futures::stream::{Stream, StreamExt, TryStreamExt};
use log::debug;
use number_general::{Number, NumberClass, NumberInstance, NumberType};

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, File, Hash, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::ValueType;
use tcgeneric::{NativeClass, TCBoxTryFuture, TCPathBuf, TCTryStream};

use super::{
    Bounds, Coord, Read, ReadValueAt, Schema, Shape, Tensor, TensorAccess, TensorBoolean,
    TensorDualIO, TensorIO, TensorInstance, TensorMath, TensorReduce, TensorTransform, TensorType,
    TensorUnary,
};

use access::*;

use crate::TensorCompare;
pub use file::BlockListFile;

mod access;
mod file;

/// The number of elements per dense tensor block, equal to (1 mebibyte / 64 bits).
pub const PER_BLOCK: usize = 131_072;

/// Common [`DenseTensor`] access methods
#[async_trait]
pub trait DenseAccess<F: File<Array>, D: Dir, T: Transaction<D>>:
    Clone + ReadValueAt<D, Txn = T> + TensorAccess + Send + Sync + Sized + 'static
{
    /// The type returned by `slice`
    type Slice: DenseAccess<F, D, T>;

    /// The type returned by `transpose`
    type Transpose: DenseAccess<F, D, T>;

    /// Return a [`DenseAccessor`] enum which contains this accessor.
    fn accessor(self) -> DenseAccessor<F, D, T>;

    /// Return a stream of the [`Array`]s which this [`DenseTensor`] comprises.
    fn block_stream<'a>(self, txn: Self::Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let blocks = self
                .value_stream(txn)
                .await?
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
                .map_ok(|values| {
                    values
                        .into_iter()
                        .map(Ok)
                        .collect::<Vec<TCResult<Number>>>()
                })
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

    /// Write a value to the slice of this [`DenseTensor`] with the given [`Bounds`].
    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()>;

    /// Overwrite a single element of this [`DenseTensor`].
    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()>;
}

/// A generic enum which can contain any [`DenseAccess`] impl
#[derive(Clone)]
pub enum DenseAccessor<F: File<Array>, D: Dir, T: Transaction<D>> {
    File(BlockListFile<F, D, T>),
    Slice(file::BlockListFileSlice<F, D, T>),
    Broadcast(Box<BlockListBroadcast<F, D, T, Self>>),
    Cast(Box<BlockListCast<F, D, T, Self>>),
    Combine(Box<BlockListCombine<F, D, T, Self, Self>>),
    Expand(Box<BlockListExpand<F, D, T, Self>>),
    Reduce(Box<BlockListReduce<F, D, T, Self>>),
    Transpose(Box<BlockListTranspose<F, D, T, Self>>),
    Unary(Box<BlockListUnary<F, D, T, Self>>),
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorAccess for DenseAccessor<F, D, T> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::File(file) => file.dtype(),
            Self::Slice(slice) => slice.dtype(),
            Self::Broadcast(broadcast) => broadcast.dtype(),
            Self::Cast(cast) => cast.dtype(),
            Self::Combine(combine) => combine.dtype(),
            Self::Expand(expansion) => expansion.dtype(),
            Self::Reduce(reduced) => reduced.dtype(),
            Self::Transpose(transpose) => transpose.dtype(),
            Self::Unary(unary) => unary.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::File(file) => file.ndim(),
            Self::Slice(slice) => slice.ndim(),
            Self::Broadcast(broadcast) => broadcast.ndim(),
            Self::Cast(cast) => cast.ndim(),
            Self::Combine(combine) => combine.ndim(),
            Self::Expand(expansion) => expansion.ndim(),
            Self::Reduce(reduced) => reduced.ndim(),
            Self::Transpose(transpose) => transpose.ndim(),
            Self::Unary(unary) => unary.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::File(file) => file.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::Broadcast(broadcast) => broadcast.shape(),
            Self::Cast(cast) => cast.shape(),
            Self::Combine(combine) => combine.shape(),
            Self::Expand(expansion) => expansion.shape(),
            Self::Reduce(reduced) => reduced.shape(),
            Self::Transpose(transpose) => transpose.shape(),
            Self::Unary(unary) => unary.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::File(file) => file.size(),
            Self::Slice(slice) => slice.size(),
            Self::Broadcast(broadcast) => broadcast.size(),
            Self::Cast(cast) => cast.size(),
            Self::Combine(combine) => combine.size(),
            Self::Expand(expansion) => expansion.size(),
            Self::Reduce(reduced) => reduced.size(),
            Self::Transpose(transpose) => transpose.size(),
            Self::Unary(unary) => unary.size(),
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> DenseAccess<F, D, T> for DenseAccessor<F, D, T> {
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
            Self::Transpose(transpose) => transpose
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
            Self::Unary(unary) => unary
                .transpose(permutation)
                .map(|transpose| transpose.accessor()),
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
            Self::Transpose(transpose) => transpose.write_value(txn_id, bounds, number).await,
            Self::Unary(unary) => unary.write_value(txn_id, bounds, number).await,
        }
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        match self {
            Self::File(file) => file.write_value_at(txn_id, coord, value),
            Self::Slice(slice) => slice.write_value_at(txn_id, coord, value),
            Self::Broadcast(broadcast) => broadcast.write_value_at(txn_id, coord, value),
            Self::Cast(cast) => cast.write_value_at(txn_id, coord, value),
            Self::Combine(combine) => combine.write_value_at(txn_id, coord, value),
            Self::Expand(expansion) => expansion.write_value_at(txn_id, coord, value),
            Self::Reduce(reduced) => reduced.write_value_at(txn_id, coord, value),
            Self::Transpose(transpose) => transpose.write_value_at(txn_id, coord, value),
            Self::Unary(unary) => unary.write_value_at(txn_id, coord, value),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> ReadValueAt<D> for DenseAccessor<F, D, T> {
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
            Self::Transpose(transpose) => transpose.read_value_at(txn, coord),
            Self::Unary(unary) => unary.read_value_at(txn, coord),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> From<BlockListFile<F, D, T>>
    for DenseAccessor<F, D, T>
{
    fn from(file: BlockListFile<F, D, T>) -> Self {
        Self::File(file)
    }
}

/// A `Tensor` stored as a [`File`] of dense [`Array`] blocks
#[derive(Clone)]
pub struct DenseTensor<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> {
    blocks: B,
    file: PhantomData<F>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> DenseTensor<F, D, T, B> {
    /// Consume this `DenseTensor` handle and return its underlying [`DenseAccessor`]
    pub fn into_inner(self) -> B {
        self.blocks
    }

    fn combine<OT: DenseAccess<F, D, T>>(
        self,
        other: DenseTensor<F, D, T, OT>,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<DenseTensor<F, D, T, BlockListCombine<F, D, T, B, OT>>> {
        if self.shape() != other.shape() {
            return Err(TCError::unsupported(format!(
                "Cannot combine tensors with different shapes: {}, {}",
                self.shape(),
                other.shape()
            )));
        }

        let blocks = BlockListCombine::new(
            self.blocks,
            other.blocks,
            combinator,
            value_combinator,
            dtype,
        )?;

        Ok(DenseTensor::from(blocks))
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> DenseTensor<F, D, T, BlockListFile<F, D, T>> {
    /// Create a new `DenseTensor` with the given [`Schema`].
    pub async fn create(file: F, schema: Schema, txn_id: TxnId) -> TCResult<Self> {
        let (shape, dtype) = schema;
        BlockListFile::constant(file, txn_id, shape, dtype.zero())
            .map_ok(Self::from)
            .await
    }

    /// Create a new `DenseTensor` filled with the given `value`.
    pub async fn constant<S>(file: F, txn_id: TxnId, shape: S, value: Number) -> TCResult<Self>
    where
        Shape: From<S>,
    {
        BlockListFile::constant(file, txn_id, shape.into(), value)
            .map_ok(Self::from)
            .await
    }

    /// Create a new `DenseTensor` filled with a range evenly distributed between `start` and `stop`.
    pub async fn range<S>(
        file: F,
        txn_id: TxnId,
        shape: S,
        start: Number,
        stop: Number,
    ) -> TCResult<Self>
    where
        Shape: From<S>,
    {
        BlockListFile::range(file, txn_id, shape.into(), start, stop)
            .map_ok(Self::from)
            .await
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorAccess
    for DenseTensor<F, D, T, B>
{
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

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorInstance<D>
    for DenseTensor<F, D, T, B>
{
    type Dense = Self;

    fn into_dense(self) -> Self::Dense {
        self
    }
}

impl<F, D, T, B, O> TensorBoolean<D, DenseTensor<F, D, T, O>> for DenseTensor<F, D, T, B>
where
    F: File<Array>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<F, D, T>,
    O: DenseAccess<F, D, T>,
{
    type Combine = DenseTensor<F, D, T, BlockListCombine<F, D, T, B, O>>;

    fn and(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::and, Number::and, NumberType::Bool)
    }

    fn or(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::or, Number::or, NumberType::Bool)
    }

    fn xor(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::xor, Number::xor, NumberType::Bool)
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    TensorBoolean<D, Tensor<F, D, T>> for DenseTensor<F, D, T, B>
{
    type Combine = Tensor<F, D, T>;

    fn and(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.and(dense).map(Tensor::from),
        }
    }

    fn or(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.or(dense).map(Tensor::from),
        }
    }

    fn xor(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.xor(dense).map(Tensor::from),
        }
    }
}

#[async_trait]
impl<F, D, T, B, O> TensorCompare<D, DenseTensor<F, D, T, O>> for DenseTensor<F, D, T, B>
where
    F: File<Array>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<F, D, T>,
    O: DenseAccess<F, D, T>,
{
    type Compare = DenseTensor<F, D, T, BlockListCombine<F, D, T, B, O>>;
    type Dense = DenseTensor<F, D, T, BlockListCombine<F, D, T, B, O>>;

    async fn eq(self, other: DenseTensor<F, D, T, O>, _txn: Self::Txn) -> TCResult<Self::Dense> {
        fn eq(l: Number, r: Number) -> Number {
            Number::from(l == r)
        }

        self.combine(other, Array::eq, eq, NumberType::Bool)
    }

    fn gt(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Compare> {
        fn gt(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::gt, gt, NumberType::Bool)
    }

    async fn gte(self, other: DenseTensor<F, D, T, O>, _txn: Self::Txn) -> TCResult<Self::Dense> {
        fn gte(l: Number, r: Number) -> Number {
            Number::from(l >= r)
        }

        self.combine(other, Array::gte, gte, NumberType::Bool)
    }

    fn lt(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Compare> {
        fn lt(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::lt, lt, NumberType::Bool)
    }

    async fn lte(self, other: DenseTensor<F, D, T, O>, _txn: Self::Txn) -> TCResult<Self::Dense> {
        fn lte(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::lte, lte, NumberType::Bool)
    }

    fn ne(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Compare> {
        fn ne(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::ne, ne, NumberType::Bool)
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    TensorCompare<D, Tensor<F, D, T>> for DenseTensor<F, D, T, B>
{
    type Compare = Tensor<F, D, T>;
    type Dense = Tensor<F, D, T>;

    async fn eq(self, other: Tensor<F, D, T>, txn: Self::Txn) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(other) => self.eq(other, txn).map_ok(Tensor::from).await,
        }
    }

    fn gt(self, other: Tensor<F, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(other) => self.gt(other).map(Tensor::from),
        }
    }

    async fn gte(self, other: Tensor<F, D, T>, txn: Self::Txn) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(other) => self.gte(other, txn).map_ok(Tensor::from).await,
        }
    }

    fn lt(self, other: Tensor<F, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(other) => self.lt(other).map(Tensor::from),
        }
    }

    async fn lte(self, other: Tensor<F, D, T>, txn: Self::Txn) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(other) => self.lte(other, txn).map_ok(Tensor::from).await,
        }
    }

    fn ne(self, other: Tensor<F, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(other) => self.ne(other).map(Tensor::from),
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorIO<D>
    for DenseTensor<F, D, T, B>
{
    type Txn = T;

    async fn read_value(&self, txn: &Self::Txn, coord: Coord) -> TCResult<Number> {
        self.blocks
            .clone()
            .read_value_at(txn.clone(), coord.to_vec())
            .map_ok(|(_, val)| val)
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        self.blocks.clone().write_value(txn_id, bounds, value).await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.blocks.write_value_at(txn_id, coord, value).await
    }
}

#[async_trait]
impl<F, D, T, B, O> TensorDualIO<D, DenseTensor<F, D, T, O>> for DenseTensor<F, D, T, B>
where
    F: File<Array>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<F, D, T>,
    O: DenseAccess<F, D, T>,
{
    async fn mask(self, _txn: T, _other: DenseTensor<F, D, T, O>) -> TCResult<()> {
        Err(TCError::not_implemented("DenseTensor::mask"))
    }

    async fn write(self, txn: T, bounds: Bounds, other: DenseTensor<F, D, T, O>) -> TCResult<()> {
        debug!("write dense tensor to dense {}", bounds);

        let dtype = self.dtype();
        let txn_id = *txn.id();
        let coords = bounds.affected();
        let slice = self.slice(bounds.clone())?;
        let other = other.broadcast(slice.shape().clone())?.cast_into(dtype)?;

        let other_values = other.blocks.value_stream(txn).await?;
        let values = futures::stream::iter(coords).zip(other_values);

        values
            .map(|(coord, r)| r.map(|value| (coord, value)))
            .inspect_ok(|(coord, value)| debug!("write {} at {:?}", value, coord))
            .map_ok(|(coord, value)| slice.write_value_at(txn_id, coord, value))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(())
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    TensorDualIO<D, Tensor<F, D, T>> for DenseTensor<F, D, T, B>
{
    async fn mask(self, txn: T, other: Tensor<F, D, T>) -> TCResult<()> {
        match other {
            Tensor::Dense(dense) => self.mask(txn, dense).await,
        }
    }

    async fn write(self, txn: T, bounds: Bounds, other: Tensor<F, D, T>) -> TCResult<()> {
        debug!("DenseTensor::write {} to {}", other, bounds);

        match other {
            Tensor::Dense(dense) => self.write(txn, bounds, dense).await,
        }
    }
}

impl<F, D, T, B, O> TensorMath<D, DenseTensor<F, D, T, O>> for DenseTensor<F, D, T, B>
where
    F: File<Array>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<F, D, T>,
    O: DenseAccess<F, D, T>,
{
    type Combine = DenseTensor<F, D, T, BlockListCombine<F, D, T, B, O>>;

    fn add(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Combine> {
        fn add_array(l: &Array, r: &Array) -> Array {
            l + r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, add_array, Add::add, dtype)
    }

    fn div(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Combine> {
        fn div_array(l: &Array, r: &Array) -> Array {
            l / r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, div_array, Div::div, dtype)
    }

    fn mul(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Combine> {
        fn mul_array(l: &Array, r: &Array) -> Array {
            l * r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, mul_array, Mul::mul, dtype)
    }

    fn sub(self, other: DenseTensor<F, D, T, O>) -> TCResult<Self::Combine> {
        fn sub_array(l: &Array, r: &Array) -> Array {
            l - r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, sub_array, Sub::sub, dtype)
    }
}

impl<F, D, T, B> TensorMath<D, Tensor<F, D, T>> for DenseTensor<F, D, T, B>
where
    F: File<Array>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<F, D, T>,
{
    type Combine = Tensor<F, D, T>;

    fn add(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(other) => self.add(other).map(Tensor::from),
        }
    }

    fn div(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(other) => self.div(other).map(Tensor::from),
        }
    }

    fn mul(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(other) => self.mul(other).map(Tensor::from),
        }
    }

    fn sub(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(other) => self.sub(other).map(Tensor::from),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> ReadValueAt<D>
    for DenseTensor<F, D, T, B>
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        self.blocks.read_value_at(txn, coord)
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorReduce<D>
    for DenseTensor<F, D, T, B>
{
    type Reduce = DenseTensor<F, D, T, BlockListReduce<F, D, T, B>>;

    fn product(self, axis: usize) -> TCResult<Self::Reduce> {
        BlockListReduce::new(self.blocks, axis, DenseTensor::product_all).map(DenseTensor::from)
    }

    fn product_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let zero = self.dtype().zero();
            let mut product = self.dtype().one();

            let blocks = self.blocks.clone().block_stream(txn).await?;
            let mut block_products = blocks.map_ok(|array| array.product());

            while let Some(block_product) = block_products.try_next().await? {
                if block_product == zero {
                    return Ok(zero);
                }

                product = product * block_product;
            }

            Ok(product)
        })
    }

    fn sum(self, axis: usize) -> TCResult<Self::Reduce> {
        BlockListReduce::new(self.blocks, axis, DenseTensor::sum_all).map(DenseTensor::from)
    }

    fn sum_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let zero = self.dtype().zero();
            let blocks = self.blocks.clone().block_stream(txn).await?;

            blocks
                .map_ok(|array| array.sum())
                .try_fold(zero, |sum, block_sum| future::ready(Ok(sum + block_sum)))
                .await
        })
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorTransform<D>
    for DenseTensor<F, D, T, B>
{
    type Broadcast = DenseTensor<F, D, T, BlockListBroadcast<F, D, T, B>>;
    type Cast = DenseTensor<F, D, T, BlockListCast<F, D, T, B>>;
    type Expand = DenseTensor<F, D, T, BlockListExpand<F, D, T, B>>;
    type Slice = DenseTensor<F, D, T, <B as DenseAccess<F, D, T>>::Slice>;
    type Transpose = DenseTensor<F, D, T, <B as DenseAccess<F, D, T>>::Transpose>;

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        let blocks = BlockListCast::new(self.blocks, dtype);
        Ok(DenseTensor::from(blocks))
    }

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        let blocks = BlockListBroadcast::new(self.blocks, shape)?;
        Ok(DenseTensor::from(blocks))
    }

    fn expand_dims(self, axis: usize) -> TCResult<Self::Expand> {
        let blocks = BlockListExpand::new(self.blocks, axis)?;
        Ok(DenseTensor::from(blocks))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let blocks = self.blocks.slice(bounds)?;
        Ok(DenseTensor::from(blocks))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let blocks = self.blocks.transpose(permutation)?;
        Ok(DenseTensor::from(blocks))
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorUnary<D>
    for DenseTensor<F, D, T, B>
{
    type Unary = DenseTensor<F, D, T, BlockListUnary<F, D, T, B>>;

    fn abs(&self) -> TCResult<Self::Unary> {
        let blocks = BlockListUnary::new(
            self.blocks.clone(),
            Array::abs,
            <Number as NumberInstance>::abs,
            NumberType::Bool,
        );

        Ok(DenseTensor::from(blocks))
    }

    async fn all(self, txn: T) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(txn).await?;

        while let Some(array) = blocks.next().await {
            if !array?.all() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self, txn: T) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(txn).await?;
        while let Some(array) = blocks.next().await {
            if array?.any() {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn not(&self) -> TCResult<Self::Unary> {
        let blocks = BlockListUnary::new(
            self.blocks.clone(),
            Array::not,
            Number::not,
            NumberType::Bool,
        );

        Ok(DenseTensor::from(blocks))
    }
}

#[async_trait]
impl<F: File<Array> + Transact, D: Dir, T: Transaction<D>> Transact
    for DenseTensor<F, D, T, BlockListFile<F, D, T>>
{
    async fn commit(&self, txn_id: &TxnId) {
        self.blocks.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.blocks.finalize(txn_id).await
    }
}

#[async_trait]
impl<'en, F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> Hash<'en, D>
    for DenseTensor<F, D, T, B>
{
    type Item = Array;
    type Txn = T;

    async fn hashable(&'en self, txn: &'en T) -> TCResult<TCTryStream<'en, Self::Item>> {
        self.blocks.clone().block_stream(txn.clone()).await
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    CopyFrom<D, DenseTensor<F, D, T, B>> for DenseTensor<F, D, T, BlockListFile<F, D, T>>
{
    async fn copy_from(instance: DenseTensor<F, D, T, B>, file: F, txn: T) -> TCResult<Self> {
        BlockListFile::copy_from(instance.blocks, file, txn)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> Persist<D>
    for DenseTensor<F, D, T, BlockListFile<F, D, T>>
{
    type Schema = Schema;
    type Store = F;
    type Txn = T;

    fn schema(&self) -> &Self::Schema {
        self.blocks.schema()
    }

    async fn load(txn: &T, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        BlockListFile::load(txn, schema, store)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> Restore<D>
    for DenseTensor<F, D, T, BlockListFile<F, D, T>>
{
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        self.blocks.restore(&backup.blocks, txn_id).await
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> From<B>
    for DenseTensor<F, D, T, B>
{
    fn from(blocks: B) -> Self {
        Self {
            blocks,
            file: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> de::FromStream
    for DenseTensor<F, D, T, BlockListFile<F, D, T>>
where
    <D as Dir>::FileClass: From<TensorType> + Send,
    F: TryFrom<<D as Dir>::File, Error = TCError>,
{
    type Context = T;

    async fn from_stream<De: de::Decoder>(txn: T, decoder: &mut De) -> Result<Self, De::Error> {
        let txn_id = *txn.id();
        let file = txn
            .context()
            .create_file_tmp(txn_id, TensorType::Dense)
            .map_err(de::Error::custom)
            .await?;

        decoder
            .decode_seq(DenseTensorVisitor::new(txn_id, file))
            .await
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> fmt::Display
    for DenseTensor<F, D, T, B>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a dense Tensor")
    }
}

struct DenseTensorVisitor<F, D, T> {
    txn_id: TxnId,
    file: F,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<F, D, T> DenseTensorVisitor<F, D, T> {
    fn new(txn_id: TxnId, file: F) -> Self {
        Self {
            txn_id,
            file,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> de::Visitor for DenseTensorVisitor<F, D, T> {
    type Value = DenseTensor<F, D, T, BlockListFile<F, D, T>>;

    fn expecting() -> &'static str {
        "a dense tensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        debug!("visit Tensor data to deserialize");

        let (shape, dtype) = seq
            .next_element::<(Vec<u64>, TCPathBuf)>(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a tensor schema"))?;

        debug!("decoded Tensor schema");

        let dtype = match ValueType::from_path(&dtype) {
            Some(ValueType::Number(nt)) => Ok(nt),
            _ => Err(de::Error::invalid_type(dtype, "a NumberType")),
        }?;

        debug!("decoding Tensor blocks");

        let cxt = (self.txn_id, self.file, (shape.into(), dtype));
        let blocks = seq
            .next_element::<BlockListFile<F, D, T>>(cxt)
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "dense tensor data"))?;

        Ok(DenseTensor::from(blocks))
    }
}

#[async_trait]
impl<'en, F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> IntoView<'en, D>
    for DenseTensor<F, D, T, B>
{
    type Txn = T;
    type View = DenseTensorView<'en>;

    async fn into_view(self, txn: T) -> TCResult<DenseTensorView<'en>> {
        let dtype = self.dtype();
        let shape = self.shape().to_vec();
        let blocks = self.blocks.block_stream(txn).await?;

        Ok(DenseTensorView {
            schema: (shape, ValueType::from(dtype).path()),
            blocks: BlockStreamView { dtype, blocks },
        })
    }
}

/// A view of a [`DenseTensor`] as of a specific [`TxnId`], used in serialization
pub struct DenseTensorView<'en> {
    schema: (Vec<u64>, TCPathBuf),
    blocks: BlockStreamView<'en>,
}

#[async_trait]
impl<'en> en::IntoStream<'en> for DenseTensorView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut seq = encoder.encode_seq(Some(2))?;
        seq.encode_element(self.schema)?;
        seq.encode_element(self.blocks)?;
        seq.end()
    }
}

struct BlockStreamView<'en> {
    dtype: NumberType,
    blocks: TCTryStream<'en, Array>,
}

impl<'en> en::IntoStream<'en> for BlockStreamView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use number_general::{
            ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT,
        };

        fn encodable<'en, E: en::Error + 'en, T: af::HasAfEnum + Clone + Default + 'en>(
            blocks: TCTryStream<'en, Array>,
        ) -> impl Stream<Item = Result<Vec<T>, E>> + 'en {
            blocks
                .map_ok(|arr| arr.type_cast())
                .map_ok(|arr| arr.to_vec())
                .map_err(en::Error::custom)
        }

        match self.dtype {
            NT::Bool => encoder.encode_array_bool(encodable(self.blocks)),
            NT::Complex(ct) => match ct {
                CT::C32 => encoder.encode_array_f32(encodable_c32(self.blocks)),
                _ => encoder.encode_array_f64(encodable_c64(self.blocks)),
            },
            NT::Float(ft) => match ft {
                FT::F32 => encoder.encode_array_f32(encodable(self.blocks)),
                _ => encoder.encode_array_f64(encodable(self.blocks)),
            },
            NT::Int(it) => match it {
                IT::I8 | IT::I16 => encoder.encode_array_i16(encodable(self.blocks)),
                IT::I32 => encoder.encode_array_i32(encodable(self.blocks)),
                _ => encoder.encode_array_i64(encodable(self.blocks)),
            },
            NT::UInt(ut) => match ut {
                UT::U8 => encoder.encode_array_u8(encodable(self.blocks)),
                UT::U16 => encoder.encode_array_u16(encodable(self.blocks)),
                UT::U32 => encoder.encode_array_u32(encodable(self.blocks)),
                _ => encoder.encode_array_u64(encodable(self.blocks)),
            },
            NT::Number => Err(en::Error::custom(format!(
                "invalid Tensor data type: {}",
                NT::Number
            ))),
        }
    }
}

fn encodable_c32<'en, E: en::Error + 'en>(
    blocks: TCTryStream<'en, Array>,
) -> impl Stream<Item = Result<Vec<f32>, E>> + 'en {
    blocks
        .map_ok(|arr| {
            let source = arr.type_cast::<afarray::Complex<f32>>();
            let re = source.re();
            let im = source.im();

            let mut i = 0;
            let mut dest = vec![0.; source.len() * 2];
            for (re, im) in re.to_vec().into_iter().zip(im.to_vec()) {
                dest[i] = re;
                dest[i + 1] = im;
                i += 2;
            }

            dest
        })
        .map_err(en::Error::custom)
}

fn encodable_c64<'en, E: en::Error + 'en>(
    blocks: TCTryStream<'en, Array>,
) -> impl Stream<Item = Result<Vec<f64>, E>> + 'en {
    blocks
        .map_ok(|arr| {
            let source = arr.type_cast::<afarray::Complex<f64>>();
            let re = source.re();
            let im = source.im();

            let mut i = 0;
            let mut dest = vec![0.; source.len() * 2];
            for (re, im) in re.to_vec().into_iter().zip(im.to_vec()) {
                dest[i] = re;
                dest[i + 1] = im;
                i += 2;
            }

            dest
        })
        .map_err(en::Error::custom)
}

fn block_offsets(
    af_indices: &af::Array<u64>,
    af_offsets: &af::Array<u64>,
    start: f64,
    block_id: u64,
) -> (af::Array<u64>, f64) {
    assert_eq!(af_indices.elements(), af_offsets.elements());

    let num_to_update = af::sum_all(&af::eq(
        af_indices,
        &af::constant(block_id, af::Dim4::new(&[1, 1, 1, 1])),
        true,
    ))
    .0;

    if num_to_update == 0 {
        return (af::Array::new_empty(af::Dim4::default()), start);
    }

    debug_assert!((start as usize + num_to_update as usize) <= af_offsets.elements());

    let num_to_update = num_to_update as f64;
    let block_offsets = af::index(
        af_offsets,
        &[af::Seq::new(start, (start + num_to_update) - 1f64, 1f64)],
    );

    (block_offsets, (start + num_to_update))
}

fn coord_block<I: Iterator<Item = Coord>>(
    coords: I,
    coord_bounds: &[u64],
    per_block: usize,
    ndim: usize,
    num_coords: u64,
) -> (Coord, af::Array<u64>, af::Array<u64>) {
    let coords: Vec<u64> = coords.flatten().collect();
    assert!(coords.len() > 0);
    assert!(ndim > 0);

    let af_per_block = af::constant(per_block as u64, af::Dim4::new(&[1, 1, 1, 1]));
    let af_coord_bounds = af::Array::new(coord_bounds, af::Dim4::new(&[ndim as u64, 1, 1, 1]));

    let af_coords = af::Array::new(
        &coords,
        af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]),
    );
    let af_coords = af::mul(&af_coords, &af_coord_bounds, true);
    let af_coords = af::sum(&af_coords, 0);

    let af_offsets = af::modulo(&af_coords, &af_per_block, true);
    let af_indices = af_coords / af_per_block;

    let af_block_ids = af::set_unique(&af_indices, true);
    let mut block_ids = vec![0u64; af_block_ids.elements()];
    af_block_ids.host(&mut block_ids);
    (block_ids, af_indices, af_offsets)
}
