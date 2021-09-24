use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use afarray::{Array, ArrayInstance};
use arrayfire as af;
use async_trait::async_trait;
use destream::{de, en, EncodeSeq};
use futures::future::{self, TryFutureExt};
use futures::stream::{Stream, StreamExt, TryStreamExt};
use log::{debug, warn};
use safecast::AsType;

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, File, Hash, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::{Number, NumberClass, NumberInstance, NumberType};
use tcgeneric::{TCBoxTryFuture, TCBoxTryStream};

use super::sparse::{DenseToSparse, SparseTensor};
use super::stream::{Read, ReadValueAt};
use super::{
    Bounds, Coord, Phantom, Schema, Shape, Tensor, TensorAccess, TensorBoolean, TensorCompare,
    TensorCompareConst, TensorDualIO, TensorIO, TensorInstance, TensorMath, TensorReduce,
    TensorTransform, TensorType, TensorUnary,
};

use access::*;
pub use access::{BlockListSparse, DenseAccess, DenseAccessor, DenseWrite};
pub use file::BlockListFile;

#[allow(unused)]
mod access;
mod file;
mod stream;

/// The number of bytes in one mebibyte.s
const MEBIBYTE: usize = 1_048_576;

/// The number of elements per dense tensor block, equal to (1 mebibyte / 64 bits).
pub const PER_BLOCK: usize = 131_072;

/// A `Tensor` stored as a [`File`] of dense [`Array`] blocks
#[derive(Clone)]
pub struct DenseTensor<FD, FS, D, T, B> {
    blocks: B,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> DenseTensor<FD, FS, D, T, B> {
    /// Consume this `DenseTensor` handle and return its underlying [`DenseAccessor`]
    pub fn into_inner(self) -> B {
        self.blocks
    }
}

impl<FD, FS, D, T, B> DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
{
    fn combine<OT: DenseAccess<FD, FS, D, T>>(
        self,
        other: DenseTensor<FD, FS, D, T, OT>,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, OT>>> {
        if self.shape() != other.shape() {
            return Err(TCError::unsupported(format!(
                "cannot combine tensors with different shapes: {}, {}",
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

impl<FD, FS, D, T> DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
{
    /// Create a new `DenseTensor` with the given [`Schema`].
    pub async fn create(file: FD, schema: Schema, txn_id: TxnId) -> TCResult<Self> {
        let Schema { shape, dtype } = schema;
        BlockListFile::constant(file, txn_id, shape, dtype.zero())
            .map_ok(Self::from)
            .await
    }

    /// Create a new `DenseTensor` filled with the given `value`.
    pub async fn constant<S>(file: FD, txn_id: TxnId, shape: S, value: Number) -> TCResult<Self>
    where
        Shape: From<S>,
    {
        BlockListFile::constant(file, txn_id, shape.into(), value)
            .map_ok(Self::from)
            .await
    }

    /// Create a new `DenseTensor` filled with a range evenly distributed between `start` and `stop`.
    pub async fn range<S>(
        file: FD,
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

impl<FD, FS, D, T, B> TensorAccess for DenseTensor<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
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

impl<FD, FS, D, T, B> TensorInstance for DenseTensor<FD, FS, D, T, B> {
    type Dense = Self;
    type Sparse = SparseTensor<FD, FS, D, T, DenseToSparse<FD, FS, D, T, B>>;

    fn into_dense(self) -> Self::Dense {
        self
    }

    fn into_sparse(self) -> Self::Sparse {
        DenseToSparse::from(self.into_inner()).into()
    }
}

impl<FD, FS, D, T, B, O> TensorBoolean<DenseTensor<FD, FS, D, T, O>>
    for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    O: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Combine = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;
    type LeftCombine = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;

    fn and(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::and, Number::and, NumberType::Bool)
    }

    fn or(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::or, Number::or, NumberType::Bool)
    }

    fn xor(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::xor, Number::xor, NumberType::Bool)
    }
}

impl<FD, FS, D, T, B> TensorBoolean<Tensor<FD, FS, D, T>> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Combine = Tensor<FD, FS, D, T>;
    type LeftCombine = Tensor<FD, FS, D, T>;

    fn and(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.and(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.into_sparse().and(sparse).map(Tensor::from),
        }
    }

    fn or(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.or(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.or(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn xor(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.xor(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.and(sparse.into_dense()).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, B, O> TensorCompare<DenseTensor<FD, FS, D, T, O>>
    for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    O: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Compare = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;
    type Dense = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;

    fn eq(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Dense> {
        fn eq(l: Number, r: Number) -> Number {
            Number::from(l == r)
        }

        self.combine(other, Array::eq, eq, NumberType::Bool)
    }

    fn gt(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Compare> {
        fn gt(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::gt, gt, NumberType::Bool)
    }

    fn gte(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Dense> {
        fn gte(l: Number, r: Number) -> Number {
            Number::from(l >= r)
        }

        self.combine(other, Array::gte, gte, NumberType::Bool)
    }

    fn lt(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Compare> {
        fn lt(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::lt, lt, NumberType::Bool)
    }

    fn lte(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Dense> {
        fn lte(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::lte, lte, NumberType::Bool)
    }

    fn ne(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Compare> {
        fn ne(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::ne, ne, NumberType::Bool)
    }
}

impl<FD, FS, D, T, B> TensorCompare<Tensor<FD, FS, D, T>> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Compare = Tensor<FD, FS, D, T>;
    type Dense = Tensor<FD, FS, D, T>;

    fn eq(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(dense) => self.eq(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.into_sparse().eq(sparse).map(Tensor::from),
        }
    }

    fn gt(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(dense) => self.gt(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.gt(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn gte(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(dense) => self.gte(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.gte(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn lt(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(dense) => self.lt(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.lt(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn lte(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(dense) => self.lte(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.lte(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn ne(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(dense) => self.ne(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.ne(sparse.into_dense()).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, B> TensorCompareConst for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    B: DenseAccess<FD, FS, D, T>,
{
    type Compare = DenseTensor<FD, FS, D, T, BlockListConst<FD, FS, D, T, B>>;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        fn eq_array(l: Array, r: Number) -> Array {
            l.eq_const(r)
        }

        fn eq_number(l: Number, r: Number) -> Number {
            (l == r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, eq_array, eq_number).into())
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        fn gt_array(l: Array, r: Number) -> Array {
            l.gt_const(r)
        }

        fn gt_number(l: Number, r: Number) -> Number {
            (l > r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, gt_array, gt_number).into())
    }

    fn gte_const(self, other: Number) -> TCResult<Self::Compare> {
        fn gte_array(l: Array, r: Number) -> Array {
            l.gte_const(r)
        }

        fn gte_number(l: Number, r: Number) -> Number {
            (l >= r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, gte_array, gte_number).into())
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        fn lt_array(l: Array, r: Number) -> Array {
            l.lt_const(r)
        }

        fn lt_number(l: Number, r: Number) -> Number {
            (l < r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, lt_array, lt_number).into())
    }

    fn lte_const(self, other: Number) -> TCResult<Self::Compare> {
        fn lte_array(l: Array, r: Number) -> Array {
            l.lte_const(r)
        }

        fn lte_number(l: Number, r: Number) -> Number {
            (l <= r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, lte_array, lte_number).into())
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        fn ne_array(l: Array, r: Number) -> Array {
            l.ne_const(r)
        }

        fn ne_number(l: Number, r: Number) -> Number {
            (l != r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, ne_array, ne_number).into())
    }
}

#[async_trait]
impl<FD, FS, D, T, B> TensorIO<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseWrite<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    async fn read_value(self, txn: Self::Txn, coord: Coord) -> TCResult<Number> {
        self.blocks
            .read_value_at(txn, coord)
            .map_ok(|(_, val)| val)
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        self.blocks.write_value(txn_id, bounds, value).await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        debug!("DenseTensor::write_value_at");

        self.blocks
            .write_value(txn_id, Bounds::from(coord), value)
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T, B, O> TensorDualIO<D, DenseTensor<FD, FS, D, T, O>>
    for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseWrite<FD, FS, D, T>,
    O: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    async fn write(
        self,
        txn: T,
        bounds: Bounds,
        other: DenseTensor<FD, FS, D, T, O>,
    ) -> TCResult<()> {
        debug!("write {} to dense {}", other, bounds);
        self.blocks.write(txn, bounds, other.blocks).await
    }
}

#[async_trait]
impl<FD, FS, D, T, B> TensorDualIO<D, Tensor<FD, FS, D, T>> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseWrite<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    async fn write(self, txn: T, bounds: Bounds, other: Tensor<FD, FS, D, T>) -> TCResult<()> {
        debug!("DenseTensor::write {} to {}", other, bounds);

        let shape = bounds.to_shape(self.shape())?;
        let other = if other.shape() == &shape {
            other
        } else {
            other.broadcast(shape)?
        };

        match other {
            Tensor::Dense(dense) => self.write(txn, bounds, dense).await,
            Tensor::Sparse(sparse) => self.write(txn, bounds, sparse.into_dense()).await,
        }
    }
}

impl<FD, FS, D, T, B, O> TensorMath<D, DenseTensor<FD, FS, D, T, O>>
    for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    O: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Combine = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;
    type LeftCombine = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;

    fn add(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn add_array(l: &Array, r: &Array) -> Array {
            debug_assert_eq!(l.len(), r.len());
            l + r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, add_array, Add::add, dtype)
    }

    fn div(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn div_array(l: &Array, r: &Array) -> Array {
            if !r.all() {
                warn!("divide by zero in DenseTensor::div");
            }

            debug_assert_eq!(l.len(), r.len());
            l / r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, div_array, Div::div, dtype)
    }

    fn mul(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn mul_array(l: &Array, r: &Array) -> Array {
            debug_assert_eq!(l.len(), r.len());
            l * r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, mul_array, Mul::mul, dtype)
    }

    fn pow(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn pow_array(l: &Array, r: &Array) -> Array {
            debug_assert_eq!(l.len(), r.len());
            l.pow(r)
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, pow_array, Number::pow, dtype)
    }

    fn sub(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn sub_array(l: &Array, r: &Array) -> Array {
            debug_assert_eq!(l.len(), r.len());
            l - r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, sub_array, Sub::sub, dtype)
    }
}

impl<FD, FS, D, T, B> TensorMath<D, Tensor<FD, FS, D, T>> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Combine = Tensor<FD, FS, D, T>;
    type LeftCombine = Tensor<FD, FS, D, T>;

    fn add(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.add(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.add(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn div(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.div(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.div(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn mul(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.mul(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => sparse.mul(self.into_sparse()).map(Tensor::from),
        }
    }

    fn pow(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.pow(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => sparse.pow(self.into_sparse()).map(Tensor::from),
        }
    }

    fn sub(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.sub(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.sub(sparse.into_dense()).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        self.blocks.read_value_at(txn, coord)
    }
}

impl<FD, FS, D, T, B> TensorReduce<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;
    type Reduce = DenseTensor<FD, FS, D, T, BlockListReduce<FD, FS, D, T, B>>;

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

impl<FD, FS, D, T, B> TensorTransform for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Broadcast = DenseTensor<FD, FS, D, T, BlockListBroadcast<FD, FS, D, T, B>>;
    type Cast = DenseTensor<FD, FS, D, T, BlockListCast<FD, FS, D, T, B>>;
    type Expand = DenseTensor<FD, FS, D, T, BlockListExpand<FD, FS, D, T, B>>;
    type Slice = DenseTensor<FD, FS, D, T, B::Slice>;
    type Transpose = DenseTensor<FD, FS, D, T, B::Transpose>;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        let blocks = BlockListBroadcast::new(self.blocks, shape)?;
        Ok(DenseTensor::from(blocks))
    }

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        let blocks = BlockListCast::new(self.blocks, dtype);
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
impl<FD, FS, D, T, B> TensorUnary<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;
    type Unary = DenseTensor<FD, FS, D, T, BlockListUnary<FD, FS, D, T, B>>;

    fn abs(&self) -> TCResult<Self::Unary> {
        let blocks = BlockListUnary::new(
            self.blocks.clone(),
            Array::abs,
            <Number as NumberInstance>::abs,
            NumberType::Bool,
        );

        Ok(DenseTensor::from(blocks))
    }

    fn exp(&self) -> TCResult<Self::Unary> {
        todo!()
    }

    async fn all(self, txn: T) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(txn).await?;

        while let Some(array) = blocks.try_next().await? {
            if !array.all() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self, txn: T) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(txn).await?;
        while let Some(array) = blocks.try_next().await? {
            if array.any() {
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
impl<FD, FS, D, T> Transact for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array> + Transact,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    async fn commit(&self, txn_id: &TxnId) {
        self.blocks.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.blocks.finalize(txn_id).await
    }
}

#[async_trait]
impl<'en, FD, FS, D, T, B> Hash<'en, D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Item = Array;
    type Txn = T;

    async fn hashable(&'en self, txn: &'en T) -> TCResult<TCBoxTryStream<'en, Self::Item>> {
        self.blocks.clone().block_stream(txn.clone()).await
    }
}

#[async_trait]
impl<FD, FS, D, T, B> CopyFrom<D, DenseTensor<FD, FS, D, T, B>>
    for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    async fn copy_from(
        instance: DenseTensor<FD, FS, D, T, B>,
        file: FD,
        txn: &T,
    ) -> TCResult<Self> {
        BlockListFile::copy_from(instance.blocks, file, txn)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T> Persist<D> for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Schema = Schema;
    type Store = FD;
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
impl<FD, FS, D, T> Restore<D> for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        self.blocks.restore(&backup.blocks, txn_id).await
    }
}

impl<FD, FS, D, T, B> From<B> for DenseTensor<FD, FS, D, T, B> {
    fn from(blocks: B) -> Self {
        Self {
            blocks,
            phantom: Phantom::default(),
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> de::FromStream for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    T: Transaction<D>,
    D::FileClass: From<TensorType> + Send,
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

impl<FD, FS, D, T, B> fmt::Display for DenseTensor<FD, FS, D, T, B>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a dense Tensor with shape {}", self.shape())
    }
}

struct DenseTensorVisitor<FD, FS, D, T> {
    txn_id: TxnId,
    file: FD,
    sparse: PhantomData<FS>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<FD, FS, D, T> DenseTensorVisitor<FD, FS, D, T> {
    fn new(txn_id: TxnId, file: FD) -> Self {
        Self {
            txn_id,
            file,
            sparse: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> de::Visitor for DenseTensorVisitor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Value = DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>;

    fn expecting() -> &'static str {
        "a dense tensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        debug!("deserialize DenseTensor");

        let schema = seq.next_element(()).await?;
        let schema = schema.ok_or_else(|| de::Error::invalid_length(0, "a tensor schema"))?;
        debug!("DenseTensor schema is {}", schema);

        let cxt = (self.txn_id, self.file, schema);
        let blocks = seq.next_element::<BlockListFile<FD, FS, D, T>>(cxt).await?;

        let blocks = blocks.ok_or_else(|| de::Error::invalid_length(1, "dense tensor data"))?;

        Ok(DenseTensor::from(blocks))
    }
}

#[async_trait]
impl<'en, FD, FS, D, T, B> IntoView<'en, D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;
    type View = DenseTensorView<'en>;

    async fn into_view(self, txn: T) -> TCResult<DenseTensorView<'en>> {
        let shape = self.shape().clone();
        let dtype = self.dtype();
        let blocks = self.blocks.block_stream(txn).await?;

        Ok(DenseTensorView {
            schema: Schema { shape, dtype },
            blocks: BlockStreamView { dtype, blocks },
        })
    }
}

/// A view of a [`DenseTensor`] as of a specific [`TxnId`], used in serialization
pub struct DenseTensorView<'en> {
    schema: Schema,
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
    blocks: TCBoxTryStream<'en, Array>,
}

impl<'en> en::IntoStream<'en> for BlockStreamView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use tc_value::{
            ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT,
        };

        fn encodable<'en, T: af::HasAfEnum + Clone + Default + 'en>(
            blocks: TCBoxTryStream<'en, Array>,
        ) -> impl Stream<Item = Vec<T>> + 'en {
            // an error can't be encoded within an array
            // so in case of a read error, let the receiver figure out that the tensor
            // doesn't have enough elements
            blocks
                .take_while(|r| future::ready(r.is_ok()))
                .map(|r| r.expect("tensor block").type_cast().to_vec())
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

fn encodable_c32<'en>(blocks: TCBoxTryStream<'en, Array>) -> impl Stream<Item = Vec<f32>> + 'en {
    blocks
        .take_while(|r| future::ready(r.is_ok()))
        .map(|block| block.expect("tensor block"))
        .map(|arr| {
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
}

fn encodable_c64<'en>(blocks: TCBoxTryStream<'en, Array>) -> impl Stream<Item = Vec<f64>> + 'en {
    blocks
        .take_while(|r| future::ready(r.is_ok()))
        .map(|block| block.expect("tensor block"))
        .map(|arr| {
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
}
