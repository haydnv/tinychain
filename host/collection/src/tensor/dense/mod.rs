//! A dense tensor

use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use collate::Collate;
use destream::de;
use futures::future::{self, TryFutureExt};
use futures::stream::{Stream, StreamExt, TryStreamExt};
use futures::{join, try_join};
use ha_ndarray::*;
use log::{debug, trace};
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_transact::{fs, Transact, Transaction, TxnId};
use tc_value::{
    Complex, ComplexType, DType, Float as FP, FloatType, Int, IntType, Number, NumberCollator,
    NumberInstance, NumberType, UInt, UIntType, ValueType,
};
use tcgeneric::{Instance, NativeClass, TCPathBuf, ThreadSafe};

use super::block::Block;
use super::complex::ComplexRead;
use super::sparse::{Node, SparseDense, SparseTensor};
use super::{
    Axes, Coord, Range, Schema, Shape, TensorBoolean, TensorBooleanConst, TensorCast,
    TensorCompare, TensorCompareConst, TensorCond, TensorConvert, TensorDiagonal, TensorInstance,
    TensorMatMul, TensorMath, TensorMathConst, TensorPermitRead, TensorPermitWrite, TensorRead,
    TensorReduce, TensorTransform, TensorType, TensorUnary, TensorUnaryBoolean, TensorWrite,
    TensorWriteDual, IDEAL_BLOCK_SIZE, IMAG, REAL,
};

pub use access::*;
pub use view::*;

mod access;
mod base;
mod file;
mod stream;
mod view;

type BlockShape = ha_ndarray::Shape;
type BlockStream<T, Block> = Pin<Box<dyn Stream<Item = TCResult<Array<T, Block>>> + Send>>;

pub type Buffer<T> = ha_ndarray::Buffer<T>;

pub trait DenseCacheFile:
    AsType<Buffer<f32>>
    + AsType<Buffer<f64>>
    + AsType<Buffer<i16>>
    + AsType<Buffer<i32>>
    + AsType<Buffer<i64>>
    + AsType<Buffer<u8>>
    + AsType<Buffer<u16>>
    + AsType<Buffer<u32>>
    + AsType<Buffer<u64>>
    + ThreadSafe
{
}

impl<FE> DenseCacheFile for FE where
    FE: AsType<Buffer<f32>>
        + AsType<Buffer<f64>>
        + AsType<Buffer<i16>>
        + AsType<Buffer<i32>>
        + AsType<Buffer<i64>>
        + AsType<Buffer<u8>>
        + AsType<Buffer<u16>>
        + AsType<Buffer<u32>>
        + AsType<Buffer<u64>>
        + ThreadSafe
{
}

#[async_trait]
pub trait DenseInstance: TensorInstance + fmt::Debug {
    type Block: Access<Self::DType>;
    type DType: CType + DType;

    fn block_size(&self) -> usize;

    async fn read_block(
        &self,
        txn_id: TxnId,
        block_id: u64,
    ) -> TCResult<Array<Self::DType, Self::Block>>;

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::DType, Self::Block>>;

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType>;
}

#[async_trait]
impl<T: DenseInstance> DenseInstance for Box<T> {
    type Block = T::Block;
    type DType = T::DType;

    fn block_size(&self) -> usize {
        (&**self).block_size()
    }

    async fn read_block(
        &self,
        txn_id: TxnId,
        block_id: u64,
    ) -> TCResult<Array<Self::DType, Self::Block>> {
        (**self).read_block(txn_id, block_id).await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::DType, Self::Block>> {
        (*self).read_blocks(txn_id).await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        (**self).read_value(txn_id, coord).await
    }
}

#[async_trait]
pub trait DenseWrite: DenseInstance {
    type BlockWrite: AccessMut<Self::DType>;

    async fn write_block(
        &self,
        txn_id: TxnId,
        block_id: u64,
    ) -> TCResult<Array<Self::DType, Self::BlockWrite>>;

    async fn write_blocks(
        self,
        txn_id: TxnId,
    ) -> TCResult<BlockStream<Self::DType, Self::BlockWrite>>;
}

#[async_trait]
pub trait DenseWriteLock<'a>: DenseInstance {
    type WriteGuard: DenseWriteGuard<Self::DType>;

    async fn write(&'a self) -> Self::WriteGuard;
}

#[async_trait]
pub trait DenseWriteGuard<T>: Send + Sync {
    async fn overwrite<O>(&self, txn_id: TxnId, other: O) -> TCResult<()>
    where
        O: DenseInstance<DType = T> + TensorPermitRead;

    async fn overwrite_value(&self, txn_id: TxnId, value: T) -> TCResult<()>;

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: T) -> TCResult<()>;
}

pub struct DenseTensor<Txn, FE, A> {
    accessor: A,
    phantom: PhantomData<(Txn, FE)>,
}

impl<Txn, FE, A: Clone> Clone for DenseTensor<Txn, FE, A> {
    fn clone(&self) -> Self {
        Self {
            accessor: self.accessor.clone(),
            phantom: self.phantom,
        }
    }
}

impl<Txn, FE, A> DenseTensor<Txn, FE, A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<Txn, FE, T: CType> DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>> {
    pub fn from_access<A: Into<DenseAccess<Txn, FE, T>>>(accessor: A) -> Self {
        Self {
            accessor: accessor.into(),
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE, A> DenseTensor<Txn, FE, A>
where
    A: DenseInstance,
{
    fn block_size(&self) -> usize {
        self.accessor.block_size()
    }

    fn resize_blocks(self, block_size: usize) -> DenseTensor<Txn, FE, DenseResizeBlocks<A>> {
        DenseTensor {
            accessor: DenseResizeBlocks::new(self.accessor, block_size),
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE, A> TensorInstance for DenseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<Txn, FE, L, R, T> TensorBoolean<DenseTensor<Txn, FE, R>> for DenseTensor<Txn, FE, L>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    L: DenseInstance<DType = T> + Into<DenseAccess<Txn, FE, T>> + fmt::Debug,
    R: DenseInstance<DType = T> + Into<DenseAccess<Txn, FE, T>> + fmt::Debug,
    T: CType + DType + fmt::Debug,
    DenseAccessCast<Txn, FE>: From<DenseAccess<Txn, FE, T>>,
    DenseTensor<Txn, FE, R>: fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
    Self: fmt::Debug,
{
    type Combine = DenseTensor<Txn, FE, DenseCompare<Txn, FE, u8>>;
    type LeftCombine = DenseTensor<Txn, FE, DenseCompare<Txn, FE, u8>>;

    fn and(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            Block::and,
            |l, r| bool_u8(l.and(r)),
        )
        .map(DenseTensor::from)
    }

    fn or(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            Block::or,
            |l, r| bool_u8(l.or(r)),
        )
        .map(DenseTensor::from)
    }

    fn xor(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            Block::xor,
            |l, r| bool_u8(l.xor(r)),
        )
        .map(DenseTensor::from)
    }
}

impl<Txn, FE, A> TensorBooleanConst for DenseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: DenseInstance + Into<DenseAccess<Txn, FE, A::DType>>,
    DenseAccessCast<Txn, FE>: From<DenseAccess<Txn, FE, A::DType>>,
{
    type Combine = DenseTensor<Txn, FE, DenseCompareConst<Txn, FE, u8>>;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(
            DenseCompareConst::new(self.accessor.into(), other, Block::and_scalar, |l, r| {
                bool_u8(l.and(r))
            })
            .into(),
        )
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(
            DenseCompareConst::new(self.accessor.into(), other, Block::or_scalar, |l, r| {
                bool_u8(l.or(r))
            })
            .into(),
        )
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(
            DenseCompareConst::new(self.accessor.into(), other, Block::xor_scalar, |l, r| {
                bool_u8(l.xor(r))
            })
            .into(),
        )
    }
}

impl<Txn, FE, L, R, T> TensorCompare<DenseTensor<Txn, FE, R>> for DenseTensor<Txn, FE, L>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    L: DenseInstance<DType = T> + Into<DenseAccessCast<Txn, FE>>,
    R: DenseInstance<DType = T> + Into<DenseAccessCast<Txn, FE>>,
    T: CType + DType,
{
    type Compare = DenseTensor<Txn, FE, DenseCompare<Txn, FE, u8>>;

    fn eq(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::eq, |l, r| {
            bool_u8(l.eq(&r))
        })
        .map(DenseTensor::from)
    }

    fn gt(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::gt, |l, r| {
            bool_u8(l.gt(&r))
        })
        .map(DenseTensor::from)
    }

    fn ge(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::ge, |l, r| {
            bool_u8(l.ge(&r))
        })
        .map(DenseTensor::from)
    }

    fn lt(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::lt, |l, r| {
            bool_u8(l.lt(&r))
        })
        .map(DenseTensor::from)
    }

    fn le(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::le, |l, r| {
            bool_u8(l.le(&r))
        })
        .map(DenseTensor::from)
    }

    fn ne(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::ne, |l, r| {
            bool_u8(l.ne(&r))
        })
        .map(DenseTensor::from)
    }
}

impl<Txn, FE, A> TensorCompareConst for DenseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: DenseInstance + Into<DenseAccessCast<Txn, FE>>,
{
    type Compare = DenseTensor<Txn, FE, DenseCompareConst<Txn, FE, u8>>;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(
            DenseCompareConst::new(self.accessor, other, Block::eq_scalar, |l, r| {
                bool_u8(l.eq(&r))
            })
            .into(),
        )
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(
            DenseCompareConst::new(self.accessor, other, Block::gt_scalar, |l, r| {
                bool_u8(l.gt(&r))
            })
            .into(),
        )
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(
            DenseCompareConst::new(self.accessor, other, Block::ge_scalar, |l, r| {
                bool_u8(l.ge(&r))
            })
            .into(),
        )
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(
            DenseCompareConst::new(self.accessor, other, Block::lt_scalar, |l, r| {
                bool_u8(l.lt(&r))
            })
            .into(),
        )
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(
            DenseCompareConst::new(self.accessor, other, Block::le_scalar, |l, r| {
                bool_u8(l.le(&r))
            })
            .into(),
        )
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(
            DenseCompareConst::new(self.accessor, other, Block::ne_scalar, |l, r| {
                bool_u8(l.ne(&r))
            })
            .into(),
        )
    }
}

impl<Txn, FE, Cond, Then, OrElse, T>
    TensorCond<DenseTensor<Txn, FE, Then>, DenseTensor<Txn, FE, OrElse>>
    for DenseTensor<Txn, FE, Cond>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    Cond: DenseInstance<DType = u8> + fmt::Debug,
    Then: DenseInstance<DType = T> + fmt::Debug,
    OrElse: DenseInstance<DType = T> + fmt::Debug,
    T: CType,
{
    type Cond = DenseTensor<Txn, FE, DenseCond<Cond, Then, OrElse>>;

    fn cond(
        self,
        then: DenseTensor<Txn, FE, Then>,
        or_else: DenseTensor<Txn, FE, OrElse>,
    ) -> TCResult<Self::Cond> {
        DenseCond::new(self.accessor, then.accessor, or_else.accessor).map(DenseTensor::from)
    }
}

impl<Txn, FE, A> TensorConvert for DenseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: DenseInstance + Into<DenseAccess<Txn, FE, A::DType>> + Clone,
{
    type Dense = Self;
    type Sparse = SparseTensor<Txn, FE, SparseDense<Txn, FE, A::DType>>;

    fn into_dense(self) -> Self::Dense {
        self
    }

    fn into_sparse(self) -> Self::Sparse {
        SparseDense::new(self.accessor).into()
    }
}

impl<Txn, FE, A> TensorDiagonal for DenseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: DenseInstance,
{
    type Diagonal = DenseTensor<Txn, FE, DenseDiagonal<A>>;

    fn diagonal(self) -> TCResult<Self::Diagonal> {
        DenseDiagonal::new(self.accessor).map(DenseTensor::from)
    }
}

impl<Txn, FE, L, R, T> TensorMath<DenseTensor<Txn, FE, R>> for DenseTensor<Txn, FE, L>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    L: DenseInstance<DType = T>,
    R: DenseInstance<DType = T>,
    T: CType + DType,
{
    type Combine = DenseTensor<Txn, FE, DenseCombine<L, R, T>>;
    type LeftCombine = DenseTensor<Txn, FE, DenseCombine<L, R, T>>;

    fn add(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Combine> {
        fn add<T: CType>(left: ArrayAccess<T>, right: ArrayAccess<T>) -> TCResult<ArrayAccess<T>> {
            left.add(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, add, T::add).map(DenseTensor::from)
    }

    fn div(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn div<T: CType>(left: ArrayAccess<T>, right: ArrayAccess<T>) -> TCResult<ArrayAccess<T>> {
            left.div(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, div, T::div).map(DenseTensor::from)
    }

    fn log(self, base: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn log<T: CType>(left: ArrayAccess<T>, right: ArrayAccess<T>) -> TCResult<ArrayAccess<T>> {
            let right = right.cast()?;
            left.log(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, base.accessor, log, |l: T, r: T| {
            T::from_float(l.to_float().log(r.to_float()))
        })
        .map(DenseTensor::from)
    }

    fn mul(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn mul<T: CType>(left: ArrayAccess<T>, right: ArrayAccess<T>) -> TCResult<ArrayAccess<T>> {
            left.mul(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, mul, T::mul).map(DenseTensor::from)
    }

    fn pow(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn pow<T: CType>(left: ArrayAccess<T>, right: ArrayAccess<T>) -> TCResult<ArrayAccess<T>> {
            let right = right.cast()?;
            left.pow(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, pow, |l: T, r: T| {
            T::from_float(l.to_float().pow(r.to_float()))
        })
        .map(DenseTensor::from)
    }

    fn sub(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Combine> {
        fn sub<T: CType>(left: ArrayAccess<T>, right: ArrayAccess<T>) -> TCResult<ArrayAccess<T>> {
            left.sub(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, sub, T::sub).map(DenseTensor::from)
    }
}

impl<Txn, FE, A> TensorMathConst for DenseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: DenseInstance,
    Number: CastInto<A::DType>,
{
    type Combine = DenseTensor<Txn, FE, DenseConst<A, A::DType>>;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(
            self.accessor,
            n,
            |block, n| block.add_scalar(n).map(Array::from).map_err(TCError::from),
            A::DType::add,
        );

        Ok(accessor.into())
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        if n != A::DType::ZERO {
            let accessor = DenseConst::new(
                self.accessor,
                n,
                |block, n| block.div_scalar(n).map(Array::from).map_err(TCError::from),
                A::DType::div,
            );

            Ok(accessor.into())
        } else {
            Err(bad_request!("cannot divide {self:?} by {other}"))
        }
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        let n = base.cast_into();

        let accessor = DenseConst::new(
            self.accessor,
            n,
            |block, n| block.log_scalar(n).map(Array::from).map_err(TCError::from),
            |l, r| A::DType::from_float(l.to_float().log(r.to_float())),
        );

        Ok(accessor.into())
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        let accessor = DenseConst::new(
            self.accessor,
            other.cast_into(),
            |block, n| block.mul_scalar(n).map(Array::from).map_err(TCError::from),
            A::DType::mul,
        );

        Ok(accessor.into())
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(
            self.accessor,
            n,
            |block, n| block.pow_scalar(n).map(Array::from).map_err(TCError::from),
            |l, r| A::DType::from_float(l.to_float().pow(r.to_float())),
        );

        Ok(accessor.into())
    }

    fn sub_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(
            self.accessor,
            n,
            |block, n| block.sub_scalar(n).map(Array::from).map_err(TCError::from),
            A::DType::sub,
        );

        Ok(accessor.into())
    }
}

impl<Txn, FE, L, R, T> TensorMatMul<DenseTensor<Txn, FE, R>> for DenseTensor<Txn, FE, L>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    L: DenseInstance<DType = T>,
    R: DenseInstance<DType = T>,
    T: CType + DType,
{
    type MatMul = DenseTensor<Txn, FE, DenseMatMul<L, R>>;

    fn matmul(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::MatMul> {
        DenseMatMul::new(self.accessor, other.accessor).map(DenseTensor::from)
    }
}

#[async_trait]
impl<Txn, FE, A> TensorRead for DenseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<A::DType>> + AsType<Node>,
    A: DenseInstance + TensorPermitRead,
    Number: From<A::DType>,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        let _permit = self
            .accessor
            .read_permit(txn_id, coord.clone().into())
            .await?;

        self.accessor
            .read_value(txn_id, coord)
            .map_ok(Number::from)
            .await
    }
}

#[async_trait]
impl<Txn, FE, A> TensorReduce for DenseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<A::DType>> + AsType<Node>,
    A: DenseInstance + TensorPermitRead + Into<DenseAccess<Txn, FE, A::DType>> + Clone,
    A::DType: fmt::Debug,
    Buffer<A::DType>: de::FromStream<Context = ()>,
    Number: From<A::DType> + CastInto<A::DType>,
{
    type Reduce = DenseTensor<Txn, FE, DenseReduce<DenseAccess<Txn, FE, A::DType>, A::DType>>;

    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let mut blocks = self.accessor.read_blocks(txn_id).await?;

        while let Some(block) = blocks.try_next().await? {
            if !block.all()? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let mut blocks = self.accessor.read_blocks(txn_id).await?;

        while let Some(block) = blocks.try_next().await? {
            if block.any()? {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn max(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        DenseReduce::max(self.accessor.into(), axes, keepdims).map(DenseTensor::from)
    }

    async fn max_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let blocks = self.accessor.read_blocks(txn_id).await?;
        let collator = NumberCollator::default();

        let max = blocks
            .map(|result| result.and_then(|block| block.max_all().map_err(TCError::from)))
            .map_ok(Number::from)
            .try_fold(Number::from(A::DType::MIN), |max, block_max| {
                let max = match collator.cmp(&max, &block_max) {
                    Ordering::Greater | Ordering::Equal => max,
                    Ordering::Less => block_max,
                };

                future::ready(Ok(max))
            })
            .await?;

        Ok(max)
    }

    fn min(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        DenseReduce::min(self.accessor.into(), axes, keepdims).map(DenseTensor::from)
    }

    async fn min_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let blocks = self.accessor.read_blocks(txn_id).await?;
        let collator = NumberCollator::default();

        let min = blocks
            .map(|result| result.and_then(|block| block.min_all().map_err(TCError::from)))
            .map_ok(Number::from)
            .try_fold(Number::from(A::DType::MAX), |min, block_min| {
                let max = match collator.cmp(&min, &block_min) {
                    Ordering::Less | Ordering::Equal => min,
                    Ordering::Greater => block_min,
                };

                future::ready(Ok(max))
            })
            .await?;

        Ok(min)
    }

    fn product(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        DenseReduce::product(self.accessor.into(), axes, keepdims).map(DenseTensor::from)
    }

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;

        if self.clone().all(txn_id).await? {
            let blocks = self.accessor.read_blocks(txn_id).await?;

            let product = blocks
                .map(|result| result.and_then(|block| block.product_all().map_err(TCError::from)))
                .try_fold(A::DType::ONE, |product, block_product| {
                    future::ready(Ok(A::DType::mul(product, block_product)))
                })
                .await?;

            Ok(product.into())
        } else {
            Ok(A::DType::ZERO.into())
        }
    }

    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        DenseReduce::sum(self.accessor.into(), axes, keepdims).map(DenseTensor::from)
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let blocks = self.accessor.read_blocks(txn_id).await?;

        let sum = blocks
            .map(|result| result.and_then(|block| block.sum_all().map_err(TCError::from)))
            .try_fold(A::DType::ZERO, |sum, block_sum| {
                future::ready(Ok(A::DType::add(sum, block_sum)))
            })
            .await?;

        Ok(sum.into())
    }
}

impl<Txn, FE, A> TensorTransform for DenseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: DenseInstance,
{
    type Broadcast = DenseTensor<Txn, FE, DenseBroadcast<A>>;
    type Expand = DenseTensor<Txn, FE, DenseExpand<A>>;
    type Reshape = DenseTensor<Txn, FE, DenseReshape<A>>;
    type Slice = DenseTensor<Txn, FE, DenseSlice<A>>;
    type Transpose = DenseTensor<Txn, FE, DenseTranspose<A>>;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        DenseBroadcast::new(self.accessor, shape).map(DenseTensor::from)
    }

    fn expand(self, axes: Axes) -> TCResult<Self::Expand> {
        DenseExpand::new(self.accessor, axes).map(DenseTensor::from)
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        DenseReshape::new(self.accessor, shape).map(DenseTensor::from)
    }

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        DenseSlice::new(self.accessor, range).map(DenseTensor::from)
    }

    fn transpose(self, permutation: Option<Axes>) -> TCResult<Self::Transpose> {
        DenseTranspose::new(self.accessor, permutation).map(DenseTensor::from)
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe, A: DenseInstance> TensorUnary for DenseTensor<Txn, FE, A> {
    type Unary = DenseTensor<Txn, FE, DenseUnary<A, A::DType>>;

    fn abs(self) -> TCResult<Self::Unary> {
        Ok(DenseUnary::abs(self.accessor).into())
    }

    fn exp(self) -> TCResult<Self::Unary> {
        Ok(DenseUnary::exp(self.accessor).into())
    }

    fn ln(self) -> TCResult<Self::Unary> {
        Ok(DenseUnary::ln(self.accessor).into())
    }

    fn round(self) -> TCResult<Self::Unary> {
        Ok(DenseUnary::round(self.accessor).into())
    }
}

impl<Txn, FE, A> TensorUnaryBoolean for DenseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile,
    A: DenseInstance + Into<DenseAccessCast<Txn, FE>>,
{
    type Unary = DenseTensor<Txn, FE, DenseUnaryCast<Txn, FE, u8>>;

    fn not(self) -> TCResult<Self::Unary> {
        Ok(DenseUnaryCast::not(self.accessor).into())
    }
}

impl<Txn, FE, A> From<A> for DenseTensor<Txn, FE, A> {
    fn from(accessor: A) -> Self {
        Self {
            accessor,
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE, A: fmt::Debug> fmt::Debug for DenseTensor<Txn, FE, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.accessor.fmt(f)
    }
}

pub enum DenseBase<Txn, FE> {
    Bool(base::DenseBase<Txn, FE, u8>),
    C32((base::DenseBase<Txn, FE, f32>, base::DenseBase<Txn, FE, f32>)),
    C64((base::DenseBase<Txn, FE, f64>, base::DenseBase<Txn, FE, f64>)),
    F32(base::DenseBase<Txn, FE, f32>),
    F64(base::DenseBase<Txn, FE, f64>),
    I16(base::DenseBase<Txn, FE, i16>),
    I32(base::DenseBase<Txn, FE, i32>),
    I64(base::DenseBase<Txn, FE, i64>),
    U8(base::DenseBase<Txn, FE, u8>),
    U16(base::DenseBase<Txn, FE, u16>),
    U32(base::DenseBase<Txn, FE, u32>),
    U64(base::DenseBase<Txn, FE, u64>),
}

impl<Txn, FE> Clone for DenseBase<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Bool(this) => Self::Bool(this.clone()),
            Self::C32(this) => Self::C32(this.clone()),
            Self::C64(this) => Self::C64(this.clone()),
            Self::F32(this) => Self::F32(this.clone()),
            Self::F64(this) => Self::F64(this.clone()),
            Self::I16(this) => Self::I16(this.clone()),
            Self::I32(this) => Self::I32(this.clone()),
            Self::I64(this) => Self::I64(this.clone()),
            Self::U8(this) => Self::U8(this.clone()),
            Self::U16(this) => Self::U16(this.clone()),
            Self::U32(this) => Self::U32(this.clone()),
            Self::U64(this) => Self::U64(this.clone()),
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> Instance for DenseBase<Txn, FE> {
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Dense
    }
}

impl<Txn, FE> DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + Clone,
{
    pub async fn constant(
        store: fs::Dir<FE>,
        txn_id: TxnId,
        shape: Shape,
        value: Number,
    ) -> TCResult<Self> {
        match value {
            Number::Bool(n) => {
                base::DenseBase::constant(store, shape, if n.into() { 1u8 } else { 0 })
                    .map_ok(Self::Bool)
                    .await
            }
            Number::Complex(Complex::C32(n)) => {
                let shape_clone = shape.clone();
                let store_clone = store.clone();
                let re = store_clone
                    .create_dir(txn_id, REAL.into())
                    .and_then(|store| base::DenseBase::constant(store, shape_clone, n.re));

                let im = store
                    .create_dir(txn_id, IMAG.into())
                    .and_then(|store| base::DenseBase::constant(store, shape, n.im));

                try_join!(re, im).map(Self::C32)
            }
            Number::Complex(Complex::C64(n)) => {
                let shape_clone = shape.clone();
                let store_clone = store.clone();
                let re = store_clone
                    .create_dir(txn_id, REAL.into())
                    .and_then(|store| base::DenseBase::constant(store, shape_clone, n.re));

                let im = store
                    .create_dir(txn_id, IMAG.into())
                    .and_then(|store| base::DenseBase::constant(store, shape, n.im));

                try_join!(re, im).map(Self::C64)
            }
            Number::Float(FP::F32(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::F32)
                    .await
            }
            Number::Float(FP::F64(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::F64)
                    .await
            }
            Number::Int(Int::I16(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::I16)
                    .await
            }
            Number::Int(Int::I32(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::I32)
                    .await
            }
            Number::Int(Int::I64(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::I64)
                    .await
            }
            Number::UInt(UInt::U8(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::U8)
                    .await
            }
            Number::UInt(UInt::U16(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::U16)
                    .await
            }
            Number::UInt(UInt::U32(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::U32)
                    .await
            }
            Number::UInt(UInt::U64(n)) => {
                base::DenseBase::constant(store, shape, n)
                    .map_ok(Self::U64)
                    .await
            }
            other => Err(bad_request!("unsupported data type: {:?}", other.class())),
        }
    }

    pub async fn from_values(
        store: fs::Dir<FE>,
        txn_id: TxnId,
        shape: Shape,
        dtype: NumberType,
        values: Vec<Number>,
    ) -> TCResult<Self> {
        match dtype {
            NumberType::Bool => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::Bool)
                    .await
            }
            NumberType::Complex(ct) => {
                let mut re = Vec::with_capacity(values.len());
                let mut im = Vec::with_capacity(values.len());
                for n in values {
                    let (r, i) = Complex::cast_from(n).into();
                    re.push(Number::Float(r));
                    im.push(Number::Float(i));
                }

                let (store_re, store_im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                match ct {
                    ComplexType::Complex | ComplexType::C32 => {
                        let (re, im) = try_join!(
                            base::DenseBase::from_values(store_re, shape.clone(), re),
                            base::DenseBase::from_values(store_im, shape, im)
                        )?;

                        Ok(Self::C32((re, im)))
                    }
                    ComplexType::C64 => {
                        let (re, im) = try_join!(
                            base::DenseBase::from_values(store_re, shape.clone(), re),
                            base::DenseBase::from_values(store_im, shape, im)
                        )?;

                        Ok(Self::C64((re, im)))
                    }
                }
            }
            NumberType::Number
            | NumberType::Float(FloatType::Float)
            | NumberType::Float(FloatType::F32) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::F32)
                    .await
            }
            NumberType::Float(FloatType::F64) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::F64)
                    .await
            }
            NumberType::Int(IntType::I16) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::F32)
                    .await
            }
            NumberType::Int(IntType::Int) | NumberType::Int(IntType::I32) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::I32)
                    .await
            }
            NumberType::Int(IntType::I64) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::I64)
                    .await
            }
            NumberType::UInt(UIntType::U8) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::U8)
                    .await
            }
            NumberType::UInt(UIntType::U16) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::U16)
                    .await
            }
            NumberType::UInt(UIntType::UInt) | NumberType::UInt(UIntType::U32) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::U32)
                    .await
            }
            NumberType::UInt(UIntType::U64) => {
                base::DenseBase::from_values(store, shape, values)
                    .map_ok(Self::U64)
                    .await
            }
            other => Err(bad_request!("cannot construct a range of type {other:?}")),
        }
    }

    pub async fn range(
        store: fs::Dir<FE>,
        shape: Shape,
        start: Number,
        stop: Number,
    ) -> TCResult<Self> {
        let dtype = Ord::max(start.class(), stop.class());

        match dtype {
            NumberType::Bool => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::Bool)
                    .await
            }
            NumberType::Complex(_) => Err(not_implemented!(
                "construct a range of complex numbers [{start}..{stop})"
            )),
            NumberType::Number
            | NumberType::Float(FloatType::Float)
            | NumberType::Float(FloatType::F32) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::F32)
                    .await
            }
            NumberType::Float(FloatType::F64) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::F64)
                    .await
            }
            NumberType::Int(IntType::I16) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::I32)
                    .await
            }
            NumberType::Int(IntType::Int) | NumberType::Int(IntType::I32) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::I32)
                    .await
            }
            NumberType::Int(IntType::I64) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::I64)
                    .await
            }
            NumberType::UInt(UIntType::U8) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::U8)
                    .await
            }
            NumberType::UInt(UIntType::U16) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::U16)
                    .await
            }
            NumberType::UInt(UIntType::UInt) | NumberType::UInt(UIntType::U32) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::U32)
                    .await
            }
            NumberType::UInt(UIntType::U64) => {
                base::DenseBase::range(store, shape, start.cast_into(), stop.cast_into())
                    .map_ok(Self::U64)
                    .await
            }
            other => Err(bad_request!("cannot construct a range of type {other:?}")),
        }
    }

    pub async fn random_normal(
        store: fs::Dir<FE>,
        shape: Shape,
        mean: f32,
        std: f32,
    ) -> TCResult<Self> {
        base::DenseBase::random_normal(store, shape, mean, std)
            .map_ok(Self::F32)
            .await
    }

    pub async fn random_uniform(store: fs::Dir<FE>, shape: Shape) -> TCResult<Self> {
        base::DenseBase::random_uniform(store, shape)
            .map_ok(Self::F32)
            .await
    }
}

macro_rules! base_dispatch {
    ($this:ident, $var:ident, $bool:expr, $complex:expr, $general:expr) => {
        match $this {
            DenseBase::Bool($var) => $bool,
            DenseBase::C32($var) => $complex,
            DenseBase::C64($var) => $complex,
            DenseBase::F32($var) => $general,
            DenseBase::F64($var) => $general,
            DenseBase::I16($var) => $general,
            DenseBase::I32($var) => $general,
            DenseBase::I64($var) => $general,
            DenseBase::U8($var) => $general,
            DenseBase::U16($var) => $general,
            DenseBase::U32($var) => $general,
            DenseBase::U64($var) => $general,
        }
    };
}

macro_rules! base_view_dispatch {
    ($self:ident, $other:ident, $this:ident, $that:ident, $bool:expr, $complex:expr, $general:expr, $mismatch:expr) => {
        match ($self, $other) {
            (DenseBase::Bool($this), DenseView::Bool($that)) => $bool,
            (DenseBase::C32($this), DenseView::C32($that)) => $complex,
            (DenseBase::C64($this), DenseView::C64($that)) => $complex,
            (DenseBase::F32($this), DenseView::F32($that)) => $general,
            (DenseBase::F64($this), DenseView::F64($that)) => $general,
            (DenseBase::I16($this), DenseView::I16($that)) => $general,
            (DenseBase::I32($this), DenseView::I32($that)) => $general,
            (DenseBase::I64($this), DenseView::I64($that)) => $general,
            (DenseBase::U8($this), DenseView::U8($that)) => $general,
            (DenseBase::U16($this), DenseView::U16($that)) => $general,
            (DenseBase::U32($this), DenseView::U32($that)) => $general,
            (DenseBase::U64($this), DenseView::U64($that)) => $general,
            ($this, $that) => $mismatch,
        }
    };
}

impl<Txn, FE> TensorInstance for DenseBase<Txn, FE>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
{
    fn dtype(&self) -> NumberType {
        match self {
            Self::Bool(this) => this.dtype(),
            Self::C32(_) => NumberType::Complex(ComplexType::C32),
            Self::C64(_) => NumberType::Complex(ComplexType::C64),
            Self::F32(this) => this.dtype(),
            Self::F64(this) => this.dtype(),
            Self::I16(this) => this.dtype(),
            Self::I32(this) => this.dtype(),
            Self::I64(this) => this.dtype(),
            Self::U8(this) => this.dtype(),
            Self::U16(this) => this.dtype(),
            Self::U32(this) => this.dtype(),
            Self::U64(this) => this.dtype(),
        }
    }

    fn shape(&self) -> &Shape {
        base_dispatch!(self, this, this.shape(), this.0.shape(), this.shape())
    }
}

#[async_trait]
impl<Txn, FE> TensorRead for DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        base_dispatch!(
            self,
            this,
            this.read_value(txn_id, coord).map_ok(Number::from).await,
            ComplexRead::read_value((Self::from(this.0), Self::from(this.1)), txn_id, coord).await,
            this.read_value(txn_id, coord).map_ok(Number::from).await
        )
    }
}

#[async_trait]
impl<Txn, FE> TensorWrite for DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()> {
        base_dispatch!(
            self,
            this,
            {
                let _permit = this.write_permit(txn_id, range.clone()).await?;
                let slice = DenseSlice::new(this.clone(), range)?;
                let slice = slice.write().await;
                slice.overwrite_value(txn_id, value.cast_into()).await
            },
            {
                let (r_value, i_value) = Complex::cast_from(value).into();

                // always acquire these locks in-order to avoid the risk of a deadlock
                let _r_permit = this.0.write_permit(txn_id, range.clone()).await?;
                let _i_permit = this.1.write_permit(txn_id, range.clone()).await?;

                let r_slice = DenseSlice::new(this.0.clone(), range.clone())?;
                let i_slice = DenseSlice::new(this.1.clone(), range)?;
                let (r_slice, i_slice) = join!(r_slice.write(), i_slice.write());

                try_join!(
                    r_slice.overwrite_value(txn_id, r_value.cast_into()),
                    i_slice.overwrite_value(txn_id, i_value.cast_into())
                )?;

                Ok(())
            },
            {
                let _permit = this.write_permit(txn_id, range.clone()).await?;
                let slice = DenseSlice::new(this.clone(), range)?;
                let slice = slice.write().await;
                slice.overwrite_value(txn_id, value.cast_into()).await
            }
        )
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        base_dispatch!(
            self,
            this,
            {
                let _permit = this.write_permit(txn_id, coord.clone().into()).await?;
                let guard = this.write().await;
                guard.write_value(txn_id, coord, value.cast_into()).await
            },
            {
                let (r_value, i_value) = Complex::cast_from(value).into();

                // always acquire these locks in-order in order to avoid a deadlock
                let _r_permit = this.0.write_permit(txn_id, coord.clone().into()).await?;
                let _i_permit = this.1.write_permit(txn_id, coord.clone().into()).await?;

                let (r_guard, i_guard) = join!(this.0.write(), this.1.write());

                try_join!(
                    r_guard.write_value(txn_id, coord.clone(), r_value.cast_into()),
                    i_guard.write_value(txn_id, coord, i_value.cast_into())
                )?;

                Ok(())
            },
            {
                let _permit = this.write_permit(txn_id, coord.clone().into()).await?;
                let guard = this.write().await;
                guard.write_value(txn_id, coord, value.cast_into()).await
            }
        )
    }
}

#[async_trait]
impl<Txn, FE> TensorWriteDual<DenseView<Txn, FE>> for DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn write(self, txn_id: TxnId, range: Range, value: DenseView<Txn, FE>) -> TCResult<()> {
        base_view_dispatch!(
            self,
            value,
            this,
            that,
            {
                // always acquire these permits in-order to avoid the risk of a deadlock
                let _write_permit = this.write_permit(txn_id, range.clone()).await?;
                let _read_permit = that.accessor.read_permit(txn_id, range.clone()).await?;

                if this.shape().is_covered_by(&range) {
                    let guard = this.write().await;
                    guard.overwrite(txn_id, that.accessor).await
                } else {
                    let slice = DenseSlice::new(this.clone(), range)?;
                    let guard = slice.write().await;
                    guard.overwrite(txn_id, that.accessor).await
                }
            },
            {
                // always acquire these permits in-order to avoid the risk of a deadlock
                let _r_this_permit = this.0.write_permit(txn_id, range.clone()).await?;
                let _i_this_permit = this.1.write_permit(txn_id, range.clone()).await?;
                let _r_that_permit = that.0.accessor.read_permit(txn_id, range.clone()).await?;
                let _i_that_permit = that.1.accessor.read_permit(txn_id, range.clone()).await?;

                debug_assert_eq!(this.0.shape(), this.1.shape());
                if this.0.shape().is_covered_by(&range) {
                    let (r_guard, i_guard) = join!(this.0.write(), this.1.write());

                    try_join!(
                        r_guard.overwrite(txn_id, that.0.accessor),
                        i_guard.overwrite(txn_id, that.1.accessor)
                    )?;

                    Ok(())
                } else {
                    let r_slice = DenseSlice::new(this.0.clone(), range.clone())?;
                    let i_slice = DenseSlice::new(this.1.clone(), range)?;

                    let (r_guard, i_guard) = join!(r_slice.write(), i_slice.write());

                    try_join!(
                        r_guard.overwrite(txn_id, that.0.accessor),
                        i_guard.overwrite(txn_id, that.1.accessor),
                    )?;

                    Ok(())
                }
            },
            {
                trace!("acquiring write permit on {this:?}[{range:?}]...");

                // always acquire these permits in-order to avoid the risk of a deadlock
                let write_permit = this.write_permit(txn_id, range.clone()).await?;
                trace!("acquired write permit {write_permit:?}");

                let read_permit = that.accessor.read_permit(txn_id, Range::default()).await?;
                trace!("acquired read permit {read_permit:?}");

                if this.shape().is_covered_by(&range) {
                    let guard = this.write().await;
                    guard.overwrite(txn_id, that.accessor).await
                } else {
                    let slice = DenseSlice::new(this.clone(), range)?;
                    let guard = slice.write().await;
                    guard.overwrite(txn_id, that.accessor).await
                }
            },
            {
                let value = TensorCast::cast_into(that, this.dtype())?;
                this.write(txn_id, range, value).await
            }
        )
    }
}

#[async_trait]
impl<Txn, FE> Transact for DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + for<'en> fs::FileSave<'en> + Clone,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        base_dispatch!(
            self,
            this,
            this.commit(txn_id).await,
            {
                join!(this.0.commit(txn_id), this.1.commit(txn_id));
            },
            this.commit(txn_id).await
        )
    }

    async fn rollback(&self, txn_id: &TxnId) {
        base_dispatch!(
            self,
            this,
            this.rollback(txn_id).await,
            {
                join!(this.0.rollback(txn_id), this.1.rollback(txn_id));
            },
            this.rollback(txn_id).await
        )
    }

    async fn finalize(&self, txn_id: &TxnId) {
        base_dispatch!(
            self,
            this,
            this.finalize(txn_id).await,
            {
                join!(this.0.finalize(txn_id), this.1.finalize(txn_id));
            },
            this.finalize(txn_id).await
        )
    }
}

#[async_trait]
impl<Txn, FE> fs::Persist<FE> for DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + Clone,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        match schema.dtype {
            NumberType::Bool => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::Bool)
                    .await
            }
            NumberType::Complex(ComplexType::C32) => {
                let (re, im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::DenseBase::create(txn_id, schema.shape.clone(), re),
                    base::DenseBase::create(txn_id, schema.shape, im)
                )?;

                Ok(Self::C32((re, im)))
            }
            NumberType::Complex(ComplexType::C64) => {
                let (re, im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::DenseBase::create(txn_id, schema.shape.clone(), re),
                    base::DenseBase::create(txn_id, schema.shape, im)
                )?;

                Ok(Self::C64((re, im)))
            }
            NumberType::Float(FloatType::F32) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::F32)
                    .await
            }
            NumberType::Float(FloatType::F64) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::F64)
                    .await
            }
            NumberType::Int(IntType::I16) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::I16)
                    .await
            }
            NumberType::Int(IntType::I32) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::I32)
                    .await
            }
            NumberType::Int(IntType::I64) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::I64)
                    .await
            }
            NumberType::UInt(UIntType::U8) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::U8)
                    .await
            }
            NumberType::UInt(UIntType::U16) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::U16)
                    .await
            }
            NumberType::UInt(UIntType::U32) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::U32)
                    .await
            }
            NumberType::UInt(UIntType::U64) => {
                base::DenseBase::create(txn_id, schema.shape, store)
                    .map_ok(Self::U64)
                    .await
            }
            other => Err(bad_request!(
                "cannot create a dense tensor of type {other:?}"
            )),
        }
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        match schema.dtype {
            NumberType::Bool => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::Bool)
                    .await
            }
            NumberType::Complex(ComplexType::C32) => {
                let (re, im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::DenseBase::load(txn_id, schema.shape.clone(), re),
                    base::DenseBase::load(txn_id, schema.shape, im)
                )?;

                Ok(Self::C32((re, im)))
            }
            NumberType::Complex(ComplexType::C64) => {
                let (re, im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::DenseBase::load(txn_id, schema.shape.clone(), re),
                    base::DenseBase::load(txn_id, schema.shape, im)
                )?;

                Ok(Self::C64((re, im)))
            }
            NumberType::Float(FloatType::F32) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::F32)
                    .await
            }
            NumberType::Float(FloatType::F64) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::F64)
                    .await
            }
            NumberType::Int(IntType::I16) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::I16)
                    .await
            }
            NumberType::Int(IntType::I32) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::I32)
                    .await
            }
            NumberType::Int(IntType::I64) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::I64)
                    .await
            }
            NumberType::UInt(UIntType::U8) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::U8)
                    .await
            }
            NumberType::UInt(UIntType::U16) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::U16)
                    .await
            }
            NumberType::UInt(UIntType::U32) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::U32)
                    .await
            }
            NumberType::UInt(UIntType::U64) => {
                base::DenseBase::load(txn_id, schema.shape, store)
                    .map_ok(Self::U64)
                    .await
            }
            other => Err(bad_request!("cannot load a dense tensor of type {other:?}")),
        }
    }

    fn dir(&self) -> fs::Inner<FE> {
        base_dispatch!(
            self,
            this,
            this.dir(),
            this.0.dir(), // FIXME: this should return the parent dir!
            this.dir()
        )
    }
}

#[async_trait]
impl<Txn, FE> fs::CopyFrom<FE, DenseView<Txn, FE>> for DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn copy_from(
        txn: &Txn,
        store: fs::Dir<FE>,
        instance: DenseView<Txn, FE>,
    ) -> TCResult<Self> {
        match instance {
            DenseView::Bool(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::Bool)
                    .await
            }
            DenseView::C32((re, im)) => {
                let txn_id = *txn.id();
                let (r_dir, i_dir) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::DenseBase::copy_from(txn, r_dir, re.into_inner()),
                    base::DenseBase::copy_from(txn, i_dir, im.into_inner())
                )?;

                Ok(Self::C32((re, im)))
            }
            DenseView::C64((re, im)) => {
                let txn_id = *txn.id();
                let (r_dir, i_dir) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::DenseBase::copy_from(txn, r_dir, re.into_inner()),
                    base::DenseBase::copy_from(txn, i_dir, im.into_inner())
                )?;

                Ok(Self::C64((re, im)))
            }
            DenseView::F32(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::F32)
                    .await
            }
            DenseView::F64(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::F64)
                    .await
            }
            DenseView::I16(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::I16)
                    .await
            }
            DenseView::I32(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::I32)
                    .await
            }
            DenseView::I64(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::I64)
                    .await
            }
            DenseView::U8(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::U8)
                    .await
            }
            DenseView::U16(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::U16)
                    .await
            }
            DenseView::U32(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::U32)
                    .await
            }
            DenseView::U64(that) => {
                base::DenseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::U64)
                    .await
            }
        }
    }
}

#[async_trait]
impl<Txn, FE> fs::Restore<FE> for DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        match (self, backup) {
            (Self::Bool(this), Self::Bool(that)) => this.restore(txn_id, that).await,
            (Self::C32((lr, li)), Self::C32((rr, ri))) => {
                try_join!(lr.restore(txn_id, rr), li.restore(txn_id, ri))?;
                Ok(())
            }
            (Self::C64((lr, li)), Self::C64((rr, ri))) => {
                try_join!(lr.restore(txn_id, rr), li.restore(txn_id, ri))?;
                Ok(())
            }
            (Self::F32(this), Self::F32(that)) => this.restore(txn_id, that).await,
            (Self::F64(this), Self::F64(that)) => this.restore(txn_id, that).await,
            (Self::I16(this), Self::I16(that)) => this.restore(txn_id, that).await,
            (Self::I32(this), Self::I32(that)) => this.restore(txn_id, that).await,
            (Self::I64(this), Self::I64(that)) => this.restore(txn_id, that).await,
            (Self::U8(this), Self::U8(that)) => this.restore(txn_id, that).await,
            (Self::U16(this), Self::U16(that)) => this.restore(txn_id, that).await,
            (Self::U32(this), Self::U32(that)) => this.restore(txn_id, that).await,
            (Self::U64(this), Self::U64(that)) => this.restore(txn_id, that).await,
            (this, that) => Err(bad_request!("cannot restore {this:?} from {that:?}")),
        }
    }
}

#[async_trait]
impl<Txn, FE> de::FromStream for DenseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + Clone,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(DenseVisitor::new(txn)).await
    }
}

impl<Txn, FE> From<base::DenseBase<Txn, FE, f32>> for DenseBase<Txn, FE> {
    fn from(base: base::DenseBase<Txn, FE, f32>) -> Self {
        Self::F32(base)
    }
}

impl<Txn, FE> From<base::DenseBase<Txn, FE, f64>> for DenseBase<Txn, FE> {
    fn from(base: base::DenseBase<Txn, FE, f64>) -> Self {
        Self::F64(base)
    }
}

impl<Txn, FE> From<DenseBase<Txn, FE>> for DenseView<Txn, FE> {
    fn from(base: DenseBase<Txn, FE>) -> Self {
        match base {
            DenseBase::Bool(this) => DenseView::Bool(dense_from(this.into())),
            DenseBase::C32((re, im)) => {
                DenseView::C32((dense_from(re.into()), dense_from(im.into())))
            }
            DenseBase::C64((re, im)) => {
                DenseView::C64((dense_from(re.into()), dense_from(im.into())))
            }
            DenseBase::F32(this) => DenseView::F32(dense_from(this.into())),
            DenseBase::F64(this) => DenseView::F64(dense_from(this.into())),
            DenseBase::I16(this) => DenseView::I16(dense_from(this.into())),
            DenseBase::I32(this) => DenseView::I32(dense_from(this.into())),
            DenseBase::I64(this) => DenseView::I64(dense_from(this.into())),
            DenseBase::U8(this) => DenseView::U8(dense_from(this.into())),
            DenseBase::U16(this) => DenseView::U16(dense_from(this.into())),
            DenseBase::U32(this) => DenseView::U32(dense_from(this.into())),
            DenseBase::U64(this) => DenseView::U64(dense_from(this.into())),
        }
    }
}

impl<Txn, FE> fmt::Debug for DenseBase<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        base_dispatch!(
            self,
            this,
            this.fmt(f),
            write!(f, "a complex tensor ({:?}, {:?})", this.0, this.1),
            this.fmt(f)
        )
    }
}

struct DenseVisitor<Txn, FE> {
    txn: Txn,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> DenseVisitor<Txn, FE> {
    fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<Txn, FE> de::Visitor for DenseVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + Clone,
{
    type Value = DenseBase<Txn, FE>;

    fn expecting() -> &'static str {
        "a dense tensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let (dtype, shape) = seq.expect_next::<(TCPathBuf, Shape)>(()).await?;
        let dtype = if let Some(ValueType::Number(dtype)) = ValueType::from_path(&dtype) {
            Ok(dtype)
        } else {
            Err(de::Error::invalid_type(dtype, "a type of number"))
        }?;

        match dtype {
            NumberType::Bool => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::Bool)
                    .await
            }
            NumberType::Complex(ComplexType::C32) => {
                let visitor = seq
                    .expect_next::<base::DenseComplexBaseVisitor<Txn, FE, f32>>((self.txn, shape))
                    .await?;

                visitor
                    .end()
                    .map_ok(DenseBase::C32)
                    .map_err(de::Error::custom)
                    .await
            }
            NumberType::Complex(ComplexType::C64) => {
                let visitor = seq
                    .expect_next::<base::DenseComplexBaseVisitor<Txn, FE, f64>>((self.txn, shape))
                    .await?;

                visitor
                    .end()
                    .map_ok(DenseBase::C64)
                    .map_err(de::Error::custom)
                    .await
            }
            NumberType::Float(FloatType::F32) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::F32)
                    .await
            }
            NumberType::Float(FloatType::F64) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::F64)
                    .await
            }
            NumberType::Int(IntType::I16) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::I16)
                    .await
            }
            NumberType::Int(IntType::I32) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::I32)
                    .await
            }
            NumberType::Int(IntType::I64) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::I64)
                    .await
            }
            NumberType::UInt(UIntType::U8) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::U8)
                    .await
            }
            NumberType::UInt(UIntType::U16) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::U16)
                    .await
            }
            NumberType::UInt(UIntType::U32) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::U32)
                    .await
            }
            NumberType::UInt(UIntType::U64) => {
                seq.expect_next((self.txn, shape))
                    .map_ok(DenseBase::U64)
                    .await
            }
            other => Err(de::Error::invalid_type(other, "a specific type of number")),
        }
    }
}

#[inline]
fn block_axis_for(shape: &[u64], block_size: usize) -> usize {
    let block_size = block_size as u64;

    debug_assert!(!shape.is_empty());
    debug_assert!(shape.iter().product::<u64>() >= block_size);

    let mut block_axis = shape.len() - 1;
    let mut trailing_size = shape[block_axis];
    for x in (0..(shape.len() - 1)).rev() {
        let size_from = trailing_size * shape[x];
        if size_from > block_size {
            break;
        } else {
            block_axis = x;
            trailing_size = size_from;
        }
    }

    block_axis
}

#[inline]
fn block_map_for(
    num_blocks: u64,
    shape: &[u64],
    block_shape: &[usize],
) -> TCResult<ArrayBuf<u64, StackVec<u64>>> {
    debug!("construct a block map for {shape:?} with block shape {block_shape:?}");

    debug_assert!(shape.len() >= block_shape.len());

    let block_axis = shape.len() - block_shape.len();
    let mut block_map_shape = BlockShape::with_capacity(block_axis + 1);
    block_map_shape.extend(
        shape
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim as usize),
    );

    block_map_shape.push(shape[block_axis].div_ceil(block_shape[0] as u64) as usize);

    ArrayBuf::new(
        (0..num_blocks as u64).into_iter().collect(),
        block_map_shape,
    )
    .map_err(TCError::from)
}

#[inline]
fn block_shape_for(axis: usize, shape: &[u64], block_size: usize) -> BlockShape {
    assert_ne!(block_size, 0);
    assert_ne!(shape.iter().product::<u64>(), 0, "invalid shape: {shape:?}");

    if axis == shape.len() - 1 {
        shape![block_size]
    } else {
        let mut block_shape = shape[axis..]
            .iter()
            .copied()
            .map(|dim| dim as usize)
            .collect::<BlockShape>();

        let trailing_size = block_shape.iter().skip(1).product::<usize>();

        block_shape[0] = if block_size % trailing_size == 0 {
            block_size / trailing_size
        } else {
            (block_size / trailing_size) + 1
        };

        block_shape
    }
}

#[inline]
fn bool_u8<N>(n: N) -> u8
where
    bool: CastFrom<N>,
{
    if bool::cast_from(n) {
        1
    } else {
        0
    }
}

#[inline]
pub fn dense_from<Txn, FE, A, T>(
    tensor: DenseTensor<Txn, FE, A>,
) -> DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>
where
    A: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    DenseTensor::from_access(tensor.into_inner())
}

#[inline]
fn ideal_block_size_for(shape: &[u64]) -> (usize, usize) {
    let ideal = IDEAL_BLOCK_SIZE as u64;
    let size = shape.iter().product::<u64>();
    let ndim = shape.len();

    assert_ne!(ndim, 0);
    assert_ne!(size, 0);

    if size < (2 * ideal) {
        (size as usize, 1)
    } else if ndim == 1 && size % ideal == 0 {
        (IDEAL_BLOCK_SIZE, (size / ideal) as usize)
    } else if ndim == 1 || (shape.iter().rev().take(2).product::<u64>() > (2 * ideal)) {
        let num_blocks = size.div_ceil(ideal) as usize;
        (IDEAL_BLOCK_SIZE, num_blocks as usize)
    } else {
        let matrix_size = shape.iter().rev().take(2).product::<u64>();
        let block_size = ideal + (matrix_size - (ideal % matrix_size));
        let num_blocks = size.div_ceil(ideal);
        (block_size as usize, num_blocks as usize)
    }
}
