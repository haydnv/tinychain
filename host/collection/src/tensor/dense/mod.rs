use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use collate::Collate;
use destream::de;
use freqfs::FileLoad;
use futures::future::{self, TryFutureExt};
use futures::stream::{Stream, StreamExt, TryStreamExt};
use futures::{join, try_join};
use ha_ndarray::*;
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::{Complex, ComplexType, DType, Number, NumberCollator, NumberInstance, NumberType};
use tcgeneric::ThreadSafe;

use super::block::Block;
use super::complex::ComplexRead;
use super::sparse::{Node, SparseDense, SparseTensor};
use super::{
    Axes, Coord, Range, Shape, TensorBoolean, TensorBooleanConst, TensorCast, TensorCompare,
    TensorCompareConst, TensorConvert, TensorDiagonal, TensorInstance, TensorMath, TensorMathConst,
    TensorPermitRead, TensorPermitWrite, TensorRead, TensorReduce, TensorTransform, TensorUnary,
    TensorUnaryBoolean, TensorWrite, TensorWriteDual, IDEAL_BLOCK_SIZE,
};

pub use access::*;
pub use view::*;

mod access;
mod base;
mod stream;
mod view;

type BlockShape = ha_ndarray::Shape;
type BlockStream<Block> = Pin<Box<dyn Stream<Item = TCResult<Block>> + Send>>;

pub trait DenseCacheFile:
    FileLoad
    + AsType<Buffer<f32>>
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
    FE: FileLoad
        + AsType<Buffer<f32>>
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
pub trait DenseInstance: TensorInstance + ThreadSafe + fmt::Debug {
    type Block: NDArrayRead<DType = Self::DType> + NDArrayTransform + Into<Array<Self::DType>>;
    type DType: CDatatype + DType;

    fn block_size(&self) -> usize;

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block>;

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>>;

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType>;
}

#[async_trait]
impl<T: DenseInstance> DenseInstance for Box<T> {
    type Block = T::Block;
    type DType = T::DType;

    fn block_size(&self) -> usize {
        (&**self).block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        (**self).read_block(txn_id, block_id).await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        (*self).read_blocks(txn_id).await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        (**self).read_value(txn_id, coord).await
    }
}

#[async_trait]
pub trait DenseWrite: DenseInstance {
    type BlockWrite: NDArrayWrite<DType = Self::DType>;

    async fn write_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::BlockWrite>;

    async fn write_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::BlockWrite>>;
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

impl<Txn, FE, T: CDatatype> DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>> {
    pub fn from_access<A: Into<DenseAccess<Txn, FE, T>>>(accessor: A) -> Self {
        Self {
            accessor: accessor.into(),
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
    T: CDatatype + DType + fmt::Debug,
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
    T: CDatatype + DType,
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

impl<Txn, FE, A> TensorConvert for DenseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: DenseInstance + Clone,
    A::Block: NDArrayTransform,
    <A::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = A::DType> + NDArrayTransform + Into<Array<A::DType>>,
{
    type Dense = Self;
    type Sparse = SparseTensor<Txn, FE, SparseDense<A>>;

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
    T: CDatatype + DType,
{
    type Combine = DenseTensor<Txn, FE, DenseCombine<L, R, T>>;
    type LeftCombine = DenseTensor<Txn, FE, DenseCombine<L, R, T>>;

    fn add(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Combine> {
        fn add<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.add(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, add, |l, r| l + r).map(DenseTensor::from)
    }

    fn div(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn div<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.div(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, div, |l, r| l / r).map(DenseTensor::from)
    }

    fn log(self, base: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn log<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            let right = right.cast()?;
            left.log(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, base.accessor, log, |l: T, r: T| {
            T::from_float(l.to_float().log(r.to_float()))
        })
        .map(DenseTensor::from)
    }

    fn mul(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn mul<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.mul(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, mul, |l, r| l * r).map(DenseTensor::from)
    }

    fn pow(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn pow<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            let right = right.cast()?;
            left.pow(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, pow, |l: T, r: T| {
            T::from_float(l.to_float().pow(r.to_float()))
        })
        .map(DenseTensor::from)
    }

    fn sub(self, other: DenseTensor<Txn, FE, R>) -> TCResult<Self::Combine> {
        fn sub<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.sub(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, sub, |l, r| l - r).map(DenseTensor::from)
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
            |l, r| l + r,
        );

        Ok(accessor.into())
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        if n != A::DType::zero() {
            let accessor = DenseConst::new(
                self.accessor,
                n,
                |block, n| block.div_scalar(n).map(Array::from).map_err(TCError::from),
                |l, r| l / r,
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
            |block, n| {
                block
                    .log_scalar(n.to_float())
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |l, r| A::DType::from_float(l.to_float().log(r.to_float())),
        );

        Ok(accessor.into())
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(
            self.accessor,
            n,
            |block, n| block.mul_scalar(n).map(Array::from).map_err(TCError::from),
            |l, r| l * r,
        );

        Ok(accessor.into())
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(
            self.accessor,
            n,
            |block, n| {
                block
                    .pow_scalar(n.to_float())
                    .map(Array::from)
                    .map_err(TCError::from)
            },
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
            |l, r| l - r,
        );

        Ok(accessor.into())
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
            .read_permit(txn_id, coord.to_vec().into())
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
            .map(|result| result.and_then(|block| block.max().map_err(TCError::from)))
            .map_ok(Number::from)
            .try_fold(Number::from(A::DType::min()), |max, block_max| {
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
            .map(|result| result.and_then(|block| block.min().map_err(TCError::from)))
            .map_ok(Number::from)
            .try_fold(Number::from(A::DType::max()), |min, block_min| {
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
                .map(|result| result.and_then(|block| block.product().map_err(TCError::from)))
                .try_fold(A::DType::one(), |product, block_product| {
                    future::ready(Ok(product * block_product))
                })
                .await?;

            Ok(product.into())
        } else {
            Ok(A::DType::zero().into())
        }
    }

    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        DenseReduce::sum(self.accessor.into(), axes, keepdims).map(DenseTensor::from)
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let blocks = self.accessor.read_blocks(txn_id).await?;

        let sum = blocks
            .map(|result| result.and_then(|block| block.sum().map_err(TCError::from)))
            .try_fold(A::DType::zero(), |sum, block_sum| {
                future::ready(Ok(sum + block_sum))
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
                let _permit = this.write_permit(txn_id, coord.to_vec().into()).await?;
                let guard = this.write().await;
                guard.write_value(txn_id, coord, value.cast_into()).await
            },
            {
                let (r_value, i_value) = Complex::cast_from(value).into();

                // always acquire these locks in-order in order to avoid a deadlock
                let _r_permit = this.0.write_permit(txn_id, coord.to_vec().into()).await?;
                let _i_permit = this.1.write_permit(txn_id, coord.to_vec().into()).await?;

                let (r_guard, i_guard) = join!(this.0.write(), this.1.write());

                try_join!(
                    r_guard.write_value(txn_id, coord.to_vec(), r_value.cast_into()),
                    i_guard.write_value(txn_id, coord, i_value.cast_into())
                )?;

                Ok(())
            },
            {
                let _permit = this.write_permit(txn_id, coord.to_vec().into()).await?;
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

                if range.is_empty() || range == Range::all(this.shape()) {
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
                if range.is_empty() || range == Range::all(this.0.shape()) {
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
                // always acquire these permits in-order to avoid the risk of a deadlock
                let _write_permit = this.write_permit(txn_id, range.clone()).await?;
                let _read_permit = that.accessor.read_permit(txn_id, range.clone()).await?;

                if range.is_empty() || range == Range::all(this.shape()) {
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

impl<Txn: ThreadSafe, FE: ThreadSafe> fmt::Debug for DenseBase<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        base_dispatch!(
            self,
            this,
            this.fmt(f),
            write!(
                f,
                "a complex transactional dense tensor of type {:?}",
                this.0.dtype()
            ),
            this.fmt(f)
        )
    }
}

#[inline]
pub fn dense_from<Txn, FE, A, T>(
    tensor: DenseTensor<Txn, FE, A>,
) -> DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>
where
    A: Into<DenseAccess<Txn, FE, T>>,
    T: CDatatype,
{
    DenseTensor::from_access(tensor.into_inner())
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
fn div_ceil(num: u64, denom: u64) -> u64 {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}

#[inline]
fn ideal_block_size_for(shape: &[u64]) -> (usize, usize) {
    let ideal = IDEAL_BLOCK_SIZE as u64;
    let size = shape.iter().product::<u64>();
    let ndim = shape.len();

    if size < (2 * ideal) {
        (size as usize, 1)
    } else if ndim == 1 && size % ideal == 0 {
        (IDEAL_BLOCK_SIZE, (size / ideal) as usize)
    } else if ndim == 1 || (shape.iter().rev().take(2).product::<u64>() > (2 * ideal)) {
        let num_blocks = div_ceil(size, ideal) as usize;
        (IDEAL_BLOCK_SIZE, num_blocks as usize)
    } else {
        let matrix_size = shape.iter().rev().take(2).product::<u64>();
        let block_size = ideal + (matrix_size - (ideal % matrix_size));
        let num_blocks = div_ceil(size, ideal);
        (block_size as usize, num_blocks as usize)
    }
}
