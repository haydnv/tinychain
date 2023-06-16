use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use collate::Collate;
use destream::de;
use freqfs::FileLoad;
use futures::future;
use futures::stream::{Stream, StreamExt, TryStreamExt};
use ha_ndarray::*;
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{DType, Number, NumberCollator, NumberInstance, NumberType};
use tcgeneric::ThreadSafe;

use super::block::Block;
use super::sparse::{Node, SparseDense, SparseTensor};
use super::{
    Axes, Coord, Range, Shape, TensorBoolean, TensorBooleanConst, TensorCompare,
    TensorCompareConst, TensorConvert, TensorDiagonal, TensorInstance, TensorMath, TensorMathConst,
    TensorPermitRead, TensorReduce, TensorTransform, TensorUnary, TensorUnaryBoolean,
    IDEAL_BLOCK_SIZE,
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

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block>;

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>>;

    async fn read_value(&self, coord: Coord) -> TCResult<Self::DType>;
}

#[async_trait]
impl<T: DenseInstance> DenseInstance for Box<T> {
    type Block = T::Block;
    type DType = T::DType;

    fn block_size(&self) -> usize {
        (&**self).block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        (**self).read_block(block_id).await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        (*self).read_blocks().await
    }

    async fn read_value(&self, coord: Coord) -> TCResult<Self::DType> {
        (**self).read_value(coord).await
    }
}

#[async_trait]
pub trait DenseWrite: DenseInstance {
    type BlockWrite: NDArrayWrite<DType = Self::DType>;

    async fn write_block(&self, block_id: u64) -> TCResult<Self::BlockWrite>;

    async fn write_blocks(self) -> TCResult<BlockStream<Self::BlockWrite>>;
}

#[async_trait]
pub trait DenseWriteLock<'a>: DenseInstance {
    type WriteGuard: DenseWriteGuard<Self::DType>;

    async fn write(&'a self) -> Self::WriteGuard;
}

#[async_trait]
pub trait DenseWriteGuard<T>: Send + Sync {
    async fn overwrite<O: DenseInstance<DType = T>>(&self, other: O) -> TCResult<()>;

    async fn overwrite_value(&self, value: T) -> TCResult<()>;

    async fn write_value(&self, coord: Coord, value: T) -> TCResult<()>;
}

pub struct DenseTensor<FE, A> {
    accessor: A,
    phantom: PhantomData<FE>,
}

impl<FE, A: Clone> Clone for DenseTensor<FE, A> {
    fn clone(&self) -> Self {
        Self {
            accessor: self.accessor.clone(),
            phantom: self.phantom,
        }
    }
}

impl<FE, A> DenseTensor<FE, A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<FE, T: CDatatype> DenseTensor<FE, DenseAccess<FE, T>> {
    pub fn from_access<A: Into<DenseAccess<FE, T>>>(accessor: A) -> Self {
        Self {
            accessor: accessor.into(),
            phantom: PhantomData,
        }
    }
}

impl<FE: ThreadSafe, A: TensorInstance> TensorInstance for DenseTensor<FE, A> {
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<FE, L, R, T> TensorBoolean<DenseTensor<FE, R>> for DenseTensor<FE, L>
where
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    L: DenseInstance<DType = T> + Into<DenseAccess<FE, T>> + fmt::Debug,
    R: DenseInstance<DType = T> + Into<DenseAccess<FE, T>> + fmt::Debug,
    T: CDatatype + DType + fmt::Debug,
    DenseAccessCast<FE>: From<DenseAccess<FE, T>>,
    DenseTensor<FE, R>: fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
    Self: fmt::Debug,
{
    type Combine = DenseTensor<FE, DenseCompare<FE, u8>>;
    type LeftCombine = DenseTensor<FE, DenseCompare<FE, u8>>;

    fn and(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            Block::and,
            |l, r| bool_u8(l.and(r)),
        )
        .map(DenseTensor::from)
    }

    fn or(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            Block::or,
            |l, r| bool_u8(l.or(r)),
        )
        .map(DenseTensor::from)
    }

    fn xor(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            Block::xor,
            |l, r| bool_u8(l.xor(r)),
        )
        .map(DenseTensor::from)
    }
}

impl<FE, A> TensorBooleanConst for DenseTensor<FE, A>
where
    FE: ThreadSafe,
    A: DenseInstance + Into<DenseAccess<FE, A::DType>>,
    DenseAccessCast<FE>: From<DenseAccess<FE, A::DType>>,
{
    type Combine = DenseTensor<FE, DenseCompareConst<FE, u8>>;

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

impl<FE, L, R, T> TensorCompare<DenseTensor<FE, R>> for DenseTensor<FE, L>
where
    FE: ThreadSafe,
    L: DenseInstance<DType = T> + Into<DenseAccessCast<FE>>,
    R: DenseInstance<DType = T> + Into<DenseAccessCast<FE>>,
    T: CDatatype + DType,
{
    type Compare = DenseTensor<FE, DenseCompare<FE, u8>>;

    fn eq(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::eq, |l, r| {
            bool_u8(l.eq(&r))
        })
        .map(DenseTensor::from)
    }

    fn gt(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::gt, |l, r| {
            bool_u8(l.gt(&r))
        })
        .map(DenseTensor::from)
    }

    fn ge(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::ge, |l, r| {
            bool_u8(l.ge(&r))
        })
        .map(DenseTensor::from)
    }

    fn lt(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::lt, |l, r| {
            bool_u8(l.lt(&r))
        })
        .map(DenseTensor::from)
    }

    fn le(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::le, |l, r| {
            bool_u8(l.le(&r))
        })
        .map(DenseTensor::from)
    }

    fn ne(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::ne, |l, r| {
            bool_u8(l.ne(&r))
        })
        .map(DenseTensor::from)
    }
}

impl<FE, A> TensorCompareConst for DenseTensor<FE, A>
where
    FE: ThreadSafe,
    A: DenseInstance + Into<DenseAccessCast<FE>>,
{
    type Compare = DenseTensor<FE, DenseCompareConst<FE, u8>>;

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

impl<FE, A> TensorConvert for DenseTensor<FE, A>
where
    FE: ThreadSafe,
    A: DenseInstance + Clone,
    A::Block: NDArrayTransform,
    <A::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = A::DType> + NDArrayTransform + Into<Array<A::DType>>,
{
    type Dense = Self;
    type Sparse = SparseTensor<FE, SparseDense<A>>;

    fn into_dense(self) -> Self::Dense {
        self
    }

    fn into_sparse(self) -> Self::Sparse {
        SparseDense::new(self.accessor).into()
    }
}

impl<FE: ThreadSafe, A: DenseInstance> TensorDiagonal for DenseTensor<FE, A> {
    type Diagonal = DenseTensor<FE, DenseDiagonal<A>>;

    fn diagonal(self) -> TCResult<Self::Diagonal> {
        DenseDiagonal::new(self.accessor).map(DenseTensor::from)
    }
}

impl<FE, L, R, T> TensorMath<DenseTensor<FE, R>> for DenseTensor<FE, L>
where
    FE: ThreadSafe,
    L: DenseInstance<DType = T>,
    R: DenseInstance<DType = T>,
    T: CDatatype + DType,
{
    type Combine = DenseTensor<FE, DenseCombine<L, R, T>>;
    type LeftCombine = DenseTensor<FE, DenseCombine<L, R, T>>;

    fn add(self, other: DenseTensor<FE, R>) -> TCResult<Self::Combine> {
        fn add<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.add(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, add, |l, r| l + r).map(DenseTensor::from)
    }

    fn div(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn div<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.div(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, div, |l, r| l / r).map(DenseTensor::from)
    }

    fn log(self, base: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn log<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            let right = right.cast()?;
            left.log(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, base.accessor, log, |l: T, r: T| {
            T::from_float(l.to_float().log(r.to_float()))
        })
        .map(DenseTensor::from)
    }

    fn mul(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn mul<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.mul(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, mul, |l, r| l * r).map(DenseTensor::from)
    }

    fn pow(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn pow<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            let right = right.cast()?;
            left.pow(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, pow, |l: T, r: T| {
            T::from_float(l.to_float().pow(r.to_float()))
        })
        .map(DenseTensor::from)
    }

    fn sub(self, other: DenseTensor<FE, R>) -> TCResult<Self::Combine> {
        fn sub<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.sub(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, sub, |l, r| l - r).map(DenseTensor::from)
    }
}

impl<FE: ThreadSafe, A: DenseInstance> TensorMathConst for DenseTensor<FE, A>
where
    Number: CastInto<A::DType>,
{
    type Combine = DenseTensor<FE, DenseConst<A, A::DType>>;

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

        let accessor = DenseConst::new(
            self.accessor,
            n,
            |block, n| block.div_scalar(n).map(Array::from).map_err(TCError::from),
            |l, r| l / r,
        );

        Ok(accessor.into())
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
impl<FE, A> TensorReduce for DenseTensor<FE, A>
where
    FE: DenseCacheFile + AsType<Buffer<A::DType>> + AsType<Node>,
    A: DenseInstance + TensorPermitRead + Into<DenseAccess<FE, A::DType>>,
    A::DType: fmt::Debug,
    Buffer<A::DType>: de::FromStream<Context = ()>,
    Number: From<A::DType> + CastInto<A::DType>,
{
    type Reduce = DenseTensor<FE, DenseReduce<DenseAccess<FE, A::DType>, A::DType>>;

    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let mut blocks = self.accessor.read_blocks().await?;

        while let Some(block) = blocks.try_next().await? {
            if !block.all()? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let mut blocks = self.accessor.read_blocks().await?;

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
        let blocks = self.accessor.read_blocks().await?;
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
        let blocks = self.accessor.read_blocks().await?;
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
        let blocks = self.accessor.read_blocks().await?;

        let product = blocks
            .map(|result| result.and_then(|block| block.product().map_err(TCError::from)))
            .try_fold(A::DType::one(), |product, block_product| {
                future::ready(Ok(product * block_product))
            })
            .await?;

        Ok(product.into())
    }

    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        DenseReduce::sum(self.accessor.into(), axes, keepdims).map(DenseTensor::from)
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let blocks = self.accessor.read_blocks().await?;

        let sum = blocks
            .map(|result| result.and_then(|block| block.sum().map_err(TCError::from)))
            .try_fold(A::DType::zero(), |sum, block_sum| {
                future::ready(Ok(sum + block_sum))
            })
            .await?;

        Ok(sum.into())
    }
}

impl<FE: ThreadSafe, A: DenseInstance> TensorTransform for DenseTensor<FE, A> {
    type Broadcast = DenseTensor<FE, DenseBroadcast<A>>;
    type Expand = DenseTensor<FE, DenseExpand<A>>;
    type Reshape = DenseTensor<FE, DenseReshape<A>>;
    type Slice = DenseTensor<FE, DenseSlice<A>>;
    type Transpose = DenseTensor<FE, DenseTranspose<A>>;

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

impl<FE: ThreadSafe, A: DenseInstance> TensorUnary for DenseTensor<FE, A> {
    type Unary = DenseTensor<FE, DenseUnary<A, A::DType>>;

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

impl<FE, A> TensorUnaryBoolean for DenseTensor<FE, A>
where
    FE: DenseCacheFile,
    A: DenseInstance + Into<DenseAccessCast<FE>>,
{
    type Unary = DenseTensor<FE, DenseUnaryCast<FE, u8>>;

    fn not(self) -> TCResult<Self::Unary> {
        Ok(DenseUnaryCast::not(self.accessor).into())
    }
}

impl<FE, A> From<A> for DenseTensor<FE, A> {
    fn from(accessor: A) -> Self {
        Self {
            accessor,
            phantom: PhantomData,
        }
    }
}

impl<FE, A: fmt::Debug> fmt::Debug for DenseTensor<FE, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.accessor.fmt(f)
    }
}

#[inline]
pub fn dense_from<FE, A, T>(tensor: DenseTensor<FE, A>) -> DenseTensor<FE, DenseAccess<FE, T>>
where
    A: Into<DenseAccess<FE, T>>,
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
