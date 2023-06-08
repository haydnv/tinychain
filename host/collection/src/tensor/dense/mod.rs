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
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{DType, Number, NumberCollator, NumberType};
use tcgeneric::ThreadSafe;

use crate::tensor::sparse::Node;
use crate::tensor::{
    TensorBoolean, TensorBooleanConst, TensorCompare, TensorCompareConst, TensorDiagonal,
    TensorMath, TensorMathConst, TensorPermitRead, TensorReduce, TensorUnary, TensorUnaryBoolean,
};

use super::{offset_of, Axes, Coord, Range, Shape, TensorInstance, TensorTransform};

use access::*;

mod access;
mod base;
mod stream;

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
pub trait DenseInstance: TensorInstance + fmt::Debug + Send + Sync + 'static {
    type Block: NDArrayRead<DType = Self::DType> + NDArrayTransform + Into<Array<Self::DType>>;
    type DType: CDatatype + DType;

    fn block_size(&self) -> usize;

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block>;

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>>;

    // TODO: remove this generic implementation
    async fn read_value(&self, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;

        let offset = offset_of(coord, self.shape());
        let block_id = offset / self.block_size() as u64;
        let block_offset = (offset % self.block_size() as u64) as usize;

        let block = self.read_block(block_id).await?;
        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, self.block_size())?;
        let buffer = block.read(&queue)?;
        Ok(buffer.to_slice()?.as_ref()[block_offset])
    }
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

#[derive(Clone)]
pub struct DenseTensor<FE, A> {
    accessor: A,
    phantom: PhantomData<FE>,
}

impl<FE, A> DenseTensor<FE, A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<FE: Send + Sync + 'static, A: TensorInstance> TensorInstance for DenseTensor<FE, A> {
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<FE, L, R, T> TensorBoolean<DenseTensor<FE, R>> for DenseTensor<FE, L>
where
    FE: Send + Sync + 'static,
    L: DenseInstance<DType = T> + Into<DenseAccessCast<FE>> + fmt::Debug,
    R: DenseInstance<DType = T> + Into<DenseAccessCast<FE>> + fmt::Debug,
    T: CDatatype + DType,
    DenseTensor<FE, R>: fmt::Debug,
    Self: fmt::Debug,
{
    type Combine = DenseTensor<FE, DenseCompare<FE, u8>>;
    type LeftCombine = DenseTensor<FE, DenseCompare<FE, u8>>;

    fn and(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(self.accessor, other.accessor, Block::and).map(DenseTensor::from)
    }

    fn or(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(self.accessor, other.accessor, Block::or).map(DenseTensor::from)
    }

    fn xor(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(self.accessor, other.accessor, Block::xor).map(DenseTensor::from)
    }
}

impl<FE, A> TensorBooleanConst for DenseTensor<FE, A>
where
    FE: Send + Sync + 'static,
    A: DenseInstance + Into<DenseAccessCast<FE>>,
{
    type Combine = DenseTensor<FE, DenseCompareConst<FE, u8>>;
    type DenseCombine = DenseTensor<FE, DenseCompareConst<FE, u8>>;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::and_scalar).into())
    }

    fn or_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::or_scalar).into())
    }

    fn xor_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::xor_scalar).into())
    }
}

impl<FE, L, R, T> TensorCompare<DenseTensor<FE, R>> for DenseTensor<FE, L>
where
    FE: Send + Sync + 'static,
    L: DenseInstance<DType = T> + Into<DenseAccessCast<FE>>,
    R: DenseInstance<DType = T> + Into<DenseAccessCast<FE>>,
    T: CDatatype + DType,
{
    type Compare = DenseTensor<FE, DenseCompare<FE, u8>>;
    type Dense = DenseTensor<FE, DenseCompare<FE, u8>>;

    fn eq(self, other: DenseTensor<FE, R>) -> TCResult<Self::Dense> {
        DenseCompare::new(self.accessor, other.accessor, Block::eq).map(DenseTensor::from)
    }

    fn gt(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::gt).map(DenseTensor::from)
    }

    fn ge(self, other: DenseTensor<FE, R>) -> TCResult<Self::Dense> {
        DenseCompare::new(self.accessor, other.accessor, Block::ge).map(DenseTensor::from)
    }

    fn lt(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::lt).map(DenseTensor::from)
    }

    fn le(self, other: DenseTensor<FE, R>) -> TCResult<Self::Dense> {
        DenseCompare::new(self.accessor, other.accessor, Block::le).map(DenseTensor::from)
    }

    fn ne(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, Block::ne).map(DenseTensor::from)
    }
}

impl<FE, A> TensorCompareConst for DenseTensor<FE, A>
where
    FE: Send + Sync + 'static,
    A: DenseInstance + Into<DenseAccessCast<FE>>,
{
    type Compare = DenseTensor<FE, DenseCompareConst<FE, u8>>;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::eq_scalar).into())
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::gt_scalar).into())
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::ge_scalar).into())
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::lt_scalar).into())
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::le_scalar).into())
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, Block::ne_scalar).into())
    }
}

impl<FE: Send + Sync + 'static, A: DenseInstance> TensorDiagonal for DenseTensor<FE, A> {
    type Diagonal = DenseTensor<FE, DenseDiagonal<A>>;

    fn diagonal(self) -> TCResult<Self::Diagonal> {
        DenseDiagonal::new(self.accessor).map(DenseTensor::from)
    }
}

impl<FE, L, R, T> TensorMath<DenseTensor<FE, R>> for DenseTensor<FE, L>
where
    FE: Send + Sync + 'static,
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

        DenseCombine::new(self.accessor, other.accessor, add).map(DenseTensor::from)
    }

    fn div(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn div<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.div(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, div).map(DenseTensor::from)
    }

    fn log(self, base: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn log<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            let right = right.cast()?;
            left.log(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, base.accessor, log).map(DenseTensor::from)
    }

    fn mul(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn mul<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.mul(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, mul).map(DenseTensor::from)
    }

    fn pow(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn pow<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            let right = right.cast()?;
            left.pow(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, pow).map(DenseTensor::from)
    }

    fn sub(self, other: DenseTensor<FE, R>) -> TCResult<Self::Combine> {
        fn sub<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            left.sub(right).map(Array::from).map_err(TCError::from)
        }

        DenseCombine::new(self.accessor, other.accessor, sub).map(DenseTensor::from)
    }
}

impl<FE: Send + Sync + 'static, A: DenseInstance> TensorMathConst for DenseTensor<FE, A>
where
    Number: CastInto<A::DType>,
{
    type Combine = DenseTensor<FE, DenseConst<A, A::DType>>;
    type DenseCombine = DenseTensor<FE, DenseConst<A, A::DType>>;

    fn add_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(self.accessor, n, |block, n| {
            block.add_scalar(n).map(Array::from).map_err(TCError::from)
        });

        Ok(accessor.into())
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(self.accessor, n, |block, n| {
            block.div_scalar(n).map(Array::from).map_err(TCError::from)
        });

        Ok(accessor.into())
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        let n = base.cast_into();

        let accessor = DenseConst::new(self.accessor, n, |block, n| {
            block
                .log_scalar(n.to_float())
                .map(Array::from)
                .map_err(TCError::from)
        });

        Ok(accessor.into())
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(self.accessor, n, |block, n| {
            block.mul_scalar(n).map(Array::from).map_err(TCError::from)
        });

        Ok(accessor.into())
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(self.accessor, n, |block, n| {
            block
                .pow_scalar(n.to_float())
                .map(Array::from)
                .map_err(TCError::from)
        });

        Ok(accessor.into())
    }

    fn sub_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        let n = other.cast_into();

        let accessor = DenseConst::new(self.accessor, n, |block, n| {
            block.sub_scalar(n).map(Array::from).map_err(TCError::from)
        });

        Ok(accessor.into())
    }
}

#[async_trait]
impl<FE, A> TensorReduce for DenseTensor<FE, A>
where
    FE: DenseCacheFile + AsType<Buffer<A::DType>> + AsType<Node>,
    A: DenseInstance + TensorPermitRead + Into<DenseAccess<FE, A::DType>>,
    A::DType: fmt::Debug,
    Array<A::DType>: Clone,
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

impl<FE: Send + Sync + 'static, A: DenseInstance> TensorTransform for DenseTensor<FE, A> {
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

impl<FE: Send + Sync + 'static, A: DenseInstance> TensorUnary for DenseTensor<FE, A> {
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
