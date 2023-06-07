use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use freqfs::FileLoad;
use futures::stream::Stream;
use ha_ndarray::{Array, Buffer, CDatatype, NDArrayRead, NDArrayTransform, NDArrayWrite};
use safecast::AsType;

use tc_error::*;
use tc_value::{DType, Number, NumberType};

use crate::tensor::{
    TensorBoolean, TensorBooleanConst, TensorCompare, TensorCompareConst, TensorDiagonal,
};

use super::{offset_of, Axes, Coord, Range, Shape, TensorInstance, TensorTransform};

use crate::tensor::dense::access::DenseDiagonal;
use access::{
    ArrayCastSource, DenseBroadcast, DenseCastSource, DenseCompare, DenseCompareConst, DenseExpand,
    DenseReshape, DenseSlice, DenseTranspose,
};

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
    + Send
    + Sync
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
        + Send
        + Sync
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
    L: DenseInstance<DType = T> + Into<DenseCastSource<FE>> + fmt::Debug,
    R: DenseInstance<DType = T> + Into<DenseCastSource<FE>> + fmt::Debug,
    T: CDatatype + DType,
    DenseTensor<FE, R>: fmt::Debug,
    Self: fmt::Debug,
{
    type Combine = DenseTensor<FE, DenseCompare<FE, u8>>;
    type LeftCombine = DenseTensor<FE, DenseCompare<FE, u8>>;

    fn and(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::and)
            .map(DenseTensor::from)
    }

    fn or(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::or).map(DenseTensor::from)
    }

    fn xor(self, other: DenseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::xor)
            .map(DenseTensor::from)
    }
}

impl<FE, A> TensorBooleanConst for DenseTensor<FE, A>
where
    FE: Send + Sync + 'static,
    A: DenseInstance + Into<DenseCastSource<FE>>,
{
    type Combine = DenseTensor<FE, DenseCompareConst<FE, u8>>;
    type DenseCombine = DenseTensor<FE, DenseCompareConst<FE, u8>>;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::and_scalar).into())
    }

    fn or_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::or_scalar).into())
    }

    fn xor_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::xor_scalar).into())
    }
}

impl<FE, L, R, T> TensorCompare<DenseTensor<FE, R>> for DenseTensor<FE, L>
where
    FE: Send + Sync + 'static,
    L: DenseInstance<DType = T> + Into<DenseCastSource<FE>>,
    R: DenseInstance<DType = T> + Into<DenseCastSource<FE>>,
    T: CDatatype + DType,
{
    type Compare = DenseTensor<FE, DenseCompare<FE, u8>>;
    type Dense = DenseTensor<FE, DenseCompare<FE, u8>>;

    fn eq(self, other: DenseTensor<FE, R>) -> TCResult<Self::Dense> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::eq).map(DenseTensor::from)
    }

    fn gt(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::gt).map(DenseTensor::from)
    }

    fn ge(self, other: DenseTensor<FE, R>) -> TCResult<Self::Dense> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::ge).map(DenseTensor::from)
    }

    fn lt(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::lt).map(DenseTensor::from)
    }

    fn le(self, other: DenseTensor<FE, R>) -> TCResult<Self::Dense> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::le).map(DenseTensor::from)
    }

    fn ne(self, other: DenseTensor<FE, R>) -> TCResult<Self::Compare> {
        DenseCompare::new(self.accessor, other.accessor, ArrayCastSource::ne).map(DenseTensor::from)
    }
}

impl<FE, A> TensorCompareConst for DenseTensor<FE, A>
where
    FE: Send + Sync + 'static,
    A: DenseInstance + Into<DenseCastSource<FE>>,
{
    type Compare = DenseTensor<FE, DenseCompareConst<FE, u8>>;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::eq_scalar).into())
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::gt_scalar).into())
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::ge_scalar).into())
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::lt_scalar).into())
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::le_scalar).into())
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        Ok(DenseCompareConst::new(self.accessor, other, ArrayCastSource::ne_scalar).into())
    }
}

impl<FE: Send + Sync + 'static, A: DenseInstance> TensorDiagonal for DenseTensor<FE, A> {
    type Diagonal = DenseTensor<FE, DenseDiagonal<A>>;

    fn diagonal(self) -> TCResult<Self::Diagonal> {
        DenseDiagonal::new(self.accessor).map(DenseTensor::from)
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

impl<FE, A> From<A> for DenseTensor<FE, A> {
    fn from(accessor: A) -> Self {
        Self {
            accessor,
            phantom: PhantomData,
        }
    }
}
