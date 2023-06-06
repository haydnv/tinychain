use std::fmt;
use std::ops::BitXor;
use std::pin::Pin;

use async_trait::async_trait;
use freqfs::FileLoad;
use futures::stream::Stream;
use ha_ndarray::{Array, Buffer, CDatatype, NDArrayRead, NDArrayTransform, NDArrayWrite};
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_value::{DType, Number, NumberType};

use crate::tensor::{TensorBoolean, TensorBooleanConst};

use super::{offset_of, Axes, Coord, Range, Shape, TensorInstance, TensorTransform};

use access::{
    DenseBroadcast, DenseCombine, DenseConst, DenseExpand, DenseReshape, DenseSlice, DenseTranspose,
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
pub struct DenseTensor<A> {
    accessor: A,
}

impl<A> DenseTensor<A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<A: TensorInstance> TensorInstance for DenseTensor<A> {
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<L, R, T> TensorBoolean<DenseTensor<R>> for DenseTensor<L>
where
    L: DenseInstance<DType = T> + fmt::Debug,
    R: DenseInstance<DType = T> + fmt::Debug,
    T: CDatatype + DType,
    DenseTensor<R>: fmt::Debug,
    Self: fmt::Debug,
{
    type Combine = DenseTensor<DenseCombine<L, R, T, u8>>;
    type LeftCombine = DenseTensor<DenseCombine<L, R, T, u8>>;

    fn and(self, other: DenseTensor<R>) -> TCResult<Self::LeftCombine> {
        fn and_block<T: CDatatype>(l: Array<T>, r: Array<T>) -> TCResult<Array<u8>> {
            ha_ndarray::NDArrayBoolean::and(l, r)
                .map(Array::from)
                .map_err(TCError::from)
        }

        fn and<T: CDatatype>(l: T, r: T) -> u8 {
            if l != T::zero() && r != T::zero() {
                1
            } else {
                0
            }
        }

        DenseCombine::new(self.accessor, other.accessor, and_block, and).map(DenseTensor::from)
    }

    fn or(self, other: DenseTensor<R>) -> TCResult<Self::Combine> {
        fn or_block<T: CDatatype>(l: Array<T>, r: Array<T>) -> TCResult<Array<u8>> {
            ha_ndarray::NDArrayBoolean::or(l, r)
                .map(Array::from)
                .map_err(TCError::from)
        }

        fn or<T: CDatatype>(l: T, r: T) -> u8 {
            if l != T::zero() || r != T::zero() {
                1
            } else {
                0
            }
        }

        DenseCombine::new(self.accessor, other.accessor, or_block, or).map(DenseTensor::from)
    }

    fn xor(self, other: DenseTensor<R>) -> TCResult<Self::Combine> {
        fn xor_block<T: CDatatype>(l: Array<T>, r: Array<T>) -> TCResult<Array<u8>> {
            ha_ndarray::NDArrayBoolean::xor(l, r)
                .map(Array::from)
                .map_err(TCError::from)
        }

        fn xor<T: CDatatype>(l: T, r: T) -> u8 {
            if (l != T::zero()).bitxor(r != T::zero()) {
                1
            } else {
                0
            }
        }

        DenseCombine::new(self.accessor, other.accessor, xor_block, xor).map(DenseTensor::from)
    }
}

impl<A: DenseInstance> TensorBooleanConst for DenseTensor<A>
where
    A::DType: CDatatype + CastFrom<Number>,
{
    type Combine = DenseTensor<DenseConst<A, A::DType, u8>>;
    type DenseCombine = DenseTensor<DenseConst<A, A::DType, u8>>;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        let other = other.cast_into();

        fn and_block<T: CDatatype>(l: Array<T>, r: T) -> TCResult<Array<u8>> {
            ha_ndarray::NDArrayBooleanConst::and_const(l, r)
                .map(Array::from)
                .map_err(TCError::from)
        }

        fn and<T: CDatatype>(l: T, r: T) -> u8 {
            if l != T::zero() && r != T::zero() {
                1
            } else {
                0
            }
        }

        Ok(DenseTensor {
            accessor: DenseConst::new(self.accessor, other, and_block, and),
        })
    }

    fn or_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        let other = other.cast_into();

        fn or_block<T: CDatatype>(l: Array<T>, r: T) -> TCResult<Array<u8>> {
            ha_ndarray::NDArrayBooleanConst::or_const(l, r)
                .map(Array::from)
                .map_err(TCError::from)
        }

        fn or<T: CDatatype>(l: T, r: T) -> u8 {
            if l != T::zero() || r != T::zero() {
                1
            } else {
                0
            }
        }

        Ok(DenseTensor {
            accessor: DenseConst::new(self.accessor, other, or_block, or),
        })
    }

    fn xor_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        let other = other.cast_into();

        fn xor_block<T: CDatatype>(l: Array<T>, r: T) -> TCResult<Array<u8>> {
            ha_ndarray::NDArrayBooleanConst::xor_const(l, r)
                .map(Array::from)
                .map_err(TCError::from)
        }

        fn xor<T: CDatatype>(l: T, r: T) -> u8 {
            if (l != T::zero()).bitxor(r != T::zero()) {
                1
            } else {
                0
            }
        }

        Ok(DenseTensor {
            accessor: DenseConst::new(self.accessor, other, xor_block, xor),
        })
    }
}

impl<A: DenseInstance> TensorTransform for DenseTensor<A> {
    type Broadcast = DenseTensor<DenseBroadcast<A>>;
    type Expand = DenseTensor<DenseExpand<A>>;
    type Reshape = DenseTensor<DenseReshape<A>>;
    type Slice = DenseTensor<DenseSlice<A>>;
    type Transpose = DenseTensor<DenseTranspose<A>>;

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

impl<A> From<A> for DenseTensor<A> {
    fn from(accessor: A) -> Self {
        Self { accessor }
    }
}
