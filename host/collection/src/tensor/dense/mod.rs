use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use freqfs::FileLoad;
use futures::stream::Stream;
use ha_ndarray::{Array, Buffer, CDatatype, NDArrayRead, NDArrayTransform, NDArrayWrite};
use safecast::AsType;

use tc_error::*;
use tc_value::{DType, NumberType};

use access::{DenseBroadcast, DenseReshape, DenseSlice, DenseTranspose};

mod access;
mod base;
mod stream;

use super::{offset_of, Axes, Coord, Range, Shape, TensorInstance, TensorTransform};

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

impl<A> TensorTransform for DenseTensor<A>
where
    A: DenseInstance,
{
    type Broadcast = DenseTensor<DenseBroadcast<A>>;
    type Expand = DenseTensor<DenseReshape<A>>;
    type Reshape = DenseTensor<DenseReshape<A>>;
    type Slice = DenseTensor<DenseSlice<A>>;
    type Transpose = DenseTensor<DenseTranspose<A>>;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        DenseBroadcast::new(self.accessor, shape).map(DenseTensor::from)
    }

    fn expand(self, mut axes: Axes) -> TCResult<Self::Expand> {
        let mut shape = self.shape().to_vec();

        axes.sort();

        for x in axes.into_iter().rev() {
            shape.insert(x, 1);
        }

        DenseReshape::new(self.accessor, shape.into()).map(DenseTensor::from)
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
