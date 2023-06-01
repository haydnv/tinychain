use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::Stream;
use ha_ndarray::{Array, CDatatype, NDArrayRead, NDArrayTransform, NDArrayWrite};

use tc_error::TCResult;
use tc_value::DType;

mod access;
mod base;
mod stream;

use super::{offset_of, Coord, TensorInstance};

type BlockShape = ha_ndarray::Shape;
type BlockStream<Block> = Pin<Box<dyn Stream<Item = TCResult<Block>> + Send>>;

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
