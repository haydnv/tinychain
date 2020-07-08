use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use crate::error;
use crate::state::file::File;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, TCType, Value};

use super::base::*;
use super::chunk::*;

const BLOCK_SIZE: usize = 1_000_000;
const ERR_CORRUPT: &str = "BlockTensor corrupted! Please restart Tinychain and file a bug report";

#[async_trait]
pub trait BlockTensorView: TensorView + Slice {
    async fn as_dtype(&self, txn: &Arc<Txn>, dtype: TCType) -> TCResult<BlockTensor>;

    async fn copy(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

    async fn sum(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<BlockTensor>;

    async fn product(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<BlockTensor>;

    async fn add<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: T) -> TCResult<BlockTensor>;

    async fn multiply<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: T)
        -> TCResult<BlockTensor>;

    async fn subtract<T: BlockTensorView>(
        &self,
        txn: &Arc<Txn>,
        other: &T,
    ) -> TCResult<BlockTensor>;

    async fn equals<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn and<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn or<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn xor<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn not(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

    async fn blocks(&self, txn_id: &Arc<TxnId>, len: usize) -> TCStream<Chunk>;
}

pub struct BlockTensor {
    dtype: TCType,
    shape: Shape,
    size: u64,
    ndim: usize,
    file: Arc<File>,
    per_block: usize,
    coord_index: Vec<u64>,
}

impl BlockTensor {
    async fn zeros(txn: Arc<Txn>, shape: Shape, dtype: TCType) -> TCResult<BlockTensor> {
        if !dtype.is_numeric() {
            return Err(error::bad_request("Tensor does not support", dtype));
        }

        let per_block = BLOCK_SIZE / dtype.size().unwrap();
        let size = shape.size();

        let blocks =
            (0..(size / per_block as u64)).map(move |_| ChunkData::new(&dtype, per_block).unwrap());
        let trailing_len = (size % (per_block as u64)) as usize;
        let blocks: TCStream<ChunkData> = if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(ChunkData::new(&dtype, trailing_len).unwrap()));
            Box::pin(stream::iter(blocks))
        } else {
            Box::pin(stream::iter(blocks))
        };
        BlockTensor::from_blocks(txn, shape, dtype, blocks, per_block).await
    }

    async fn from_blocks(
        txn: Arc<Txn>,
        shape: Shape,
        dtype: TCType,
        mut blocks: TCStream<ChunkData>,
        per_block: usize,
    ) -> TCResult<BlockTensor> {
        let file = txn
            .context()
            .create_file(txn.id().clone(), "block_tensor".parse()?)
            .await?;

        let size = shape.size();
        let ndim = shape.len();
        let mut i: u64 = 0;
        while let Some(block) = blocks.next().await {
            file.clone()
                .create_block(txn.id(), i.into(), block.into())
                .await?;
            i += 1;
        }

        let coord_index = (0..ndim)
            .map(|axis| shape[axis + 1..].iter().product())
            .collect();

        Ok(BlockTensor {
            dtype,
            shape,
            size,
            ndim,
            file,
            per_block,
            coord_index,
        })
    }

    async fn get_chunk(&self, txn_id: &TxnId, chunk_id: u64) -> TCResult<Chunk> {
        if let Some(block) = self.file.get_block(txn_id, &chunk_id.into()).await? {
            Chunk::try_from(block, self.dtype).await
        } else {
            Err(error::internal(ERR_CORRUPT))
        }
    }

    async fn write<T: TensorView + Broadcast>(
        &self,
        txn_id: &TxnId,
        coord: &Index,
        value: T,
    ) -> TCResult<()> {
        if !self.shape.contains(coord) {
            return Err(error::bad_request(
                &format!("Tensor with shape {} does not contain", self.shape),
                coord,
            ));
        }

        if self.shape.selection_shape(coord).is_empty() {
            if value.size() != 1 {
                return Err(error::bad_request(
                    "Cannot assign to Tensor index using value with shape",
                    value.shape(),
                ));
            }

            let index: u64 = coord
                .clone()
                .to_coord()
                .iter()
                .zip(self.coord_index.iter())
                .map(|(c, i)| c * i)
                .sum();
            let block_id = index / (self.per_block as u64);
            let offset = index % (self.per_block as u64);

            let mut chunk = self.get_chunk(txn_id, block_id).await?.upgrade().await?;
            chunk
                .data()
                .set(offset as usize, value.at(txn_id, &[]).await?)?;
            chunk.sync().await
        } else {
            Err(error::not_implemented())
        }
    }
}

#[async_trait]
impl TensorView for BlockTensor {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.size
    }

    async fn all(&self, _txn_id: &TxnId) -> TCResult<bool> {
        panic!("NOT IMPLEMENTED")
    }

    async fn any(&self, _txn_id: &TxnId) -> TCResult<bool> {
        panic!("NOT IMPLEMENTED")
    }

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Value> {
        let index: u64 = coord
            .iter()
            .zip(self.shape.to_vec().iter())
            .map(|(c, i)| c * i)
            .sum();
        let block_id = index / (self.per_block as u64);
        let offset = index % (self.per_block as u64);
        let chunk = self.get_chunk(txn_id, block_id).await?;
        Ok(chunk.data().get(offset as usize))
    }
}

pub struct DenseRebase<T: Rebase + 'static> {
    source: T,
}

#[async_trait]
impl<T: Rebase> TensorView for DenseRebase<T> {
    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.all(txn_id).await
    }

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.any(txn_id).await
    }

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Value> {
        self.source.at(txn_id, coord).await
    }
}

type DenseBroadcast<T> = DenseRebase<TensorBroadcast<T>>;
type DenseExpansion<T> = DenseRebase<Expansion<T>>;
type DensePermutation<T> = DenseRebase<Permutation<T>>;
type DenseTensorSlice<T> = DenseRebase<TensorSlice<T>>;
