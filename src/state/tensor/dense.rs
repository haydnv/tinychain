use std::iter;
use std::pin::Pin;
use std::sync::Arc;

use arrayfire as af;
use async_trait::async_trait;
use futures::future;
use futures::stream::{self, FuturesOrdered, Stream, StreamExt};
use itertools::Itertools;

use crate::error;
use crate::state::file::File;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, TCType, Value};

use super::base::*;
use super::chunk::*;
use super::index::*;

const BLOCK_SIZE: usize = 1_000_000;
const ERR_CORRUPT: &str = "BlockTensor corrupted! Please restart Tinychain and file a bug report";

#[async_trait]
pub trait BlockTensorView: TensorView {
    async fn as_dtype(&self, _txn: &Arc<Txn>, _dtype: TCType) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn copy(self: Arc<Self>, _txn: Arc<Txn>) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn sum(&self, _txn: &Arc<Txn>, _axis: Option<usize>) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn product(&self, _txn: &Arc<Txn>, _axis: Option<usize>) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn add<T: BlockTensorView>(&self, _txn: &Arc<Txn>, _other: &T) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn multiply<T: BlockTensorView>(
        &self,
        _txn: &Arc<Txn>,
        _other: &T,
    ) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn subtract<T: BlockTensorView>(
        &self,
        _txn: &Arc<Txn>,
        _other: &T,
    ) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn equals<'b, T: BlockTensorView>(
        &self,
        _txn: &Arc<Txn>,
        _other: &T,
    ) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn and<T: BlockTensorView>(&self, _txn: &Arc<Txn>, _other: &T) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn or<T: BlockTensorView>(&self, _txn: &Arc<Txn>, _other: &T) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn xor<T: BlockTensorView>(&self, _txn: &Arc<Txn>, _other: &T) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }

    async fn not(&self, _txn: &Arc<Txn>) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }
}

impl Slice for BlockTensor {
    type Slice = TensorSlice<BlockTensor>;

    fn slice(self: Arc<Self>, index: Index) -> TCResult<Arc<Self::Slice>> {
        Ok(Arc::new(TensorSlice::new(self, index)?))
    }
}

#[async_trait]
impl BlockTensorView for TensorSlice<BlockTensor> {}

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
    async fn constant(txn: Arc<Txn>, shape: Shape, value: Value) -> TCResult<BlockTensor> {
        if !value.dtype().is_numeric() {
            return Err(error::bad_request("Tensor does not support", value.dtype()));
        }

        let per_block = BLOCK_SIZE / value.dtype().size().unwrap();
        let size = shape.size();

        let value_clone = value.clone();
        let blocks = (0..(size / per_block as u64))
            .map(move |_| ChunkData::constant(value_clone.clone(), per_block).unwrap());
        let trailing_len = (size % (per_block as u64)) as usize;
        let blocks: TCStream<ChunkData> = if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(
                ChunkData::constant(value.clone(), trailing_len).unwrap(),
            ));
            Box::pin(stream::iter(blocks))
        } else {
            Box::pin(stream::iter(blocks))
        };
        BlockTensor::from_blocks(txn, shape, value.dtype(), blocks, per_block).await
    }

    async fn from_blocks(
        txn: Arc<Txn>,
        shape: Shape,
        dtype: TCType,
        mut blocks: Pin<Box<dyn Stream<Item = ChunkData> + Send>>,
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

    async fn write_dense(
        self: Arc<Self>,
        txn_id: TxnId,
        index: &Index,
        value: Arc<BlockTensor>,
    ) -> TCResult<()> {
        if !self.shape.contains(index) {
            return Err(error::bad_request(
                &format!("Tensor with shape {} does not contain", self.shape),
                index,
            ));
        }

        if self.dtype != value.dtype() {
            return Err(error::bad_request(
                &format!("Cannot write to Tensor of type {} with", self.dtype),
                value.dtype(),
            ));
        }

        let selection_shape = self.shape.selection(index);
        if &selection_shape != value.shape() {
            return Err(error::bad_request(
                &format!(
                    "Cannot write a Tensor of shape {} to a Tensor slice of shape",
                    value.shape()
                ),
                selection_shape,
            ));
        }

        let block_size = value.per_block;
        let ndim = value.ndim();

        let coord_index = af::Array::new(
            &self.coord_index,
            af::Dim4::new(&[self.ndim as u64, 1, 1, 1]),
        );
        let per_block = self.per_block as u64;

        value
            .blocks(txn_id.clone())
            .zip(stream::iter(&index.affected().chunks(block_size)))
            .then(|(values, coords)| {
                let coords: Vec<u64> = coords.flatten().collect();
                let num_coords = coords.len() / ndim;
                let af_coords_dim = af::Dim4::new(&[num_coords as u64, ndim as u64, 1, 1]);
                let af_coords = af::Array::new(&coords, af_coords_dim)
                    * af::tile(&coord_index.copy(), af_coords_dim);
                let af_coords = af::sum(&af_coords, 1);
                let af_per_block =
                    af::constant(per_block, af::Dim4::new(&[1, num_coords as u64, 1, 1]));
                let af_offsets = af_coords.copy() % af_per_block.copy();
                let af_indices = af_coords / af_per_block;
                let af_chunk_ids = af::set_unique(&af_indices, true);

                let mut chunk_ids: Vec<u64> = Vec::with_capacity(af_chunk_ids.elements());
                af_chunk_ids.host(&mut chunk_ids);

                let this = self.clone();
                let txn_id = txn_id.clone();
                async move {
                    let mut i = 0.0f64;
                    for chunk_id in chunk_ids {
                        let num_to_update = af::sum_all(&af::eq(
                            &af_indices,
                            &af::constant(chunk_id, af_indices.dims()),
                            false,
                        ))
                        .0;
                        let block_offsets = af::index(
                            &af_offsets,
                            &[
                                af::Seq::new(chunk_id as f64, chunk_id as f64, 1.0f64),
                                af::Seq::new(i, (i + num_to_update) - 1.0f64, 1.0f64),
                            ],
                        );
                        let block_offsets = af::moddims(
                            &block_offsets,
                            af::Dim4::new(&[num_coords as u64, 1, 1, 1]),
                        );

                        let mut chunk = this.get_chunk(&txn_id, chunk_id).await?.upgrade().await?;
                        chunk.data().set(block_offsets, values.data())?;
                        chunk.sync().await?;
                        i += num_to_update;
                    }

                    Ok(())
                }
            })
            .take_while(|r| future::ready(r.is_ok()))
            .fold(Ok(()), |_, last| future::ready(last))
            .await
    }

    fn blocks(self: Arc<Self>, txn_id: TxnId) -> impl Stream<Item = Chunk> {
        let mut blocks = FuturesOrdered::new();
        for chunk_id in 0..(self.size / self.per_block as u64) {
            let txn_id = txn_id.clone();
            let this = self.clone();
            blocks.push(async move { this.get_chunk(&txn_id, chunk_id).await.unwrap() })
        }

        blocks
    }
}

#[async_trait]
impl TensorView for BlockTensor {
    fn dtype(&self) -> TCType {
        self.dtype
    }

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

    async fn at<'a>(&'a self, txn_id: &'a TxnId, coord: Vec<u64>) -> TCResult<Value> {
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

#[async_trait]
impl BlockTensorView for BlockTensor {
    async fn copy(self: Arc<Self>, txn: Arc<Txn>) -> TCResult<BlockTensor> {
        let blocks = self.clone().blocks(txn.id().clone());
        let blocks = blocks.map(|chunk| chunk.data().clone());
        BlockTensor::from_blocks(
            txn,
            self.shape().clone(),
            self.dtype(),
            Box::pin(blocks),
            self.per_block,
        )
        .await
    }
}

#[async_trait]
impl<T: BlockTensorView + Slice + 'static> BlockTensorView for TensorBroadcast<T> {
    async fn copy(self: Arc<Self>, txn: Arc<Txn>) -> TCResult<BlockTensor> {
        let dtype = self.source().dtype();
        let shape = self.shape().clone();
        let size = shape.size();
        let ndim = shape.len();
        let per_block = BLOCK_SIZE / self.dtype().size().unwrap();

        let source_coords = stream::iter(self.shape().all().affected())
            .map(|coord| self.invert_index(coord.into()));
        let mut value_stream = source_coords.then(|index| self.at(txn.id(), index.to_coord()));

        let file = txn
            .context()
            .create_file(txn.id().clone(), "block_tensor".parse()?)
            .await?;

        let mut chunk: Vec<Value> = Vec::with_capacity(per_block);
        let mut block_id = 0;
        while let Some(value) = value_stream.next().await {
            chunk.push(value?);

            if chunk.len() == per_block {
                let block = ChunkData::try_from_values(chunk.drain(..).collect(), dtype)?;
                file.clone()
                    .create_block(txn.id(), block_id.into(), block.into())
                    .await?;
                block_id += 1;
            }
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
}
