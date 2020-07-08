use std::iter;
use std::marker::PhantomData;
use std::sync::Arc;

use arrayfire as af;
use async_trait::async_trait;
use futures::future;
use futures::stream::{self, StreamExt};
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
pub trait BlockTensorView<'a>: TensorView<'a> {
    async fn as_dtype(&'a self, _txn: &'a Arc<Txn>, _dtype: TCType) -> TCResult<BlockTensor<'a>> {
        Err(error::not_implemented())
    }

    async fn copy<'b>(&'a self, _txn: &'a Arc<Txn>) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn sum<'b>(
        &'a self,
        _txn: &'a Arc<Txn>,
        _axis: Option<usize>,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn product<'b>(
        &'a self,
        _txn: &'a Arc<Txn>,
        _axis: Option<usize>,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn add<'b, T: BlockTensorView<'a>>(
        &'a self,
        _txn: &Arc<Txn>,
        _other: &'a T,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn multiply<'b, T: BlockTensorView<'a>>(
        &'a self,
        _txn: &'a Arc<Txn>,
        _other: &'a T,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn subtract<'b, T: BlockTensorView<'a>>(
        &'a self,
        _txn: &'a Arc<Txn>,
        _other: &'a T,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn equals<'b, T: BlockTensorView<'a>>(
        &'a self,
        _txn: &'a Arc<Txn>,
        _other: &'a T,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn and<'b, T: BlockTensorView<'a>>(
        &'a self,
        _txn: &'a Arc<Txn>,
        _other: &'a T,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn or<'b, T: BlockTensorView<'a>>(
        &'a self,
        _txn: &'a Arc<Txn>,
        _other: &'a T,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn xor<'b, T: BlockTensorView<'a>>(
        &'a self,
        _txn: &'a Arc<Txn>,
        _other: &'a T,
    ) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    async fn not<'b>(&'a self, _txn: &'a Arc<Txn>) -> TCResult<BlockTensor<'b>> {
        Err(error::not_implemented())
    }

    fn blocks(&'a self, _txn_id: &TxnId, _len: usize) -> TCStream<Chunk> {
        Box::pin(stream::empty())
    }

    fn into_blocks(self, _txn_id: TxnId, _len: usize) -> TCStream<Chunk> {
        Box::pin(stream::empty())
    }
}

impl<'a> Slice<'a> for BlockTensor<'a> {
    type Slice = TensorSlice<'a, BlockTensor<'a>>;

    fn slice(&'a self, index: Index) -> TCResult<Self::Slice> {
        TensorSlice::new(self, index)
    }
}

#[async_trait]
impl<'a> BlockTensorView<'a> for TensorSlice<'a, BlockTensor<'a>> {}

pub struct BlockTensor<'a> {
    dtype: TCType,
    shape: Shape,
    size: u64,
    ndim: usize,
    file: Arc<File>,
    per_block: usize,
    coord_index: Vec<u64>,
    phantom: PhantomData<&'a Shape>,
}

impl<'a> BlockTensor<'a> {
    async fn zeros(txn: Arc<Txn>, shape: Shape, dtype: TCType) -> TCResult<BlockTensor<'a>> {
        if !dtype.is_numeric() {
            return Err(error::bad_request("Tensor does not support", dtype));
        }

        let per_block = BLOCK_SIZE / dtype.size().unwrap();
        let size = shape.size();

        let blocks =
            (0..(size / per_block as u64)).map(move |_| ChunkData::new(dtype, per_block).unwrap());
        let trailing_len = (size % (per_block as u64)) as usize;
        let blocks: TCStream<ChunkData> = if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(ChunkData::new(dtype, trailing_len).unwrap()));
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
    ) -> TCResult<BlockTensor<'a>> {
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
            phantom: PhantomData,
        })
    }

    async fn get_chunk(&self, txn_id: &TxnId, chunk_id: u64) -> TCResult<Chunk> {
        if let Some(block) = self.file.get_block(txn_id, &chunk_id.into()).await? {
            Chunk::try_from(block, self.dtype).await
        } else {
            Err(error::internal(ERR_CORRUPT))
        }
    }

    async fn write_dense<T: BlockTensorView<'a> + Broadcast<'a> + Slice<'a> + 'static>(
        &self,
        txn_id: TxnId,
        index: &Index,
        value: T,
    ) -> TCResult<()> {
        if !self.shape.contains(index) {
            return Err(error::bad_request(
                &format!("Tensor with shape {} does not contain", self.shape),
                index,
            ));
        }

        let value = value.broadcast(self.shape.selection(index))?;
        let block_size = BLOCK_SIZE / (8 * value.ndim()); // how many coordinates take up one block
        let ndim = value.ndim();

        let coord_index = af::Array::new(
            &self.coord_index,
            af::Dim4::new(&[self.ndim as u64, 1, 1, 1]),
        );
        let per_block = self.per_block as u64;

        value
            .into_blocks(txn_id.clone(), block_size)
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

                        let mut chunk = self.get_chunk(&txn_id, chunk_id).await?.upgrade().await?;
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
}

#[async_trait]
impl<'a, T: BlockTensorView<'a> + Slice<'a>> BlockTensorView<'a> for TensorBroadcast<'a, T> {}

#[async_trait]
impl<'a> TensorView<'a> for BlockTensor<'a> {
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
