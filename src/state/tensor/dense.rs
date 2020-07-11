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
use crate::value::class::{ComplexType, FloatType, NumberType};
use crate::value::class::{Impl, NumberClass};
use crate::value::{Number, TCResult, TCStream};

use super::base::*;
use super::chunk::*;
use super::index::*;
use super::stream::{ValueChunkStream, ValueStream};

const BLOCK_SIZE: usize = 1_000_000;
const ERR_CORRUPT: &str = "BlockTensor corrupted! Please restart Tinychain and file a bug report";

#[async_trait]
pub trait BlockTensorView: TensorView + 'static {
    type ChunkStream: Stream<Item = TCResult<ChunkData>> + Send + Unpin;
    type ValueStream: Stream<Item = TCResult<Number>> + Send;

    fn chunk_stream(self: Arc<Self>, txn_id: TxnId) -> Self::ChunkStream;

    fn value_stream(self: Arc<Self>, txn_id: TxnId, index: Index) -> Self::ValueStream;
}

#[async_trait]
impl<T: BlockTensorView + Slice> TensorUnary for T {
    type Base = BlockTensor;
    type Dense = BlockTensor;

    async fn as_dtype(self: Arc<Self>, txn: Arc<Txn>, dtype: NumberType) -> TCResult<BlockTensor> {
        let shape = self.shape().clone();
        let per_block = per_block(dtype);
        let source = self
            .chunk_stream(txn.id().clone())
            .map(move |data| data.and_then(|d| d.into_type(dtype.clone())));
        let values = ValueStream::new(source);
        let chunks = ValueChunkStream::new(values, dtype, per_block);
        BlockTensor::from_blocks(txn, shape, dtype, Box::pin(chunks)).await
    }

    async fn copy(self: Arc<Self>, txn: Arc<Txn>) -> TCResult<Self::Base> {
        let shape = self.shape().clone();
        let dtype = self.dtype();
        let blocks = Box::pin(self.chunk_stream(txn.id().clone()));
        BlockTensor::from_blocks(txn, shape, dtype, blocks).await
    }

    async fn abs(self: Arc<Self>, txn: Arc<Txn>) -> TCResult<Self::Base> {
        let shape = self.shape().clone();
        let txn_id = txn.id().clone();

        use NumberType::*;
        let (chunks, dtype): (
            Pin<Box<dyn Stream<Item = TCResult<ChunkData>> + Send>>,
            NumberType,
        ) = match self.dtype() {
            Bool => (Box::pin(self.chunk_stream(txn_id)), Bool),
            Complex(c) => match c {
                ComplexType::C32 => {
                    let dtype = FloatType::F32.into();
                    let source = self.chunk_stream(txn_id).map(|d| d?.abs());
                    let per_block = per_block(dtype);
                    let values = ValueStream::new(source);
                    let chunks = ValueChunkStream::new(values, dtype, per_block);
                    (Box::pin(chunks), dtype)
                }
                ComplexType::C64 => {
                    let dtype = FloatType::F64.into();
                    let source = self.chunk_stream(txn_id).map(|d| d?.abs());
                    let per_block = per_block(dtype);
                    let values = ValueStream::new(source);
                    let chunks = ValueChunkStream::new(values, dtype, per_block);
                    (Box::pin(chunks), dtype)
                }
            },
            Float(f) => (
                Box::pin(self.chunk_stream(txn_id).map(|d| d?.abs())),
                f.into(),
            ),
            Int(i) => (
                Box::pin(self.chunk_stream(txn_id).map(|d| d?.abs())),
                i.into(),
            ),
            UInt(u) => (Box::pin(self.chunk_stream(txn_id)), u.into()),
        };

        BlockTensor::from_blocks(txn, shape, dtype, chunks).await
    }

    async fn sum(self: Arc<Self>, txn: Arc<Txn>, axis: usize) -> TCResult<Self::Base> {
        if axis >= self.ndim() {
            return Err(error::bad_request("Axis out of range", axis));
        }

        let mut shape = self.shape().clone();
        shape.remove(axis);
        let _summed = BlockTensor::constant(txn, shape, self.dtype().zero()).await?;

        Err(error::not_implemented())
    }

    async fn sum_all(self: Arc<Self>, txn_id: TxnId) -> TCResult<Number> {
        let mut sum = self.dtype().zero();
        let mut chunks = self.chunk_stream(txn_id);
        while let Some(chunk) = chunks.next().await {
            sum = sum + chunk?.product();
        }

        Ok(sum)
    }

    async fn product(self: Arc<Self>, _txn: Arc<Txn>, _axis: usize) -> TCResult<Self::Base> {
        Err(error::not_implemented())
    }

    async fn product_all(self: Arc<Self>, txn_id: TxnId) -> TCResult<Number> {
        let mut product = self.dtype().one();
        let mut chunks = self.chunk_stream(txn_id);
        while let Some(chunk) = chunks.next().await {
            product = product * chunk?.product();
        }

        Ok(product)
    }

    async fn not(self: Arc<Self>, _txn: &Arc<Txn>) -> TCResult<Self::Dense> {
        Err(error::not_implemented())
    }
}

pub struct BlockTensor {
    dtype: NumberType,
    shape: Shape,
    size: u64,
    ndim: usize,
    file: Arc<File>,
    per_block: usize,
    coord_index: Vec<u64>,
}

impl BlockTensor {
    async fn constant(txn: Arc<Txn>, shape: Shape, value: Number) -> TCResult<BlockTensor> {
        let per_block = BLOCK_SIZE / value.class().size();
        let size = shape.size();

        let value_clone = value.clone();
        let blocks = (0..(size / per_block as u64))
            .map(move |_| Ok(ChunkData::constant(value_clone.clone(), per_block)));
        let trailing_len = (size % (per_block as u64)) as usize;
        let blocks: TCStream<TCResult<ChunkData>> = if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(Ok(ChunkData::constant(
                value.clone(),
                trailing_len,
            ))));
            Box::pin(stream::iter(blocks))
        } else {
            Box::pin(stream::iter(blocks))
        };
        BlockTensor::from_blocks(txn, shape, value.class(), blocks).await
    }

    async fn from_blocks(
        txn: Arc<Txn>,
        shape: Shape,
        dtype: NumberType,
        mut blocks: Pin<Box<dyn Stream<Item = TCResult<ChunkData>> + Send>>,
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
                .create_block(txn.id(), i.into(), block?.into())
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
            per_block: per_block(dtype),
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

    async fn assign<T: BlockTensorView>(
        self: Arc<Self>,
        txn_id: TxnId,
        index: &Index,
        value: Arc<T>,
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

        let ndim = value.ndim();

        let coord_index = af::Array::new(
            &self.coord_index,
            af::Dim4::new(&[self.ndim as u64, 1, 1, 1]),
        );
        let per_block = self.per_block as u64;

        value
            .chunk_stream(txn_id.clone())
            .zip(stream::iter(&index.affected().chunks(self.per_block)))
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
                    let values = values?;
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
                        chunk.data().set(block_offsets, &values)?;
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

    async fn at(self: Arc<Self>, txn_id: TxnId, coord: Vec<u64>) -> TCResult<Number> {
        let index: u64 = coord
            .iter()
            .zip(self.shape.to_vec().iter())
            .map(|(c, i)| c * i)
            .sum();
        let block_id = index / (self.per_block as u64);
        let offset = index % (self.per_block as u64);
        let chunk = self.get_chunk(&txn_id, block_id).await?;
        chunk
            .data()
            .get_one(offset as usize)
            .ok_or_else(|| error::internal(ERR_CORRUPT))
    }
}

impl TensorView for BlockTensor {
    fn dtype(&self) -> NumberType {
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
}

#[async_trait]
impl AnyAll for BlockTensor {
    async fn all(self: Arc<Self>, txn_id: TxnId) -> TCResult<bool> {
        let mut chunks = self.chunk_stream(txn_id);
        while let Some(chunk) = chunks.next().await {
            if !chunk?.all() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self: Arc<Self>, txn_id: TxnId) -> TCResult<bool> {
        let mut chunks = self.chunk_stream(txn_id);
        while let Some(chunk) = chunks.next().await {
            if !chunk?.any() {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

#[async_trait]
impl BlockTensorView for BlockTensor {
    type ChunkStream = Pin<Box<dyn Stream<Item = TCResult<ChunkData>> + Send>>;
    type ValueStream = Pin<Box<dyn Stream<Item = TCResult<Number>> + Send>>;

    fn chunk_stream(self: Arc<Self>, txn_id: TxnId) -> Self::ChunkStream {
        Box::pin(self.blocks(txn_id).map(|chunk| Ok(chunk.data().clone())))
    }

    fn value_stream(self: Arc<Self>, txn_id: TxnId, index: Index) -> Self::ValueStream {
        if index == self.shape().all() {
            return Box::pin(ValueStream::new(self.chunk_stream(txn_id)));
        }

        assert!(self.shape().contains(&index));
        let mut selected = FuturesOrdered::new();

        let ndim = index.ndim();

        let coord_index = af::Array::new(
            &self.coord_index,
            af::Dim4::new(&[self.ndim as u64, 1, 1, 1]),
        );
        let per_block = self.per_block as u64;

        for coords in &index.affected().chunks(self.per_block) {
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

            selected.push(async move {
                let mut i = 0.0f64;
                let mut values = vec![];
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
                    let block_offsets =
                        af::moddims(&block_offsets, af::Dim4::new(&[num_coords as u64, 1, 1, 1]));

                    match this.get_chunk(&txn_id, chunk_id).await {
                        Ok(chunk) => values.extend(chunk.data().get(block_offsets)),
                        Err(cause) => return stream::iter(vec![Err(cause)]),
                    }
                    i += num_to_update;
                }

                let values: Vec<TCResult<Number>> = values.drain(..).map(Ok).collect();
                stream::iter(values)
            });
        }

        Box::pin(selected.flatten())
    }
}

impl Slice for BlockTensor {
    type Slice = TensorSlice<BlockTensor>;

    fn slice(self: Arc<Self>, index: Index) -> TCResult<Arc<Self::Slice>> {
        Ok(Arc::new(TensorSlice::new(self, index)?))
    }
}

#[async_trait]
impl<T: Rebase + 'static> BlockTensorView for T
where
    <T as Rebase>::Source: BlockTensorView,
{
    type ChunkStream = ValueChunkStream<Self::ValueStream>;
    type ValueStream = <<T as Rebase>::Source as BlockTensorView>::ValueStream;

    fn chunk_stream(self: Arc<Self>, txn_id: TxnId) -> Self::ChunkStream {
        let dtype = self.source().dtype();
        let index = self.shape().all();
        ValueChunkStream::new(self.value_stream(txn_id, index), dtype, per_block(dtype))
    }

    fn value_stream(self: Arc<Self>, txn_id: TxnId, index: Index) -> Self::ValueStream {
        assert!(self.shape().contains(&index));
        self.source().value_stream(txn_id, self.invert_index(index))
    }
}

#[async_trait]
impl AnyAll for TensorSlice<BlockTensor> {
    async fn all(self: Arc<Self>, txn_id: TxnId) -> TCResult<bool> {
        let mut chunks = self.chunk_stream(txn_id);
        while let Some(chunk) = chunks.next().await {
            if !chunk?.all() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self: Arc<Self>, txn_id: TxnId) -> TCResult<bool> {
        let mut chunks = self.chunk_stream(txn_id);
        while let Some(chunk) = chunks.next().await {
            if !chunk?.any() {
                return Ok(true);
            }
        }

        Ok(false)
    }
}

fn per_block(dtype: NumberType) -> usize {
    BLOCK_SIZE / dtype.size()
}
