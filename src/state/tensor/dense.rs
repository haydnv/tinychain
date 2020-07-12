use std::iter;
use std::pin::Pin;
use std::sync::Arc;

use arrayfire as af;
use async_trait::async_trait;
use futures::future;
use futures::stream::{self, FuturesOrdered, FuturesUnordered, Stream, StreamExt};
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

    async fn write<T: BlockTensorView>(
        self: Arc<Self>,
        txn_id: &TxnId,
        index: Index,
        value: Arc<T>,
    ) -> TCResult<()>;

    async fn write_one(
        self: Arc<Self>,
        txn_id: &TxnId,
        index: Index,
        number: Number,
    ) -> TCResult<()>;
}

#[async_trait]
impl<T: BlockTensorView, O: BlockTensorView> TensorBoolean<O> for T {
    type Base = BlockTensor;
    type Dense = BlockTensor;

    async fn and(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
        compatible(&self, &other)?;

        let blocks = self
            .clone()
            .chunk_stream(txn.id().clone())
            .zip(other.chunk_stream(txn.id().clone()))
            .map(|(l, r)| l?.and(&r?))
            .take_while(|r| future::ready(r.is_ok()));

        BlockTensor::from_blocks(txn, self.shape().clone(), self.dtype(), Box::pin(blocks)).await
    }

    async fn or(self: Arc<Self>, other: Arc<O>, _txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
        compatible(&self, &other)?;

        Err(error::not_implemented())
    }

    async fn xor(self: Arc<Self>, other: Arc<O>, _txn: Arc<Txn>) -> TCResult<Arc<Self::Dense>> {
        compatible(&self, &other)?;

        Err(error::not_implemented())
    }
}

#[async_trait]
impl<T: BlockTensorView + Slice, O: BlockTensorView> TensorCompare<O> for T {
    type Base = BlockTensor;
    type Dense = BlockTensor;

    async fn equals(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
        compatible(&self, &other)?;

        let blocks = self
            .clone()
            .chunk_stream(txn.id().clone())
            .zip(other.chunk_stream(txn.id().clone()))
            .map(|(l, r)| l?.equals(&r?))
            .take_while(|r| future::ready(r.is_ok()));

        BlockTensor::from_blocks(txn, self.shape().clone(), self.dtype(), Box::pin(blocks)).await
    }

    async fn gt(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
        compatible(&self, &other)?;

        let blocks = self
            .clone()
            .chunk_stream(txn.id().clone())
            .zip(other.chunk_stream(txn.id().clone()))
            .map(|(l, r)| l?.gt(&r?))
            .take_while(|r| future::ready(r.is_ok()));

        BlockTensor::from_blocks(txn, self.shape().clone(), self.dtype(), Box::pin(blocks)).await
    }

    async fn gte(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
        compatible(&self, &other)?;

        let blocks = self
            .clone()
            .chunk_stream(txn.id().clone())
            .zip(other.chunk_stream(txn.id().clone()))
            .map(|(l, r)| l?.gte(&r?))
            .take_while(|r| future::ready(r.is_ok()));

        BlockTensor::from_blocks(txn, self.shape().clone(), self.dtype(), Box::pin(blocks)).await
    }

    async fn lt(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
        compatible(&self, &other)?;

        let blocks = self
            .clone()
            .chunk_stream(txn.id().clone())
            .zip(other.chunk_stream(txn.id().clone()))
            .map(|(l, r)| l?.lt(&r?))
            .take_while(|r| future::ready(r.is_ok()));

        BlockTensor::from_blocks(txn, self.shape().clone(), self.dtype(), Box::pin(blocks)).await
    }

    async fn lte(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
        compatible(&self, &other)?;

        let blocks = self
            .clone()
            .chunk_stream(txn.id().clone())
            .zip(other.chunk_stream(txn.id().clone()))
            .map(|(l, r)| l?.lte(&r?))
            .take_while(|r| future::ready(r.is_ok()));

        BlockTensor::from_blocks(txn, self.shape().clone(), self.dtype(), Box::pin(blocks)).await
    }
}

#[async_trait]
impl<T: BlockTensorView + Slice> TensorUnary for T
where
    <T as Slice>::Slice: TensorUnary,
    <<T as Slice>::Slice as TensorUnary>::Base: BlockTensorView,
{
    type Base = BlockTensor;
    type Dense = BlockTensor;

    async fn as_dtype(
        self: Arc<Self>,
        txn: Arc<Txn>,
        dtype: NumberType,
    ) -> TCResult<Arc<BlockTensor>> {
        let shape = self.shape().clone();
        let per_block = per_block(dtype);
        let source = self
            .chunk_stream(txn.id().clone())
            .map(move |data| data.and_then(|d| d.into_type(dtype.clone())));
        let values = ValueStream::new(source);
        let chunks = ValueChunkStream::new(values, dtype, per_block);
        BlockTensor::from_blocks(txn, shape, dtype, Box::pin(chunks)).await
    }

    async fn copy(self: Arc<Self>, txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
        let shape = self.shape().clone();
        let dtype = self.dtype();
        let blocks = Box::pin(self.chunk_stream(txn.id().clone()));
        BlockTensor::from_blocks(txn, shape, dtype, blocks).await
    }

    async fn abs(self: Arc<Self>, txn: Arc<Txn>) -> TCResult<Arc<Self::Base>> {
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

    async fn sum(self: Arc<Self>, txn: Arc<Txn>, axis: usize) -> TCResult<Arc<Self::Base>> {
        if axis >= self.ndim() {
            return Err(error::bad_request("Axis out of range", axis));
        }

        let txn_id = txn.id().clone();
        let mut shape = self.shape().clone();
        shape.remove(axis);
        let summed = BlockTensor::constant(txn.clone(), shape, self.dtype().zero()).await?;

        if axis == 0 {
            reduce_axis0(self)
                .then(|r| async {
                    let (index, slice) = r?;
                    let value = slice.sum_all(txn_id.clone()).await?;
                    summed.clone().write_one(&txn_id, index, value).await
                })
                .fold(Ok(()), |_, r| future::ready(r))
                .await?;
        } else {
            reduce_axis(self, axis)
                .then(|r| async {
                    let (index, slice) = r?;
                    let value = slice.sum(txn.clone().subcontext_tmp().await?, 0).await?;
                    summed.clone().write(&txn_id, index, value).await
                })
                .fold(Ok(()), |_, r| future::ready(r))
                .await?;
        }

        Ok(summed)
    }

    async fn sum_all(self: Arc<Self>, txn_id: TxnId) -> TCResult<Number> {
        let mut sum = self.dtype().zero();
        let mut chunks = self.chunk_stream(txn_id);
        while let Some(chunk) = chunks.next().await {
            sum = sum + chunk?.product();
        }

        Ok(sum)
    }

    async fn product(self: Arc<Self>, txn: Arc<Txn>, axis: usize) -> TCResult<Arc<Self::Base>> {
        if axis >= self.ndim() {
            return Err(error::bad_request("Axis out of range", axis));
        }

        let txn_id = txn.id().clone();
        let mut shape = self.shape().clone();
        shape.remove(axis);
        let product = BlockTensor::constant(txn.clone(), shape, self.dtype().zero()).await?;

        if axis == 0 {
            reduce_axis0(self)
                .then(|r| async {
                    let (index, slice) = r?;
                    let value = slice.product_all(txn_id.clone()).await?;
                    product.clone().write_one(&txn_id, index, value).await
                })
                .fold(Ok(()), |_, r| future::ready(r))
                .await?;
        } else {
            reduce_axis(self, axis)
                .then(|r| async {
                    let (index, slice) = r?;
                    let value = slice
                        .product(txn.clone().subcontext_tmp().await?, 0)
                        .await?;
                    product.clone().write(&txn_id, index, value).await
                })
                .fold(Ok(()), |_, r| future::ready(r))
                .await?;
        }

        Ok(product)
    }

    async fn product_all(self: Arc<Self>, txn_id: TxnId) -> TCResult<Number> {
        let mut product = self.dtype().one();
        let mut chunks = self.chunk_stream(txn_id);
        while let Some(chunk) = chunks.next().await {
            product = product * chunk?.product();
        }

        Ok(product)
    }

    async fn not(self: Arc<Self>, txn: Arc<Txn>) -> TCResult<Arc<Self::Dense>> {
        let blocks = self
            .clone()
            .as_dtype(txn.clone(), NumberType::Bool)
            .await?
            .chunk_stream(txn.id().clone())
            .map(|c| Ok(c?.not()))
            .take_while(|r| future::ready(r.is_ok()));

        BlockTensor::from_blocks(
            txn,
            self.shape().clone(),
            NumberType::Bool,
            Box::pin(blocks),
        )
        .await
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
    async fn constant(txn: Arc<Txn>, shape: Shape, value: Number) -> TCResult<Arc<BlockTensor>> {
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
    ) -> TCResult<Arc<BlockTensor>> {
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

        Ok(Arc::new(BlockTensor {
            dtype,
            shape,
            size,
            ndim,
            file,
            per_block: per_block(dtype),
            coord_index,
        }))
    }

    async fn get_chunk(&self, txn_id: &TxnId, chunk_id: u64) -> TCResult<Chunk> {
        if let Some(block) = self.file.get_block(txn_id, &chunk_id.into()).await? {
            Chunk::try_from(block, self.dtype).await
        } else {
            Err(error::internal(ERR_CORRUPT))
        }
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
impl TensorBase for BlockTensor {
    async fn zeros(txn: Arc<Txn>, shape: Shape, dtype: NumberType) -> TCResult<Arc<Self>> {
        Self::constant(txn, shape, dtype.zero()).await
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
            let (chunk_ids, af_indices, af_offsets, num_coords) =
                coord_chunk(coords, &coord_index, per_block, ndim);

            let this = self.clone();
            let txn_id = txn_id.clone();

            selected.push(async move {
                let mut start = 0.0f64;
                let mut values = vec![];
                for chunk_id in chunk_ids {
                    let (block_offsets, new_start) =
                        block_offsets(&af_indices, &af_offsets, num_coords, start, chunk_id);

                    match this.get_chunk(&txn_id, chunk_id).await {
                        Ok(chunk) => values.extend(chunk.data().get(block_offsets)),
                        Err(cause) => return stream::iter(vec![Err(cause)]),
                    }

                    start = new_start;
                }

                let values: Vec<TCResult<Number>> = values.drain(..).map(Ok).collect();
                stream::iter(values)
            });
        }

        Box::pin(selected.flatten())
    }

    async fn write<T: BlockTensorView>(
        self: Arc<Self>,
        txn_id: &TxnId,
        index: Index,
        value: Arc<T>,
    ) -> TCResult<()> {
        if !self.shape().contains(&index) {
            return Err(error::bad_request("Index out of bounds", index));
        }

        let writes = FuturesUnordered::new();
        let ndim = index.ndim();

        let coord_index = af::Array::new(
            &self.coord_index,
            af::Dim4::new(&[self.ndim as u64, 1, 1, 1]),
        );
        let per_block = self.per_block as u64;

        let mut blocks = stream::iter(index.affected())
            .chunks(self.per_block)
            .zip(value.chunk_stream(txn_id.clone()));
        while let Some((coords, block)) = blocks.next().await {
            let (chunk_ids, af_indices, af_offsets, num_coords) =
                coord_chunk(coords.into_iter(), &coord_index, per_block, ndim);

            let this = self.clone();

            let write = async move {
                let values = block?;
                let mut start = 0.0f64;
                for chunk_id in chunk_ids {
                    let (block_offsets, new_start) =
                        block_offsets(&af_indices, &af_offsets, num_coords, start, chunk_id);

                    let mut chunk = this.get_chunk(txn_id, chunk_id).await?.upgrade().await?;
                    chunk.data().set(block_offsets, &values)?;
                    chunk.sync().await?;
                    start = new_start;
                }

                Ok(())
            };

            writes.push(write)
        }

        writes
            .take_while(|r| future::ready(r.is_ok()))
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    async fn write_one(
        self: Arc<Self>,
        txn_id: &TxnId,
        index: Index,
        value: Number,
    ) -> TCResult<()> {
        if !self.shape().contains(&index) {
            return Err(error::bad_request("Index out of bounds", index));
        }

        let writes = FuturesUnordered::new();
        let ndim = index.ndim();

        let coord_index = af::Array::new(
            &self.coord_index,
            af::Dim4::new(&[self.ndim as u64, 1, 1, 1]),
        );
        let per_block = self.per_block as u64;

        let mut blocks = stream::iter(index.affected()).chunks(self.per_block);
        while let Some(coords) = blocks.next().await {
            let (chunk_ids, af_indices, af_offsets, num_coords) =
                coord_chunk(coords.into_iter(), &coord_index, per_block, ndim);

            let this = self.clone();
            let value = value.clone();

            let write = async move {
                let mut start = 0.0f64;
                for chunk_id in chunk_ids {
                    let value = value.clone();
                    let (block_offsets, new_start) =
                        block_offsets(&af_indices, &af_offsets, num_coords, start, chunk_id);

                    let mut chunk = this.get_chunk(txn_id, chunk_id).await?.upgrade().await?;
                    chunk.data().set(
                        block_offsets,
                        &ChunkData::constant(value, (new_start - start) as usize),
                    )?;
                    chunk.sync().await?;
                    start = new_start;
                }

                Ok(())
            };

            writes.push(write)
        }

        writes
            .take_while(|r| future::ready(r.is_ok()))
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }
}

impl Slice for BlockTensor {
    type Slice = TensorSlice<BlockTensor>;

    fn slice(self: Arc<Self>, index: Index) -> TCResult<Arc<Self::Slice>> {
        Ok(Arc::new(TensorSlice::new(self, index)?))
    }
}

impl Transpose for BlockTensor {
    type Permutation = Permutation<BlockTensor>;

    fn transpose(
        self: Arc<Self>,
        permutation: Option<Vec<usize>>,
    ) -> TCResult<Arc<Self::Permutation>> {
        Ok(Arc::new(Permutation::new(self, permutation)))
    }
}

#[async_trait]
impl<T: Rebase + Slice + 'static> BlockTensorView for T
where
    <Self as Rebase>::Source: BlockTensorView,
{
    type ChunkStream = ValueChunkStream<Self::ValueStream>;
    type ValueStream = <<Self as Rebase>::Source as BlockTensorView>::ValueStream;

    fn chunk_stream(self: Arc<Self>, txn_id: TxnId) -> Self::ChunkStream {
        let dtype = self.source().dtype();
        let index = self.shape().all();
        ValueChunkStream::new(self.value_stream(txn_id, index), dtype, per_block(dtype))
    }

    fn value_stream(self: Arc<Self>, txn_id: TxnId, index: Index) -> Self::ValueStream {
        assert!(self.shape().contains(&index));
        self.source().value_stream(txn_id, self.invert_index(index))
    }

    async fn write<O: BlockTensorView>(
        self: Arc<Self>,
        txn_id: &TxnId,
        index: Index,
        value: Arc<O>,
    ) -> TCResult<()> {
        self.source()
            .write(txn_id, self.invert_index(index), value)
            .await
    }

    async fn write_one(
        self: Arc<Self>,
        txn_id: &TxnId,
        index: Index,
        value: Number,
    ) -> TCResult<()> {
        self.source()
            .write_one(txn_id, self.invert_index(index), value)
            .await
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

fn compatible<L: TensorView, R: TensorView>(l: &Arc<L>, r: &Arc<R>) -> TCResult<()> {
    if l.shape() != r.shape() {
        Err(error::bad_request(
            "Can't compare shapes (try broadcasting first)",
            format!("{} != {}", l.shape(), r.shape()),
        ))
    } else if l.dtype() != r.dtype() {
        Err(error::bad_request(
            "Can't compare data types (try casting first)",
            format!("{} != {}", l.dtype(), r.dtype()),
        ))
    } else {
        Ok(())
    }
}

fn block_offsets(
    af_indices: &af::Array<u64>,
    af_offsets: &af::Array<u64>,
    num_coords: u64,
    start: f64,
    chunk_id: u64,
) -> (af::Array<u64>, f64) {
    let num_to_update = af::sum_all(&af::eq(
        af_indices,
        &af::constant(chunk_id, af_indices.dims()),
        false,
    ))
    .0;
    let block_offsets = af::index(
        af_offsets,
        &[
            af::Seq::new(chunk_id as f64, chunk_id as f64, 1.0f64),
            af::Seq::new(start, (start + num_to_update) - 1.0f64, 1.0f64),
        ],
    );
    let block_offsets = af::moddims(&block_offsets, af::Dim4::new(&[num_coords as u64, 1, 1, 1]));

    (block_offsets, (start + num_to_update))
}

fn coord_chunk<I: Iterator<Item = Vec<u64>>>(
    coords: I,
    coord_index: &af::Array<u64>,
    per_block: u64,
    ndim: usize,
) -> (Vec<u64>, af::Array<u64>, af::Array<u64>, u64) {
    let coords: Vec<u64> = coords.flatten().collect();
    let num_coords = coords.len() / ndim;
    let af_coords_dim = af::Dim4::new(&[num_coords as u64, ndim as u64, 1, 1]);
    let af_coords = af::Array::new(&coords, af_coords_dim) * af::tile(coord_index, af_coords_dim);
    let af_coords = af::sum(&af_coords, 1);
    let af_per_block = af::constant(per_block, af::Dim4::new(&[1, num_coords as u64, 1, 1]));
    let af_offsets = af_coords.copy() % af_per_block.copy();
    let af_indices = af_coords / af_per_block;
    let af_chunk_ids = af::set_unique(&af_indices, true);

    let mut chunk_ids: Vec<u64> = Vec::with_capacity(af_chunk_ids.elements());
    af_chunk_ids.host(&mut chunk_ids);
    (chunk_ids, af_indices, af_offsets, num_coords as u64)
}

fn reduce_axis0<T: BlockTensorView + Slice>(
    source: Arc<T>,
) -> impl Stream<Item = TCResult<(Index, Arc<<T as Slice>::Slice>)>> {
    assert!(source.shape().len() > 1);
    let shape: Shape = source.shape()[1..].to_vec().into();
    let axis_index = source.shape().all()[0].clone();
    stream::iter(shape.all().affected())
        .map(move |coord| {
            let source_index: Index = (axis_index.clone(), coord.clone()).into();
            let slice = source.clone().slice(source_index)?;
            Ok((coord.into(), slice))
        })
        .take_while(|r| future::ready(r.is_ok()))
}

fn reduce_axis<T: BlockTensorView + Slice>(
    source: Arc<T>,
    axis: usize,
) -> impl Stream<Item = TCResult<(Index, Arc<<T as Slice>::Slice>)>> {
    let prefix_range: Shape = source.shape()[0..axis].to_vec().into();
    stream::iter(prefix_range.all().affected())
        .map(move |coord| {
            let index: Index = coord.into();
            let slice = source.clone().slice(index.clone())?;
            Ok((index, slice))
        })
        .take_while(|r| future::ready(r.is_ok()))
}
