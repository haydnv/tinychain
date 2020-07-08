use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use arrayfire as af;
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{self, StreamExt};
use num::Complex;

use crate::error;
use crate::state::file::{Block, File};
use crate::transaction::lock::{TxnLockReadGuard, TxnLockWriteGuard};
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, TCType, Value};

use super::base::*;

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

        Ok(BlockTensor {
            dtype,
            shape,
            size,
            ndim,
            file,
            per_block,
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
                .zip(self.shape.to_vec().iter())
                .map(|(c, i)| c * i)
                .sum();
            let block_id = index / (self.per_block as u64);
            let offset = index % (self.per_block as u64);
            let mut chunk = self.get_chunk(txn_id, block_id).await?.upgrade().await?;
            chunk
                .data
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
        Ok(chunk.data.get(offset as usize))
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

pub struct Chunk {
    block: TxnLockReadGuard<Block>,
    data: ChunkData,
}

impl Chunk {
    async fn try_from(block: TxnLockReadGuard<Block>, dtype: TCType) -> TCResult<Chunk> {
        let data = ChunkData::try_from(block.as_bytes().await, dtype)?;
        Ok(Chunk { block, data })
    }

    async fn upgrade(self) -> TCResult<ChunkMut> {
        Ok(ChunkMut {
            block: self.block.upgrade().await?,
            data: self.data,
        })
    }
}

struct ChunkMut {
    block: TxnLockWriteGuard<Block>,
    data: ChunkData,
}

impl ChunkMut {
    async fn sync(self) -> TCResult<()> {
        self.block.rewrite(self.data.into()).await;

        Ok(())
    }

    async fn sync_and_downgrade(self, txn_id: &TxnId) -> TCResult<Chunk> {
        self.block.rewrite(self.data.clone().into()).await;

        Ok(Chunk {
            block: self.block.downgrade(txn_id).await?,
            data: self.data,
        })
    }
}

#[derive(Clone)]
enum ChunkData {
    Bool(af::Array<bool>),
    C32(af::Array<Complex<f32>>),
    C64(af::Array<Complex<f64>>),
    F32(af::Array<f32>),
    F64(af::Array<f64>),
    I16(af::Array<i16>),
    I32(af::Array<i32>),
    I64(af::Array<i64>),
    U8(af::Array<u8>),
    U16(af::Array<u16>),
    U32(af::Array<u32>),
    U64(af::Array<u64>),
}

impl ChunkData {
    fn new(dtype: &TCType, len: usize) -> TCResult<ChunkData> {
        let dim = dim4(len as u64);

        use ChunkData::*;
        use TCType::*;
        match dtype {
            TCType::Bool => Ok(ChunkData::Bool(af::constant(false, dim))),
            Complex32 => Ok(C32(af::constant(Complex::new(0.0f32, 0.0f32), dim))),
            Complex64 => Ok(C64(af::constant(Complex::new(0.0f64, 0.0f64), dim))),
            Float32 => Ok(F32(af::constant(0.0f32, dim))),
            Float64 => Ok(F64(af::constant(0.0f64, dim))),
            Int16 => Ok(I16(af::constant(0i16, dim))),
            Int32 => Ok(I32(af::constant(0i32, dim))),
            Int64 => Ok(I64(af::constant(0i64, dim))),
            UInt8 => Ok(U8(af::constant(0u8, dim))),
            UInt16 => Ok(U16(af::constant(0u16, dim))),
            UInt32 => Ok(U32(af::constant(0u32, dim))),
            UInt64 => Ok(U64(af::constant(0u64, dim))),
            _ => Err(error::bad_request("Tensor does not support", dtype)),
        }
    }

    fn try_from(data: Bytes, dtype: TCType) -> TCResult<ChunkData> {
        use ChunkData::*;
        use TCType::*;
        match dtype {
            TCType::Bool => {
                let mut array = Vec::with_capacity(data.len());
                for byte in data {
                    match byte as u8 {
                        0 => array.push(false),
                        1 => array.push(false),
                        _ => {
                            return Err(error::internal(ERR_CORRUPT));
                        }
                    }
                }
                let dim = dim4(array.len() as u64);
                Ok(ChunkData::Bool(af::Array::new(&array, dim)))
            }
            Complex32 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for c in data[..].chunks_exact(8) {
                    let re = f32::from_be_bytes(c[0..4].try_into().unwrap());
                    let im = f32::from_be_bytes(c[4..8].try_into().unwrap());
                    array.push(Complex::new(re, im));
                }
                let dim = dim4(array.len() as u64);
                Ok(C32(af::Array::new(&array, dim)))
            }
            Complex64 => {
                assert!(data.len() % 16 == 0);

                let mut array = Vec::with_capacity(data.len() / 16);
                for c in data[..].chunks_exact(16) {
                    let re = f64::from_be_bytes(c[0..8].try_into().unwrap());
                    let im = f64::from_be_bytes(c[8..16].try_into().unwrap());
                    array.push(Complex::new(re, im));
                }
                let dim = dim4(array.len() as u64);
                Ok(C64(af::Array::new(&array, dim)))
            }
            Float32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(f32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len() as u64);
                Ok(F32(af::Array::new(&array, dim)))
            }
            Float64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(f64::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len() as u64);
                Ok(F64(af::Array::new(&array, dim)))
            }
            Int16 => {
                assert!(data.len() % 2 == 0);

                let mut array = Vec::with_capacity(data.len() / 2);
                for f in data[..].chunks_exact(2) {
                    array.push(i16::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len() as u64);
                Ok(I16(af::Array::new(&array, dim)))
            }
            Int32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(i32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len() as u64);
                Ok(I32(af::Array::new(&array, dim)))
            }
            Int64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(i32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len() as u64);
                Ok(I32(af::Array::new(&array, dim)))
            }
            UInt8 => {
                let dim = dim4(data.len() as u64);
                Ok(U8(af::Array::new(&data, dim)))
            }
            UInt16 => {
                assert!(data.len() % 2 == 0);

                let mut array = Vec::with_capacity(data.len() / 2);
                for f in data[..].chunks_exact(2) {
                    array.push(u16::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len() as u64);
                Ok(U16(af::Array::new(&array, dim)))
            }
            UInt32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(u32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len() as u64);
                Ok(U32(af::Array::new(&array, dim)))
            }
            UInt64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(u32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len() as u64);
                Ok(U32(af::Array::new(&array, dim)))
            }
            other => Err(error::bad_request("Tensor does not support", other)),
        }
    }

    fn get(&self, index: usize) -> Value {
        let seq = af::Seq::new(index as f64, index as f64, 1.0f64);
        use ChunkData::*;
        match self {
            Bool(b) => {
                let mut value: Vec<bool> = Vec::with_capacity(1);
                af::index(b, &[seq]).host(&mut value);
                value[0].into()
            }
            C32(c) => {
                let mut value: Vec<Complex<f32>> = Vec::with_capacity(1);
                af::index(c, &[seq]).host(&mut value);
                value[0].into()
            }
            C64(c) => {
                let mut value: Vec<Complex<f64>> = Vec::with_capacity(1);
                af::index(c, &[seq]).host(&mut value);
                value[0].into()
            }
            F32(f) => {
                let mut value: Vec<f32> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
            F64(f) => {
                let mut value: Vec<f64> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
            I16(f) => {
                let mut value: Vec<i16> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
            I32(f) => {
                let mut value: Vec<i32> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
            I64(f) => {
                let mut value: Vec<i64> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
            U8(f) => {
                let mut value: Vec<u8> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
            U16(f) => {
                let mut value: Vec<u16> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
            U32(f) => {
                let mut value: Vec<u32> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
            U64(f) => {
                let mut value: Vec<u64> = Vec::with_capacity(1);
                af::index(f, &[seq]).host(&mut value);
                value[0].into()
            }
        }
    }

    fn set(&mut self, index: usize, value: Value) -> TCResult<()> {
        let seq = af::Seq::new(index as f64, index as f64, 1.0f64);
        use ChunkData::*;
        match self {
            Bool(b) => {
                let value: bool = value.try_into()?;
                af::assign_seq(b, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            C32(c) => {
                let value: Complex<f32> = value.try_into()?;
                af::assign_seq(c, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            C64(c) => {
                let value: Complex<f64> = value.try_into()?;
                af::assign_seq(c, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            F32(f) => {
                let value: f32 = value.try_into()?;
                af::assign_seq(f, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            F64(c) => {
                let value: f64 = value.try_into()?;
                af::assign_seq(c, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            I16(i) => {
                let value: i16 = value.try_into()?;
                af::assign_seq(i, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            I32(i) => {
                let value: i32 = value.try_into()?;
                af::assign_seq(i, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            I64(i) => {
                let value: i64 = value.try_into()?;
                af::assign_seq(i, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            U8(u) => {
                let value: u8 = value.try_into()?;
                af::assign_seq(u, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            U16(u) => {
                let value: u16 = value.try_into()?;
                af::assign_seq(u, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            U32(u) => {
                let value: u32 = value.try_into()?;
                af::assign_seq(u, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
            U64(u) => {
                let value: u64 = value.try_into()?;
                af::assign_seq(u, &[seq], &af::constant(value, dim4(1)));
                Ok(())
            }
        }
    }
}

impl From<ChunkData> for Bytes {
    fn from(chunk: ChunkData) -> Bytes {
        use ChunkData::*;
        match chunk {
            Bool(b) => {
                let mut data: Vec<bool> = Vec::with_capacity(b.elements());
                b.host(&mut data);
                let data: Vec<u8> = data.drain(..).map(|i| if i { 1u8 } else { 0u8 }).collect();
                data.into()
            }
            C32(c) => {
                let mut data: Vec<Complex<f32>> = Vec::with_capacity(c.elements());
                c.host(&mut data);
                let data: Vec<Vec<u8>> = data
                    .drain(..)
                    .map(|b| [b.re.to_be_bytes(), b.im.to_be_bytes()].concat())
                    .collect();
                data.into_iter().flatten().collect::<Vec<u8>>().into()
            }
            C64(c) => {
                let mut data: Vec<Complex<f64>> = Vec::with_capacity(c.elements());
                c.host(&mut data);
                let data: Vec<Vec<u8>> = data
                    .drain(..)
                    .map(|b| [b.re.to_be_bytes(), b.im.to_be_bytes()].concat())
                    .collect();
                data.into_iter().flatten().collect::<Vec<u8>>().into()
            }
            F32(f) => {
                let mut data: Vec<f32> = Vec::with_capacity(f.elements());
                f.host(&mut data);
                let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            F64(f) => {
                let mut data: Vec<f64> = Vec::with_capacity(f.elements());
                f.host(&mut data);
                let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            I16(i) => {
                let mut data: Vec<i16> = Vec::with_capacity(i.elements());
                i.host(&mut data);
                let data: Vec<[u8; 2]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            I32(i) => {
                let mut data: Vec<i32> = Vec::with_capacity(i.elements());
                i.host(&mut data);
                let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            I64(i) => {
                let mut data: Vec<i64> = Vec::with_capacity(i.elements());
                i.host(&mut data);
                let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            U8(b) => {
                let mut data: Vec<u8> = Vec::with_capacity(b.elements());
                b.host(&mut data);
                data.into()
            }
            U16(u) => {
                let mut data: Vec<u16> = Vec::with_capacity(u.elements());
                u.host(&mut data);
                let data: Vec<[u8; 2]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            U32(u) => {
                let mut data: Vec<u32> = Vec::with_capacity(u.elements());
                u.host(&mut data);
                let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            U64(u) => {
                let mut data: Vec<u64> = Vec::with_capacity(u.elements());
                u.host(&mut data);
                let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
        }
    }
}

fn dim4(size: u64) -> af::Dim4 {
    af::Dim4::new(&[size, 1, 1, 1])
}
