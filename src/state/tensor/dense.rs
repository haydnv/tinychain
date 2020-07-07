use std::iter;
use std::sync::Arc;

use arrayfire::{Array, Dim4};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{self, StreamExt};
use num::Complex;

use crate::error;
use crate::state::file::File;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, TCType};

use super::base::*;

const BLOCK_SIZE: usize = 1_000_000;

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

    async fn blocks(
        &self,
        txn_id: &Arc<TxnId>,
        len: usize,
    ) -> TCStream<Array<<Self as TensorView>::DType>>;
}

pub struct BlockTensor {
    dtype: TCType,
    shape: Shape,
    size: u64,
    ndim: usize,
    file: Arc<File>,
}

impl BlockTensor {
    async fn zeros(txn: Arc<Txn>, shape: Shape, dtype: TCType) -> TCResult<BlockTensor> {
        if !dtype.is_numeric() {
            return Err(error::bad_request("Tensor does not support", dtype));
        }

        let per_block = BLOCK_SIZE / dtype.size().unwrap();
        let size = shape.size();

        let blocks =
            (0..(size / per_block as u64)).map(move |_| Block::new(&dtype, per_block).unwrap());
        let trailing_len = (size % (per_block as u64)) as usize;
        let blocks: TCStream<Block> = if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(Block::new(&dtype, trailing_len).unwrap()));
            Box::pin(stream::iter(blocks))
        } else {
            Box::pin(stream::iter(blocks))
        };
        BlockTensor::from_blocks(txn, shape, dtype, blocks).await
    }

    async fn from_blocks(
        txn: Arc<Txn>,
        shape: Shape,
        dtype: TCType,
        mut blocks: TCStream<Block>,
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
        })
    }
}

pub struct DenseRebase<T: Rebase + 'static> {
    source: T,
}

#[async_trait]
impl<T: Rebase> TensorView for DenseRebase<T> {
    type DType = T::DType;

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
}

type DenseBroadcast<T> = DenseRebase<TensorBroadcast<T>>;
type DenseExpansion<T> = DenseRebase<Expansion<T>>;
type DensePermutation<T> = DenseRebase<Permutation<T>>;
type DenseTensorSlice<T> = DenseRebase<TensorSlice<T>>;

enum Block {
    Bool(Array<bool>),
    C32(Array<Complex<f32>>),
    C64(Array<Complex<f64>>),
    F32(Array<f32>),
    F64(Array<f64>),
    I16(Array<i16>),
    I32(Array<i32>),
    I64(Array<i64>),
    U8(Array<u8>),
    U16(Array<u16>),
    U32(Array<u32>),
    U64(Array<u64>),
}

impl Block {
    fn new(dtype: &TCType, len: usize) -> TCResult<Block> {
        let dim = Dim4::new(&[len as u64, 0, 0, 0]);

        use TCType::*;
        match dtype {
            Bool => {
                let data: Vec<bool> = iter::repeat(false).take(len).collect();
                Ok(Block::Bool(Array::new(&data, dim)))
            }
            Complex32 => {
                let data: Vec<Complex<f32>> =
                    iter::repeat(Complex::new(0., 0.)).take(len).collect();
                Ok(Block::C32(Array::new(&data, dim)))
            }
            Complex64 => {
                let data: Vec<Complex<f64>> =
                    iter::repeat(Complex::new(0., 0.)).take(len).collect();
                Ok(Block::C64(Array::new(&data, dim)))
            }
            Float32 => {
                let data: Vec<f32> = iter::repeat(0.).take(len).collect();
                Ok(Block::F32(Array::new(&data, dim)))
            }
            Float64 => {
                let data: Vec<f64> = iter::repeat(0.).take(len).collect();
                Ok(Block::F64(Array::new(&data, dim)))
            }
            Int16 => {
                let data: Vec<i16> = iter::repeat(0).take(len).collect();
                Ok(Block::I16(Array::new(&data, dim)))
            }
            Int32 => {
                let data: Vec<i32> = iter::repeat(0).take(len).collect();
                Ok(Block::I32(Array::new(&data, dim)))
            }
            Int64 => {
                let data: Vec<i64> = iter::repeat(0).take(len).collect();
                Ok(Block::I64(Array::new(&data, dim)))
            }
            UInt8 => {
                let data: Vec<u8> = iter::repeat(0).take(len).collect();
                Ok(Block::U8(Array::new(&data, dim)))
            }
            UInt16 => {
                let data: Vec<u16> = iter::repeat(0).take(len).collect();
                Ok(Block::U16(Array::new(&data, dim)))
            }
            UInt32 => {
                let data: Vec<u32> = iter::repeat(0).take(len).collect();
                Ok(Block::U32(Array::new(&data, dim)))
            }
            UInt64 => {
                let data: Vec<u64> = iter::repeat(0).take(len).collect();
                Ok(Block::U64(Array::new(&data, dim)))
            }
            _ => Err(error::bad_request("Tensor does not support", dtype)),
        }
    }
}

impl From<Block> for Bytes {
    fn from(block: Block) -> Bytes {
        use Block::*;
        match block {
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
