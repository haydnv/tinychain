use std::iter;
use std::sync::Arc;

use arrayfire::{Array, Dim4};
use async_trait::async_trait;
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
    shape: Shape,
    size: u64,
    ndim: usize,
    file: Arc<File>,
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
    fn new(dtype: TCType, len: usize) -> TCResult<Block> {
        let dim = Dim4::new(&[len as u64, 0, 0, 0]);

        use TCType::*;
        match dtype {
            Bool => {
                let data: Vec<bool> = iter::repeat(false).take(len).collect();
                Ok(Block::Bool(Array::new(&data, dim)))
            },
            Complex32 => {
                let data: Vec<Complex<f32>> = iter::repeat(Complex::new(0., 0.)).take(len).collect();
                Ok(Block::C32(Array::new(&data, dim)))
            },
            Complex64 => {
                let data: Vec<Complex<f64>> = iter::repeat(Complex::new(0., 0.)).take(len).collect();
                Ok(Block::C64(Array::new(&data, dim)))
            },
            Float32 => {
                let data: Vec<f32> = iter::repeat(0.).take(len).collect();
                Ok(Block::F32(Array::new(&data, dim)))
            },
            Float64 => {
                let data: Vec<f64> = iter::repeat(0.).take(len).collect();
                Ok(Block::F64(Array::new(&data, dim)))
            },
            Int16 => {
                let data: Vec<i16> = iter::repeat(0).take(len).collect();
                Ok(Block::I16(Array::new(&data, dim)))
            },
            Int32 => {
                let data: Vec<i32> = iter::repeat(0).take(len).collect();
                Ok(Block::I32(Array::new(&data, dim)))
            },
            Int64 => {
                let data: Vec<i64> = iter::repeat(0).take(len).collect();
                Ok(Block::I64(Array::new(&data, dim)))
            },
            UInt8 => {
                let data: Vec<u8> = iter::repeat(0).take(len).collect();
                Ok(Block::U8(Array::new(&data, dim)))
            },
            UInt16 => {
                let data: Vec<u16> = iter::repeat(0).take(len).collect();
                Ok(Block::U16(Array::new(&data, dim)))
            },
            UInt32 => {
                let data: Vec<u32> = iter::repeat(0).take(len).collect();
                Ok(Block::U32(Array::new(&data, dim)))
            },
            UInt64 => {
                let data: Vec<u64> = iter::repeat(0).take(len).collect();
                Ok(Block::U64(Array::new(&data, dim)))
            },
            _ => Err(error::bad_request("Tensor does not support", dtype))
        }
    }
}
