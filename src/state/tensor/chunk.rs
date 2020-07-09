use std::convert::{TryFrom, TryInto};

use arrayfire as af;
use bytes::Bytes;
use num::Complex;

use crate::error;
use crate::state::file::Block;
use crate::transaction::lock::{TxnLockReadGuard, TxnLockWriteGuard};
use crate::transaction::TxnId;
use crate::value::{TCResult, TCType, Value};

pub struct Chunk {
    block: TxnLockReadGuard<Block>,
    data: ChunkData,
}

impl Chunk {
    pub fn data(&'_ self) -> &'_ ChunkData {
        &self.data
    }

    pub async fn try_from(block: TxnLockReadGuard<Block>, dtype: TCType) -> TCResult<Chunk> {
        let data = ChunkData::try_from(block.as_bytes().await, dtype)?;
        Ok(Chunk { block, data })
    }

    pub async fn upgrade(self) -> TCResult<ChunkMut> {
        Ok(ChunkMut {
            block: self.block.upgrade().await?,
            data: self.data,
        })
    }
}

pub struct ChunkMut {
    block: TxnLockWriteGuard<Block>,
    data: ChunkData,
}

impl ChunkMut {
    pub fn data(&'_ mut self) -> &'_ mut ChunkData {
        &mut self.data
    }

    pub async fn sync(self) -> TCResult<()> {
        self.block.rewrite(self.data.into()).await;

        Ok(())
    }

    pub async fn sync_and_downgrade(self, txn_id: &TxnId) -> TCResult<Chunk> {
        self.block.rewrite(self.data.clone().into()).await;

        Ok(Chunk {
            block: self.block.downgrade(txn_id).await?,
            data: self.data,
        })
    }
}

#[derive(Clone)]
pub enum ChunkData {
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
    pub fn constant(value: Value, len: usize) -> TCResult<ChunkData> {
        let dim = dim4(len);

        use ChunkData::*;
        match value {
            Value::Bool(b) => Ok(ChunkData::Bool(af::constant(b, dim))),
            Value::Complex32(c) => Ok(C32(af::constant(c, dim))),
            Value::Complex64(c) => Ok(C64(af::constant(c, dim))),
            Value::Float32(f) => Ok(F32(af::constant(f, dim))),
            Value::Float64(f) => Ok(F64(af::constant(f, dim))),
            Value::Int16(i) => Ok(I16(af::constant(i, dim))),
            Value::Int32(i) => Ok(I32(af::constant(i, dim))),
            Value::Int64(i) => Ok(I64(af::constant(i, dim))),
            Value::UInt8(i) => Ok(U8(af::constant(i, dim))),
            Value::UInt16(u) => Ok(U16(af::constant(u, dim))),
            Value::UInt32(u) => Ok(U32(af::constant(u, dim))),
            Value::UInt64(u) => Ok(U64(af::constant(u, dim))),
            _ => Err(error::bad_request("Tensor does not support", value.dtype())),
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
                        other => panic!("BlockTensor corrupted! {} is not a valid boolean", other),
                    }
                }
                let dim = dim4(array.len());
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
                let dim = dim4(array.len());
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
                let dim = dim4(array.len());
                Ok(C64(af::Array::new(&array, dim)))
            }
            Float32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(f32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(F32(af::Array::new(&array, dim)))
            }
            Float64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(f64::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(F64(af::Array::new(&array, dim)))
            }
            Int16 => {
                assert!(data.len() % 2 == 0);

                let mut array = Vec::with_capacity(data.len() / 2);
                for f in data[..].chunks_exact(2) {
                    array.push(i16::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(I16(af::Array::new(&array, dim)))
            }
            Int32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(i32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(I32(af::Array::new(&array, dim)))
            }
            Int64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(i32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(I32(af::Array::new(&array, dim)))
            }
            UInt8 => {
                let dim = dim4(data.len());
                Ok(U8(af::Array::new(&data, dim)))
            }
            UInt16 => {
                assert!(data.len() % 2 == 0);

                let mut array = Vec::with_capacity(data.len() / 2);
                for f in data[..].chunks_exact(2) {
                    array.push(u16::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(U16(af::Array::new(&array, dim)))
            }
            UInt32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(u32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(U32(af::Array::new(&array, dim)))
            }
            UInt64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(u32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(U32(af::Array::new(&array, dim)))
            }
            other => Err(error::bad_request("Tensor does not support", other)),
        }
    }

    pub fn get(&self, index: usize) -> Value {
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

    pub fn set(&mut self, index: af::Array<u64>, other: &ChunkData) -> TCResult<()> {
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, Some(false));
        self.set_at(indexer, other)
    }

    pub fn set_one(&mut self, index: usize, value: Value) -> TCResult<()> {
        let mut indexer = af::Indexer::default();
        let seq = af::Seq::new(index as f64, index as f64, 1.0f64);
        indexer.set_index(&seq, 0, Some(false));
        self.set_at(indexer, &value.try_into()?)
    }

    fn set_at(&mut self, index: af::Indexer, value: &ChunkData) -> TCResult<()> {
        use ChunkData::*;
        match (self, value) {
            (Bool(l), Bool(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (C32(l), C32(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (C64(l), C64(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (F32(l), F32(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (F64(l), F64(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (I16(l), I16(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (I32(l), I32(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (I64(l), I64(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (U8(l), U8(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (U16(l), U16(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (U32(l), U32(r)) => {
                af::assign_gen(l, &index, &r);
            }
            (U64(l), U64(r)) => {
                af::assign_gen(l, &index, &r);
            }
            _ => {
                return Err(error::internal(
                    "Attempted to assign a Tensor chunk with the wrong datatype!",
                ));
            }
        }

        Ok(())
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

impl TryFrom<Value> for ChunkData {
    type Error = error::TCError;

    fn try_from(value: Value) -> TCResult<ChunkData> {
        match value {
            Value::Bool(b) => Ok(vec![b].into()),
            _ => Err(error::internal(format!(
                "Attepted to update a Tensor index to {}",
                value.dtype()
            ))),
        }
    }
}

impl From<Vec<bool>> for ChunkData {
    fn from(b: Vec<bool>) -> ChunkData {
        let data = af::Array::new(&b, dim4(b.len()));
        ChunkData::Bool(data)
    }
}

impl From<Vec<Complex<f32>>> for ChunkData {
    fn from(c: Vec<Complex<f32>>) -> ChunkData {
        let data = af::Array::new(&c, dim4(c.len()));
        ChunkData::C32(data)
    }
}

impl From<Vec<Complex<f64>>> for ChunkData {
    fn from(c: Vec<Complex<f64>>) -> ChunkData {
        let data = af::Array::new(&c, dim4(c.len()));
        ChunkData::C64(data)
    }
}

impl From<Vec<f32>> for ChunkData {
    fn from(f: Vec<f32>) -> ChunkData {
        let data = af::Array::new(&f, dim4(f.len()));
        ChunkData::F32(data)
    }
}

impl From<Vec<f64>> for ChunkData {
    fn from(f: Vec<f64>) -> ChunkData {
        let data = af::Array::new(&f, dim4(f.len()));
        ChunkData::F64(data)
    }
}

impl From<Vec<i16>> for ChunkData {
    fn from(i: Vec<i16>) -> ChunkData {
        let data = af::Array::new(&i, dim4(i.len()));
        ChunkData::I16(data)
    }
}

impl From<Vec<i32>> for ChunkData {
    fn from(i: Vec<i32>) -> ChunkData {
        let data = af::Array::new(&i, dim4(i.len()));
        ChunkData::I32(data)
    }
}

impl From<Vec<i64>> for ChunkData {
    fn from(i: Vec<i64>) -> ChunkData {
        let data = af::Array::new(&i, dim4(i.len()));
        ChunkData::I64(data)
    }
}

impl From<Vec<u8>> for ChunkData {
    fn from(u: Vec<u8>) -> ChunkData {
        let data = af::Array::new(&u, dim4(u.len()));
        ChunkData::U8(data)
    }
}

impl From<Vec<u16>> for ChunkData {
    fn from(u: Vec<u16>) -> ChunkData {
        let data = af::Array::new(&u, dim4(u.len()));
        ChunkData::U16(data)
    }
}

impl From<Vec<u32>> for ChunkData {
    fn from(u: Vec<u32>) -> ChunkData {
        let data = af::Array::new(&u, dim4(u.len()));
        ChunkData::U32(data)
    }
}

impl From<Vec<u64>> for ChunkData {
    fn from(u: Vec<u64>) -> ChunkData {
        let data = af::Array::new(&u, dim4(u.len()));
        ChunkData::U64(data)
    }
}

fn dim4(size: usize) -> af::Dim4 {
    af::Dim4::new(&[size as u64, 1, 1, 1])
}
