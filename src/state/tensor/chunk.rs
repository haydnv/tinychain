use std::convert::TryInto;

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
    pub fn new(dtype: &TCType, len: usize) -> TCResult<ChunkData> {
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
                        other => panic!("BlockTensor corrupted! {} is not a valid boolean", other),
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

    pub fn set(&mut self, index: usize, value: Value) -> TCResult<()> {
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
