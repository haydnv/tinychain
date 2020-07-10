use std::convert::{TryFrom, TryInto};

use arrayfire as af;
use bytes::Bytes;
use num::Complex;

use crate::error;
use crate::state::file::Block;
use crate::transaction::lock::{TxnLockReadGuard, TxnLockWriteGuard};
use crate::transaction::TxnId;
use crate::value::{Number, NumberType, TCResult};

pub struct Chunk {
    block: TxnLockReadGuard<Block>,
    data: ChunkData,
}

impl Chunk {
    pub fn data(&'_ self) -> &'_ ChunkData {
        &self.data
    }

    pub async fn try_from(block: TxnLockReadGuard<Block>, dtype: NumberType) -> TCResult<Chunk> {
        let data = ChunkData::try_from_bytes(block.as_bytes().await, dtype)?;
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

pub trait TensorChunk {
    type DType: af::HasAfEnum + Into<Number>;

    fn array(&'_ self) -> &'_ af::Array<Self::DType>;

    fn as_type<T: af::HasAfEnum>(&self) -> ArrayExt<T> {
        ArrayExt(self.array().cast())
    }

    fn get(&self, index: af::Indexer) -> Vec<Number> {
        let array = af::index_gen(self.array(), index);
        let mut value: Vec<Self::DType> = Vec::with_capacity(array.elements());
        array.host(&mut value);
        value.drain(..).map(|v| v.into()).collect()
    }

    fn set<T: TensorChunk<DType = Self::DType>>(&self, index: &af::Indexer, other: &T) {
        af::assign_gen(self.array(), index, other.array());
    }
}

impl<T: af::HasAfEnum + Into<Number>> TensorChunk for ArrayExt<T> {
    type DType = T;

    fn array(&'_ self) -> &'_ af::Array<Self::DType> {
        &self.0
    }
}

impl<T: af::HasAfEnum> From<ArrayExt<T>> for Vec<T> {
    fn from(array: ArrayExt<T>) -> Vec<T> {
        let mut v: Vec<T> = Vec::with_capacity(array.0.elements());
        array.0.host(&mut v);
        v
    }
}

impl<T: af::HasAfEnum + Into<Number>> From<ArrayExt<T>> for Vec<Number> {
    fn from(array: ArrayExt<T>) -> Vec<Number> {
        let mut v: Vec<T> = array.into();
        v.drain(..).map(|i| i.into()).collect()
    }
}

#[derive(Clone)]
pub struct ArrayExt<T: af::HasAfEnum>(af::Array<T>);

impl<T: af::HasAfEnum> From<af::Array<T>> for ArrayExt<T> {
    fn from(array: af::Array<T>) -> ArrayExt<T> {
        ArrayExt(array)
    }
}

impl<E: Into<error::TCError>, T: af::HasAfEnum + TryFrom<Number, Error = E>> TryFrom<Vec<Number>>
    for ArrayExt<T>
{
    type Error = error::TCError;

    fn try_from(mut values: Vec<Number>) -> TCResult<ArrayExt<T>> {
        let array = values
            .drain(..)
            .map(|v| v.try_into().map_err(|e: E| e.into()))
            .collect::<TCResult<Vec<T>>>()?;
        let dim = dim4(array.len());
        Ok(ArrayExt(af::Array::new(&array, dim)))
    }
}

pub trait TensorChunkAbs: TensorChunk {
    type AbsValue: af::HasAfEnum + Into<Number>;

    fn abs(&self) -> ArrayExt<Self::AbsValue>;
}

impl TensorChunkAbs for ArrayExt<Complex<f32>> {
    type AbsValue = f32;

    fn abs(&self) -> ArrayExt<f32> {
        ArrayExt(af::abs(self.array()))
    }
}

impl TensorChunkAbs for ArrayExt<Complex<f64>> {
    type AbsValue = f64;

    fn abs(&self) -> ArrayExt<f64> {
        ArrayExt(af::abs(self.array()))
    }
}

impl TensorChunkAbs for ArrayExt<f32> {
    type AbsValue = f32;

    fn abs(&self) -> ArrayExt<f32> {
        ArrayExt(af::abs(self.array()))
    }
}

impl TensorChunkAbs for ArrayExt<f64> {
    type AbsValue = f64;

    fn abs(&self) -> ArrayExt<f64> {
        ArrayExt(af::abs(self.array()))
    }
}

impl TensorChunkAbs for ArrayExt<i16> {
    type AbsValue = i16;

    fn abs(&self) -> ArrayExt<i16> {
        ArrayExt(af::abs(self.array()).cast())
    }
}

impl TensorChunkAbs for ArrayExt<i32> {
    type AbsValue = i32;

    fn abs(&self) -> ArrayExt<i32> {
        ArrayExt(af::abs(self.array()).cast())
    }
}

impl TensorChunkAbs for ArrayExt<i64> {
    type AbsValue = i64;

    fn abs(&self) -> ArrayExt<i64> {
        ArrayExt(af::abs(self.array()).cast())
    }
}

pub trait TensorChunkAnyAll: TensorChunk {
    fn all(&self) -> bool {
        af::all_true_all(self.array()).0 > 0.0f64
    }

    fn any(&self) -> bool {
        af::any_true_all(self.array()).0 > 0.0f64
    }
}

impl TensorChunkAnyAll for ArrayExt<bool> {}
impl TensorChunkAnyAll for ArrayExt<f32> {}
impl TensorChunkAnyAll for ArrayExt<f64> {}
impl TensorChunkAnyAll for ArrayExt<i16> {}
impl TensorChunkAnyAll for ArrayExt<i32> {}
impl TensorChunkAnyAll for ArrayExt<i64> {}
impl TensorChunkAnyAll for ArrayExt<u8> {}
impl TensorChunkAnyAll for ArrayExt<u16> {}
impl TensorChunkAnyAll for ArrayExt<u32> {}
impl TensorChunkAnyAll for ArrayExt<u64> {}

impl TensorChunkAnyAll for ArrayExt<Complex<f32>> {
    fn all(&self) -> bool {
        let all = af::all_true_all(self.array());
        all.0 > 0.0f64 && all.1 > 0.0f64
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.array());
        any.0 > 0.0f64 || any.1 > 0.0f64
    }
}

impl TensorChunkAnyAll for ArrayExt<Complex<f64>> {
    fn all(&self) -> bool {
        let all = af::all_true_all(self.array());
        all.0 > 0.0f64 && all.1 > 0.0f64
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.array());
        any.0 > 0.0f64 || any.1 > 0.0f64
    }
}

#[derive(Clone)]
pub enum ChunkData {
    Bool(ArrayExt<bool>),
    C32(ArrayExt<Complex<f32>>),
    C64(ArrayExt<Complex<f64>>),
    F32(ArrayExt<f32>),
    F64(ArrayExt<f64>),
    I16(ArrayExt<i16>),
    I32(ArrayExt<i32>),
    I64(ArrayExt<i64>),
    U8(ArrayExt<u8>),
    U16(ArrayExt<u16>),
    U32(ArrayExt<u32>),
    U64(ArrayExt<u64>),
}

impl ChunkData {
    pub fn constant(value: Number, len: usize) -> ChunkData {
        let dim = dim4(len);

        use ChunkData::*;
        match value {
            Number::Bool(b) => ChunkData::Bool(af::constant(b, dim).into()),
            Number::Complex32(c) => C32(af::constant(c, dim).into()),
            Number::Complex64(c) => C64(af::constant(c, dim).into()),
            Number::Float32(f) => F32(af::constant(f, dim).into()),
            Number::Float64(f) => F64(af::constant(f, dim).into()),
            Number::Int16(i) => I16(af::constant(i, dim).into()),
            Number::Int32(i) => I32(af::constant(i, dim).into()),
            Number::Int64(i) => I64(af::constant(i, dim).into()),
            Number::UInt8(i) => U8(af::constant(i, dim).into()),
            Number::UInt16(u) => U16(af::constant(u, dim).into()),
            Number::UInt32(u) => U32(af::constant(u, dim).into()),
            Number::UInt64(u) => U64(af::constant(u, dim).into()),
        }
    }

    fn try_from_bytes(data: Bytes, dtype: NumberType) -> TCResult<ChunkData> {
        // TODO: replace calls to assert! with `return Err...`

        use ChunkData::*;
        use NumberType::*;
        match dtype {
            NumberType::Bool => {
                let mut array = Vec::with_capacity(data.len());
                for byte in data {
                    match byte as u8 {
                        0 => array.push(false),
                        1 => array.push(false),
                        other => panic!("BlockTensor corrupted! {} is not a valid boolean", other),
                    }
                }
                let dim = dim4(array.len());
                Ok(ChunkData::Bool(af::Array::new(&array, dim).into()))
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
                Ok(C32(af::Array::new(&array, dim).into()))
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
                Ok(C64(af::Array::new(&array, dim).into()))
            }
            Float32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(f32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(F32(af::Array::new(&array, dim).into()))
            }
            Float64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(f64::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(F64(af::Array::new(&array, dim).into()))
            }
            Int16 => {
                assert!(data.len() % 2 == 0);

                let mut array = Vec::with_capacity(data.len() / 2);
                for f in data[..].chunks_exact(2) {
                    array.push(i16::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(I16(af::Array::new(&array, dim).into()))
            }
            Int32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(i32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(I32(af::Array::new(&array, dim).into()))
            }
            Int64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(i32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(I32(af::Array::new(&array, dim).into()))
            }
            UInt8 => {
                let dim = dim4(data.len());
                Ok(U8(af::Array::new(&data, dim).into()))
            }
            UInt16 => {
                assert!(data.len() % 2 == 0);

                let mut array = Vec::with_capacity(data.len() / 2);
                for f in data[..].chunks_exact(2) {
                    array.push(u16::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(U16(af::Array::new(&array, dim).into()))
            }
            UInt32 => {
                assert!(data.len() % 4 == 0);

                let mut array = Vec::with_capacity(data.len() / 4);
                for f in data[..].chunks_exact(4) {
                    array.push(u32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(U32(af::Array::new(&array, dim).into()))
            }
            UInt64 => {
                assert!(data.len() % 8 == 0);

                let mut array = Vec::with_capacity(data.len() / 8);
                for f in data[..].chunks_exact(8) {
                    array.push(u32::from_be_bytes(f.try_into().unwrap()));
                }
                let dim = dim4(array.len());
                Ok(U32(af::Array::new(&array, dim).into()))
            }
        }
    }

    pub fn try_from_values(data: Vec<Number>, dtype: NumberType) -> TCResult<ChunkData> {
        use ChunkData::*;
        use NumberType::*;
        let chunk = match dtype {
            NumberType::Bool => ChunkData::Bool(data.try_into()?),
            Complex32 => C32(data.try_into()?),
            Complex64 => C64(data.try_into()?),
            Float32 => F32(data.try_into()?),
            Float64 => F64(data.try_into()?),
            Int16 => I16(data.try_into()?),
            Int32 => I32(data.try_into()?),
            Int64 => I64(data.try_into()?),
            UInt8 => U8(data.try_into()?),
            UInt16 => U16(data.try_into()?),
            UInt32 => U32(data.try_into()?),
            UInt64 => U64(data.try_into()?),
        };

        Ok(chunk)
    }

    pub fn into_type(self, dtype: NumberType) -> TCResult<ChunkData> {
        use ChunkData::*;
        use NumberType::*;
        let converted = match self {
            ChunkData::Bool(b) => match dtype {
                NumberType::Bool => ChunkData::Bool(b.as_type()),
                Complex32 => C32(b.as_type()),
                Complex64 => C64(b.as_type()),
                Float32 => F32(b.as_type()),
                Float64 => F64(b.as_type()),
                Int16 => I16(b.as_type()),
                Int32 => I32(b.as_type()),
                Int64 => I64(b.as_type()),
                UInt8 => U8(b.as_type()),
                UInt16 => U16(b.as_type()),
                UInt32 => U32(b.as_type()),
                UInt64 => U64(b.as_type()),
            },
            C32(c) => match dtype {
                NumberType::Bool => ChunkData::Bool(c.as_type()),
                Complex32 => C32(c.as_type()),
                Complex64 => C64(c.as_type()),
                Float32 => F32(c.as_type()),
                Float64 => F64(c.as_type()),
                Int16 => I16(c.as_type()),
                Int32 => I32(c.as_type()),
                Int64 => I64(c.as_type()),
                UInt8 => U8(c.as_type()),
                UInt16 => U16(c.as_type()),
                UInt32 => U32(c.as_type()),
                UInt64 => U64(c.as_type()),
            },
            C64(c) => match dtype {
                NumberType::Bool => ChunkData::Bool(c.as_type()),
                Complex32 => C32(c.as_type()),
                Complex64 => C64(c.as_type()),
                Float32 => F32(c.as_type()),
                Float64 => F64(c.as_type()),
                Int16 => I16(c.as_type()),
                Int32 => I32(c.as_type()),
                Int64 => I64(c.as_type()),
                UInt8 => U8(c.as_type()),
                UInt16 => U16(c.as_type()),
                UInt32 => U32(c.as_type()),
                UInt64 => U64(c.as_type()),
            },
            F32(f) => match dtype {
                NumberType::Bool => ChunkData::Bool(f.as_type()),
                Complex32 => C32(f.as_type()),
                Complex64 => C64(f.as_type()),
                Float32 => F32(f.as_type()),
                Float64 => F64(f.as_type()),
                Int16 => I16(f.as_type()),
                Int32 => I32(f.as_type()),
                Int64 => I64(f.as_type()),
                UInt8 => U8(f.as_type()),
                UInt16 => U16(f.as_type()),
                UInt32 => U32(f.as_type()),
                UInt64 => U64(f.as_type()),
            },
            F64(f) => match dtype {
                NumberType::Bool => ChunkData::Bool(f.as_type()),
                Complex32 => C32(f.as_type()),
                Complex64 => C64(f.as_type()),
                Float32 => F32(f.as_type()),
                Float64 => F64(f.as_type()),
                Int16 => I16(f.as_type()),
                Int32 => I32(f.as_type()),
                Int64 => I64(f.as_type()),
                UInt8 => U8(f.as_type()),
                UInt16 => U16(f.as_type()),
                UInt32 => U32(f.as_type()),
                UInt64 => U64(f.as_type()),
            },
            I16(i) => match dtype {
                NumberType::Bool => ChunkData::Bool(i.as_type()),
                Complex32 => C32(i.as_type()),
                Complex64 => C64(i.as_type()),
                Float32 => F32(i.as_type()),
                Float64 => F64(i.as_type()),
                Int16 => I16(i.as_type()),
                Int32 => I32(i.as_type()),
                Int64 => I64(i.as_type()),
                UInt8 => U8(i.as_type()),
                UInt16 => U16(i.as_type()),
                UInt32 => U32(i.as_type()),
                UInt64 => U64(i.as_type()),
            },
            I32(c) => match dtype {
                NumberType::Bool => ChunkData::Bool(c.as_type()),
                Complex32 => C32(c.as_type()),
                Complex64 => C64(c.as_type()),
                Float32 => F32(c.as_type()),
                Float64 => F64(c.as_type()),
                Int16 => I16(c.as_type()),
                Int32 => I32(c.as_type()),
                Int64 => I64(c.as_type()),
                UInt8 => U8(c.as_type()),
                UInt16 => U16(c.as_type()),
                UInt32 => U32(c.as_type()),
                UInt64 => U64(c.as_type()),
            },
            I64(c) => match dtype {
                NumberType::Bool => ChunkData::Bool(c.as_type()),
                Complex32 => C32(c.as_type()),
                Complex64 => C64(c.as_type()),
                Float32 => F32(c.as_type()),
                Float64 => F64(c.as_type()),
                Int16 => I16(c.as_type()),
                Int32 => I32(c.as_type()),
                Int64 => I64(c.as_type()),
                UInt8 => U8(c.as_type()),
                UInt16 => U16(c.as_type()),
                UInt32 => U32(c.as_type()),
                UInt64 => U64(c.as_type()),
            },
            U8(u) => match dtype {
                NumberType::Bool => ChunkData::Bool(u.as_type()),
                Complex32 => C32(u.as_type()),
                Complex64 => C64(u.as_type()),
                Float32 => F32(u.as_type()),
                Float64 => F64(u.as_type()),
                Int16 => I16(u.as_type()),
                Int32 => I32(u.as_type()),
                Int64 => I64(u.as_type()),
                UInt8 => U8(u.as_type()),
                UInt16 => U16(u.as_type()),
                UInt32 => U32(u.as_type()),
                UInt64 => U64(u.as_type()),
            },
            U16(u) => match dtype {
                NumberType::Bool => ChunkData::Bool(u.as_type()),
                Complex32 => C32(u.as_type()),
                Complex64 => C64(u.as_type()),
                Float32 => F32(u.as_type()),
                Float64 => F64(u.as_type()),
                Int16 => I16(u.as_type()),
                Int32 => I32(u.as_type()),
                Int64 => I64(u.as_type()),
                UInt8 => U8(u.as_type()),
                UInt16 => U16(u.as_type()),
                UInt32 => U32(u.as_type()),
                UInt64 => U64(u.as_type()),
            },
            U32(u) => match dtype {
                NumberType::Bool => ChunkData::Bool(u.as_type()),
                Complex32 => C32(u.as_type()),
                Complex64 => C64(u.as_type()),
                Float32 => F32(u.as_type()),
                Float64 => F64(u.as_type()),
                Int16 => I16(u.as_type()),
                Int32 => I32(u.as_type()),
                Int64 => I64(u.as_type()),
                UInt8 => U8(u.as_type()),
                UInt16 => U16(u.as_type()),
                UInt32 => U32(u.as_type()),
                UInt64 => U64(u.as_type()),
            },
            U64(u) => match dtype {
                NumberType::Bool => ChunkData::Bool(u.as_type()),
                Complex32 => C32(u.as_type()),
                Complex64 => C64(u.as_type()),
                Float32 => F32(u.as_type()),
                Float64 => F64(u.as_type()),
                Int16 => I16(u.as_type()),
                Int32 => I32(u.as_type()),
                Int64 => I64(u.as_type()),
                UInt8 => U8(u.as_type()),
                UInt16 => U16(u.as_type()),
                UInt32 => U32(u.as_type()),
                UInt64 => U64(u.as_type()),
            },
        };

        Ok(converted)
    }

    pub fn abs(&self) -> TCResult<ChunkData> {
        use ChunkData::*;
        match self {
            C32(c) => Ok(F32(c.abs())),
            C64(c) => Ok(F64(c.abs())),
            F32(f) => Ok(F32(f.abs())),
            F64(f) => Ok(F64(f.abs())),
            I16(i) => Ok(I16(i.abs())),
            I32(i) => Ok(I32(i.abs())),
            I64(i) => Ok(I64(i.abs())),
            other => Ok(other.clone()),
        }
    }

    pub fn all(&self) -> bool {
        use ChunkData::*;
        match self {
            Bool(b) => b.all(),
            C32(c) => c.all(),
            C64(c) => c.all(),
            F32(f) => f.all(),
            F64(f) => f.all(),
            I16(i) => i.all(),
            I32(i) => i.all(),
            I64(i) => i.all(),
            U8(u) => u.all(),
            U16(u) => u.all(),
            U32(u) => u.all(),
            U64(u) => u.all(),
        }
    }

    pub fn any(&self) -> bool {
        use ChunkData::*;
        match self {
            Bool(b) => b.any(),
            C32(c) => c.any(),
            C64(c) => c.any(),
            F32(f) => f.any(),
            F64(f) => f.any(),
            I16(i) => i.any(),
            I32(i) => i.any(),
            I64(i) => i.any(),
            U8(u) => u.any(),
            U16(u) => u.any(),
            U32(u) => u.any(),
            U64(u) => u.any(),
        }
    }

    pub fn get_one(&self, index: usize) -> Option<Number> {
        let seq = af::Seq::new(index as f64, index as f64, 1.0f64);
        let mut indexer = af::Indexer::default();
        indexer.set_index(&seq, 0, None);
        self.get_at(indexer).pop()
    }

    pub fn get(&self, index: af::Array<u64>) -> Vec<Number> {
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, Some(false));
        self.get_at(indexer)
    }

    fn get_at(&self, index: af::Indexer) -> Vec<Number> {
        use ChunkData::*;
        match self {
            Bool(b) => b.get(index),
            C32(c) => c.get(index),
            C64(c) => c.get(index),
            F32(f) => f.get(index),
            F64(f) => f.get(index),
            I16(i) => i.get(index),
            I32(i) => i.get(index),
            I64(i) => i.get(index),
            U8(i) => i.get(index),
            U16(i) => i.get(index),
            U32(i) => i.get(index),
            U64(i) => i.get(index),
        }
    }

    pub fn set(&mut self, index: af::Array<u64>, other: &ChunkData) -> TCResult<()> {
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, Some(false));
        self.set_at(indexer, other)
    }

    fn set_at(&mut self, index: af::Indexer, value: &ChunkData) -> TCResult<()> {
        use ChunkData::*;
        match (self, value) {
            (Bool(l), Bool(r)) => l.set(&index, r),
            (C32(l), C32(r)) => l.set(&index, r),
            (C64(l), C64(r)) => l.set(&index, r),
            (F32(l), F32(r)) => l.set(&index, r),
            (F64(l), F64(r)) => l.set(&index, r),
            (I16(l), I16(r)) => l.set(&index, r),
            (I32(l), I32(r)) => l.set(&index, r),
            (I64(l), I64(r)) => l.set(&index, r),
            (U8(l), U8(r)) => l.set(&index, r),
            (U16(l), U16(r)) => l.set(&index, r),
            (U32(l), U32(r)) => l.set(&index, r),
            (U64(l), U64(r)) => l.set(&index, r),
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
                let mut data: Vec<bool> = b.into();
                let data: Vec<u8> = data.drain(..).map(|i| if i { 1u8 } else { 0u8 }).collect();
                data.into()
            }
            C32(c) => {
                let mut data: Vec<Complex<f32>> = c.into();
                let data: Vec<Vec<u8>> = data
                    .drain(..)
                    .map(|b| [b.re.to_be_bytes(), b.im.to_be_bytes()].concat())
                    .collect();
                data.into_iter().flatten().collect::<Vec<u8>>().into()
            }
            C64(c) => {
                let mut data: Vec<Complex<f64>> = c.into();
                let data: Vec<Vec<u8>> = data
                    .drain(..)
                    .map(|b| [b.re.to_be_bytes(), b.im.to_be_bytes()].concat())
                    .collect();
                data.into_iter().flatten().collect::<Vec<u8>>().into()
            }
            F32(f) => {
                let mut data: Vec<f32> = f.into();
                let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            F64(f) => {
                let mut data: Vec<f64> = f.into();
                let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            I16(i) => {
                let mut data: Vec<i16> = i.into();
                let data: Vec<[u8; 2]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            I32(i) => {
                let mut data: Vec<i32> = i.into();
                let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            I64(i) => {
                let mut data: Vec<i64> = i.into();
                let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            U8(b) => {
                let data: Vec<u8> = b.into();
                data.into()
            }
            U16(u) => {
                let mut data: Vec<u16> = u.into();
                let data: Vec<[u8; 2]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            U32(u) => {
                let mut data: Vec<u32> = u.into();
                let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
            U64(u) => {
                let mut data: Vec<u64> = u.into();
                let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
                data[..].concat().into()
            }
        }
    }
}

impl From<Vec<bool>> for ChunkData {
    fn from(b: Vec<bool>) -> ChunkData {
        let data = af::Array::new(&b, dim4(b.len()));
        ChunkData::Bool(data.into())
    }
}

impl From<Vec<Complex<f32>>> for ChunkData {
    fn from(c: Vec<Complex<f32>>) -> ChunkData {
        let data = af::Array::new(&c, dim4(c.len()));
        ChunkData::C32(data.into())
    }
}

impl From<Vec<Complex<f64>>> for ChunkData {
    fn from(c: Vec<Complex<f64>>) -> ChunkData {
        let data = af::Array::new(&c, dim4(c.len()));
        ChunkData::C64(data.into())
    }
}

impl From<Vec<f32>> for ChunkData {
    fn from(f: Vec<f32>) -> ChunkData {
        let data = af::Array::new(&f, dim4(f.len()));
        ChunkData::F32(data.into())
    }
}

impl From<Vec<f64>> for ChunkData {
    fn from(f: Vec<f64>) -> ChunkData {
        let data = af::Array::new(&f, dim4(f.len()));
        ChunkData::F64(data.into())
    }
}

impl From<Vec<i16>> for ChunkData {
    fn from(i: Vec<i16>) -> ChunkData {
        let data = af::Array::new(&i, dim4(i.len()));
        ChunkData::I16(data.into())
    }
}

impl From<Vec<i32>> for ChunkData {
    fn from(i: Vec<i32>) -> ChunkData {
        let data = af::Array::new(&i, dim4(i.len()));
        ChunkData::I32(data.into())
    }
}

impl From<Vec<i64>> for ChunkData {
    fn from(i: Vec<i64>) -> ChunkData {
        let data = af::Array::new(&i, dim4(i.len()));
        ChunkData::I64(data.into())
    }
}

impl From<Vec<u8>> for ChunkData {
    fn from(u: Vec<u8>) -> ChunkData {
        let data = af::Array::new(&u, dim4(u.len()));
        ChunkData::U8(data.into())
    }
}

impl From<Vec<u16>> for ChunkData {
    fn from(u: Vec<u16>) -> ChunkData {
        let data = af::Array::new(&u, dim4(u.len()));
        ChunkData::U16(data.into())
    }
}

impl From<Vec<u32>> for ChunkData {
    fn from(u: Vec<u32>) -> ChunkData {
        let data = af::Array::new(&u, dim4(u.len()));
        ChunkData::U32(data.into())
    }
}

impl From<Vec<u64>> for ChunkData {
    fn from(u: Vec<u64>) -> ChunkData {
        let data = af::Array::new(&u, dim4(u.len()));
        ChunkData::U64(data.into())
    }
}

impl From<ChunkData> for Vec<Number> {
    fn from(chunk: ChunkData) -> Vec<Number> {
        use ChunkData::*;
        match chunk {
            Bool(b) => b.into(),
            C32(c) => c.into(),
            C64(c) => c.into(),
            F32(f) => f.into(),
            F64(f) => f.into(),
            I16(i) => i.into(),
            I32(i) => i.into(),
            I64(i) => i.into(),
            U8(u) => u.into(),
            U16(u) => u.into(),
            U32(u) => u.into(),
            U64(u) => u.into(),
        }
    }
}

fn dim4(size: usize) -> af::Dim4 {
    af::Dim4::new(&[size as u64, 1, 1, 1])
}
