use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::{Add, Mul};

use arrayfire as af;
use bytes::Bytes;

use crate::error;
use crate::state::file::block::BlockData;
use crate::value::class::{ComplexType, FloatType, IntType, NumberType, UIntType};
use crate::value::{Boolean, Complex, Float, Int, Number, TCResult, UInt};

const BATCH: bool = true;

pub trait ArrayInstance {
    type DType: af::HasAfEnum;

    fn af(&'_ self) -> &'_ af::Array<Self::DType>;

    fn len(&self) -> usize {
        self.af().elements()
    }

    fn as_type<T: af::HasAfEnum>(&self) -> ArrayExt<T> {
        ArrayExt(self.af().cast())
    }

    fn get(&self, index: af::Indexer) -> ArrayExt<Self::DType> {
        ArrayExt(af::index_gen(self.af(), index))
    }

    fn set<T: ArrayInstance<DType = Self::DType>>(&self, index: &af::Indexer, other: &T) {
        af::assign_gen(self.af(), index, other.af());
    }

    fn set_at(&self, offset: usize, value: Self::DType) {
        let seq = af::Seq::new(offset as f32, offset as f32, 1.0f32);
        af::assign_seq(self.af(), &[seq], &af::Array::new(&[value], dim4(1)));
    }
}

#[derive(Clone)]
pub struct ArrayExt<T: af::HasAfEnum>(af::Array<T>);

impl<T: af::HasAfEnum> ArrayInstance for ArrayExt<T> {
    type DType = T;

    fn af(&'_ self) -> &'_ af::Array<T> {
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

impl From<ArrayExt<bool>> for Vec<Number> {
    fn from(array: ArrayExt<bool>) -> Vec<Number> {
        let array: Vec<bool> = array.into();
        let array: Vec<Boolean> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<bool>> for Bytes {
    fn from(array: ArrayExt<bool>) -> Bytes {
        let mut data: Vec<bool> = array.into();
        let data: Vec<u8> = data.drain(..).map(|i| if i { 1u8 } else { 0u8 }).collect();
        data.into()
    }
}

impl From<ArrayExt<num::Complex<f32>>> for Vec<Number> {
    fn from(array: ArrayExt<num::Complex<f32>>) -> Vec<Number> {
        let array: Vec<num::Complex<f32>> = array.into();
        let array: Vec<Complex> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<num::Complex<f32>>> for Bytes {
    fn from(array: ArrayExt<num::Complex<f32>>) -> Bytes {
        let mut data: Vec<num::Complex<f32>> = array.into();
        let data: Vec<Vec<u8>> = data
            .drain(..)
            .map(|b| [b.re.to_be_bytes(), b.im.to_be_bytes()].concat())
            .collect();
        data.into_iter().flatten().collect::<Vec<u8>>().into()
    }
}

impl From<ArrayExt<num::Complex<f64>>> for Vec<Number> {
    fn from(array: ArrayExt<num::Complex<f64>>) -> Vec<Number> {
        let array: Vec<num::Complex<f64>> = array.into();
        let array: Vec<Complex> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<num::Complex<f64>>> for Bytes {
    fn from(array: ArrayExt<num::Complex<f64>>) -> Bytes {
        let mut data: Vec<num::Complex<f64>> = array.into();
        let data: Vec<Vec<u8>> = data
            .drain(..)
            .map(|b| [b.re.to_be_bytes(), b.im.to_be_bytes()].concat())
            .collect();
        data.into_iter().flatten().collect::<Vec<u8>>().into()
    }
}

impl From<ArrayExt<f32>> for Vec<Number> {
    fn from(array: ArrayExt<f32>) -> Vec<Number> {
        let array: Vec<f32> = array.into();
        let array: Vec<Float> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<f32>> for Bytes {
    fn from(array: ArrayExt<f32>) -> Bytes {
        let mut data: Vec<f32> = array.into();
        let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
        data[..].concat().into()
    }
}

impl From<ArrayExt<f64>> for Vec<Number> {
    fn from(array: ArrayExt<f64>) -> Vec<Number> {
        let array: Vec<f64> = array.into();
        let array: Vec<Float> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<f64>> for Bytes {
    fn from(array: ArrayExt<f64>) -> Bytes {
        let mut data: Vec<f64> = array.into();
        let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
        data[..].concat().into()
    }
}

impl From<ArrayExt<i16>> for Vec<Number> {
    fn from(array: ArrayExt<i16>) -> Vec<Number> {
        let array: Vec<i16> = array.into();
        let array: Vec<Int> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<i16>> for Bytes {
    fn from(array: ArrayExt<i16>) -> Bytes {
        let mut data: Vec<i16> = array.into();
        let data: Vec<[u8; 2]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
        data[..].concat().into()
    }
}

impl From<ArrayExt<i32>> for Vec<Number> {
    fn from(array: ArrayExt<i32>) -> Vec<Number> {
        let array: Vec<i32> = array.into();
        let array: Vec<Int> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<i32>> for Bytes {
    fn from(array: ArrayExt<i32>) -> Bytes {
        let mut data: Vec<i32> = array.into();
        let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
        data[..].concat().into()
    }
}

impl From<ArrayExt<i64>> for Vec<Number> {
    fn from(array: ArrayExt<i64>) -> Vec<Number> {
        let array: Vec<i64> = array.into();
        let array: Vec<Int> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<i64>> for Bytes {
    fn from(array: ArrayExt<i64>) -> Bytes {
        let mut data: Vec<i64> = array.into();
        let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
        data[..].concat().into()
    }
}

impl From<ArrayExt<u8>> for Vec<Number> {
    fn from(array: ArrayExt<u8>) -> Vec<Number> {
        let array: Vec<u8> = array.into();
        let array: Vec<UInt> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<u8>> for Bytes {
    fn from(array: ArrayExt<u8>) -> Bytes {
        let data: Vec<u8> = array.into();
        data.into()
    }
}

impl From<ArrayExt<u16>> for Vec<Number> {
    fn from(array: ArrayExt<u16>) -> Vec<Number> {
        let array: Vec<u16> = array.into();
        let array: Vec<UInt> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<u16>> for Bytes {
    fn from(array: ArrayExt<u16>) -> Bytes {
        let mut data: Vec<u16> = array.into();
        let data: Vec<[u8; 2]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
        data[..].concat().into()
    }
}

impl From<ArrayExt<u32>> for Vec<Number> {
    fn from(array: ArrayExt<u32>) -> Vec<Number> {
        let array: Vec<u32> = array.into();
        let array: Vec<UInt> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<u32>> for Bytes {
    fn from(array: ArrayExt<u32>) -> Bytes {
        let mut data: Vec<u32> = array.into();
        let data: Vec<[u8; 4]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
        data[..].concat().into()
    }
}

impl From<ArrayExt<u64>> for Vec<Number> {
    fn from(array: ArrayExt<u64>) -> Vec<Number> {
        let array: Vec<u64> = array.into();
        let array: Vec<UInt> = vec_into(array);
        vec_into(array)
    }
}

impl From<ArrayExt<u64>> for Bytes {
    fn from(array: ArrayExt<u64>) -> Bytes {
        let mut data: Vec<u64> = array.into();
        let data: Vec<[u8; 8]> = data.drain(..).map(|b| b.to_be_bytes()).collect();
        data[..].concat().into()
    }
}

impl<T: af::HasAfEnum> From<af::Array<T>> for ArrayExt<T> {
    fn from(array: af::Array<T>) -> ArrayExt<T> {
        ArrayExt(array)
    }
}

impl<T: af::HasAfEnum> From<Vec<T>> for ArrayExt<T> {
    fn from(values: Vec<T>) -> ArrayExt<T> {
        let dim = dim4(values.len());
        ArrayExt(af::Array::new(&values, dim))
    }
}

impl TryFrom<Bytes> for ArrayExt<bool> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<ArrayExt<bool>> {
        let mut array = Vec::with_capacity(data.len());
        for byte in data {
            match byte as u8 {
                0 => array.push(false),
                1 => array.push(false),
                other => return Err(err_corrupt(format!("invalid boolean: {}", other))),
            }
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<num::Complex<f32>> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 8 != 0 {
            return Err(err_corrupt("invalid byte sequence for Complex<f32>"));
        }

        let mut array = Vec::with_capacity(data.len() / 8);
        for c in data[..].chunks_exact(8) {
            let re = f32::from_be_bytes(c[0..4].try_into().unwrap());
            let im = f32::from_be_bytes(c[4..8].try_into().unwrap());
            array.push(num::Complex::new(re, im));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<num::Complex<f64>> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 16 != 0 {
            return Err(err_corrupt("invalid byte sequence for Complex<f64>"));
        }

        let mut array = Vec::with_capacity(data.len() / 16);
        for c in data[..].chunks_exact(8) {
            let re = f64::from_be_bytes(c[0..8].try_into().unwrap());
            let im = f64::from_be_bytes(c[8..16].try_into().unwrap());
            array.push(num::Complex::new(re, im));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<f32> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 4 != 0 {
            return Err(err_corrupt(
                "invalid byte sequence for 32-bit floating point",
            ));
        }

        let mut array = Vec::with_capacity(data.len() / 4);
        for f in data[..].chunks_exact(4) {
            array.push(f32::from_be_bytes(f.try_into().unwrap()));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<f64> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 8 != 0 {
            return Err(err_corrupt(
                "invalid byte sequence for 64-bit floating point",
            ));
        }

        let mut array = Vec::with_capacity(data.len() / 8);
        for f in data[..].chunks_exact(8) {
            array.push(f64::from_be_bytes(f.try_into().unwrap()));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<i16> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 2 != 0 {
            return Err(err_corrupt(
                "invalid byte sequence for 16-bit signed integer",
            ));
        }

        let mut array = Vec::with_capacity(data.len() / 2);
        for f in data[..].chunks_exact(2) {
            array.push(i16::from_be_bytes(f.try_into().unwrap()));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<i32> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 4 != 0 {
            return Err(err_corrupt(
                "invalid byte sequence for 32-bit signed integer",
            ));
        }

        let mut array = Vec::with_capacity(data.len() / 4);
        for f in data[..].chunks_exact(4) {
            array.push(i32::from_be_bytes(f.try_into().unwrap()));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<i64> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 8 != 0 {
            return Err(err_corrupt(
                "invalid byte sequence for 64-bit signed integer",
            ));
        }

        let mut array = Vec::with_capacity(data.len() / 8);
        for f in data[..].chunks_exact(8) {
            array.push(i64::from_be_bytes(f.try_into().unwrap()));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<u8> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        Ok(af::Array::new(&data[..], dim4(data.len())).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<u16> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 2 != 0 {
            return Err(err_corrupt(
                "invalid byte sequence for 16-bit unsigned integer",
            ));
        }

        let mut array = Vec::with_capacity(data.len() / 2);
        for f in data[..].chunks_exact(2) {
            array.push(u16::from_be_bytes(f.try_into().unwrap()));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<u32> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 4 != 0 {
            return Err(err_corrupt(
                "invalid byte sequence for 32-bit unsigned integer",
            ));
        }

        let mut array = Vec::with_capacity(data.len() / 4);
        for f in data[..].chunks_exact(4) {
            array.push(u32::from_be_bytes(f.try_into().unwrap()));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl TryFrom<Bytes> for ArrayExt<u64> {
    type Error = error::TCError;

    fn try_from(data: Bytes) -> TCResult<Self> {
        if data.len() % 8 != 0 {
            return Err(err_corrupt(
                "invalid byte sequence for 64-bit signed integer",
            ));
        }

        let mut array = Vec::with_capacity(data.len() / 8);
        for f in data[..].chunks_exact(8) {
            array.push(u64::from_be_bytes(f.try_into().unwrap()));
        }
        let dim = dim4(array.len());
        Ok(af::Array::new(&array, dim).into())
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Add for ArrayExt<T>
where
    <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
    <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
    <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn add(self, other: Self) -> Self::Output {
        ArrayExt(af::add(&self.0, &other.0, BATCH))
    }
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T> + af::Convertable<OutType = T>> Mul for ArrayExt<T>
where
    <T as af::ImplicitPromote<T>>::Output: af::HasAfEnum,
    <T as af::Convertable>::OutType: af::ImplicitPromote<<T as af::Convertable>::OutType>,
    <<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output: af::HasAfEnum, {

    type Output = ArrayExt<<<T as af::Convertable>::OutType as af::ImplicitPromote<<T as af::Convertable>::OutType>>::Output>;

    fn mul(self, other: Self) -> Self::Output {
        ArrayExt(af::mul(&self.0, &other.0, BATCH))
    }
}

pub trait ArrayInstanceAbs: ArrayInstance {
    type AbsValue: af::HasAfEnum;

    fn abs(&self) -> ArrayExt<Self::AbsValue>;
}

impl ArrayInstanceAbs for ArrayExt<num::Complex<f32>> {
    type AbsValue = f32;

    fn abs(&self) -> ArrayExt<f32> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<num::Complex<f64>> {
    type AbsValue = f64;

    fn abs(&self) -> ArrayExt<f64> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<f32> {
    type AbsValue = f32;

    fn abs(&self) -> ArrayExt<f32> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<f64> {
    type AbsValue = f64;

    fn abs(&self) -> ArrayExt<f64> {
        ArrayExt(af::abs(self.af()))
    }
}

impl ArrayInstanceAbs for ArrayExt<i16> {
    type AbsValue = i16;

    fn abs(&self) -> ArrayExt<i16> {
        ArrayExt(af::abs(self.af()).cast())
    }
}

impl ArrayInstanceAbs for ArrayExt<i32> {
    type AbsValue = i32;

    fn abs(&self) -> ArrayExt<i32> {
        ArrayExt(af::abs(self.af()).cast())
    }
}

impl ArrayInstanceAbs for ArrayExt<i64> {
    type AbsValue = i64;

    fn abs(&self) -> ArrayExt<i64> {
        ArrayExt(af::abs(self.af()).cast())
    }
}

pub trait ArrayInstanceAnyAll: ArrayInstance {
    fn all(&self) -> bool {
        af::all_true_all(self.af()).0 > 0.0f64
    }

    fn any(&self) -> bool {
        af::any_true_all(self.af()).0 > 0.0f64
    }
}

impl ArrayInstanceAnyAll for ArrayExt<bool> {}
impl ArrayInstanceAnyAll for ArrayExt<f32> {}
impl ArrayInstanceAnyAll for ArrayExt<f64> {}
impl ArrayInstanceAnyAll for ArrayExt<i16> {}
impl ArrayInstanceAnyAll for ArrayExt<i32> {}
impl ArrayInstanceAnyAll for ArrayExt<i64> {}
impl ArrayInstanceAnyAll for ArrayExt<u8> {}
impl ArrayInstanceAnyAll for ArrayExt<u16> {}
impl ArrayInstanceAnyAll for ArrayExt<u32> {}
impl ArrayInstanceAnyAll for ArrayExt<u64> {}

impl ArrayInstanceAnyAll for ArrayExt<num::Complex<f32>> {
    fn all(&self) -> bool {
        let all = af::all_true_all(self.af());
        all.0 > 0.0f64 && all.1 > 0.0f64
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.af());
        any.0 > 0.0f64 || any.1 > 0.0f64
    }
}

impl ArrayInstanceAnyAll for ArrayExt<num::Complex<f64>> {
    fn all(&self) -> bool {
        let all = af::all_true_all(self.af());
        all.0 > 0.0f64 && all.1 > 0.0f64
    }

    fn any(&self) -> bool {
        let any = af::any_true_all(self.af());
        any.0 > 0.0f64 || any.1 > 0.0f64
    }
}

trait ArrayInstanceNot: ArrayInstance {
    fn not(&self) -> ArrayExt<bool>;
}

impl ArrayInstanceNot for ArrayExt<bool> {
    fn not(&self) -> ArrayExt<bool> {
        ArrayExt(!self.af())
    }
}

trait ArrayInstanceBool: ArrayInstance {
    fn and(&self, other: &Self) -> ArrayExt<bool>;

    fn or(&self, other: &Self) -> ArrayExt<bool>;

    fn xor(&self, other: &Self) -> ArrayExt<bool>;
}

impl<T: af::HasAfEnum> ArrayInstanceBool for ArrayExt<T> {
    fn and(&self, other: &Self) -> ArrayExt<bool> {
        let l: af::Array<bool> = self.af().cast();
        let r: af::Array<bool> = other.af().cast();
        ArrayExt(af::and(&l, &r, BATCH))
    }

    fn or(&self, other: &Self) -> ArrayExt<bool> {
        let l: af::Array<bool> = self.af().cast();
        let r: af::Array<bool> = other.af().cast();
        ArrayExt(af::and(&l, &r, BATCH))
    }

    fn xor(&self, other: &Self) -> ArrayExt<bool> {
        let l: af::Array<bool> = self.af().cast();
        let r: af::Array<bool> = other.af().cast();

        let one = af::or(&l, &r, BATCH);
        let not_both = !(&af::and(&l, &r, BATCH));
        let one_and_not_both = af::and(&one, &not_both, BATCH);
        ArrayExt(one_and_not_both.cast())
    }
}

trait ArrayInstanceCompare {
    fn eq(&self, other: &Self) -> ArrayExt<bool>;

    fn gt(&self, other: &Self) -> ArrayExt<bool>;

    fn gte(&self, other: &Self) -> ArrayExt<bool>;

    fn lt(&self, other: &Self) -> ArrayExt<bool>;

    fn lte(&self, other: &Self) -> ArrayExt<bool>;
}

impl<T: af::HasAfEnum + af::ImplicitPromote<T>> ArrayInstanceCompare for ArrayExt<T> {
    fn eq(&self, other: &Self) -> ArrayExt<bool> {
        af::eq(self.af(), other.af(), BATCH).into()
    }

    fn gt(&self, other: &Self) -> ArrayExt<bool> {
        af::gt(self.af(), other.af(), BATCH).into()
    }

    fn gte(&self, other: &Self) -> ArrayExt<bool> {
        af::ge(self.af(), other.af(), BATCH).into()
    }

    fn lt(&self, other: &Self) -> ArrayExt<bool> {
        af::lt(self.af(), other.af(), BATCH).into()
    }

    fn lte(&self, other: &Self) -> ArrayExt<bool> {
        af::le(self.af(), other.af(), BATCH).into()
    }
}

trait ArrayInstanceReduce: ArrayInstance {
    type Product: af::HasAfEnum;
    type Sum: af::HasAfEnum;

    fn product(&self) -> Self::Product;

    fn sum(&self) -> Self::Sum;
}

impl ArrayInstanceReduce for ArrayExt<bool> {
    type Sum = u64;
    type Product = u64;

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }
}

impl ArrayInstanceReduce for ArrayExt<num::Complex<f32>> {
    type Sum = num::Complex<f64>;
    type Product = num::Complex<f64>;

    fn sum(&self) -> Self::Sum {
        let sum = af::sum_all(self.af());
        num::Complex::new(sum.0, sum.1)
    }

    fn product(&self) -> Self::Product {
        let product = af::product_all(self.af());
        num::Complex::new(product.0, product.1)
    }
}

impl ArrayInstanceReduce for ArrayExt<num::Complex<f64>> {
    type Sum = num::Complex<f64>;
    type Product = num::Complex<f64>;

    fn sum(&self) -> Self::Sum {
        let sum = af::sum_all(self.af());
        num::Complex::new(sum.0, sum.1)
    }

    fn product(&self) -> Self::Product {
        let product = af::product_all(self.af());
        num::Complex::new(product.0, product.1)
    }
}

impl ArrayInstanceReduce for ArrayExt<f32> {
    type Sum = f64;
    type Product = f64;

    fn sum(&self) -> f64 {
        af::sum_all(self.af()).0
    }

    fn product(&self) -> f64 {
        af::product_all(self.af()).0
    }
}

impl ArrayInstanceReduce for ArrayExt<f64> {
    type Sum = f64;
    type Product = f64;

    fn sum(&self) -> f64 {
        af::sum_all(self.af()).0
    }

    fn product(&self) -> f64 {
        af::product_all(self.af()).0
    }
}

impl ArrayInstanceReduce for ArrayExt<i16> {
    type Sum = i64;
    type Product = i64;

    fn sum(&self) -> i64 {
        af::sum_all(self.af()).0 as i64
    }

    fn product(&self) -> i64 {
        af::product_all(self.af()).0 as i64
    }
}

impl ArrayInstanceReduce for ArrayExt<i32> {
    type Sum = i64;
    type Product = i64;

    fn sum(&self) -> i64 {
        af::sum_all(self.af()).0 as i64
    }

    fn product(&self) -> i64 {
        af::product_all(self.af()).0 as i64
    }
}

impl ArrayInstanceReduce for ArrayExt<i64> {
    type Sum = i64;
    type Product = i64;

    fn sum(&self) -> i64 {
        af::sum_all(self.af()).0 as i64
    }

    fn product(&self) -> i64 {
        af::product_all(self.af()).0 as i64
    }
}

impl ArrayInstanceReduce for ArrayExt<u8> {
    type Sum = u64;
    type Product = u64;

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }
}

impl ArrayInstanceReduce for ArrayExt<u16> {
    type Sum = u64;
    type Product = u64;

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }
}

impl ArrayInstanceReduce for ArrayExt<u32> {
    type Sum = u64;
    type Product = u64;

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }
}

impl ArrayInstanceReduce for ArrayExt<u64> {
    type Sum = u64;
    type Product = u64;

    fn sum(&self) -> u64 {
        af::sum_all(self.af()).0 as u64
    }

    fn product(&self) -> u64 {
        af::product_all(self.af()).0 as u64
    }
}

#[derive(Clone)]
pub enum Array {
    Bool(ArrayExt<bool>),
    C32(ArrayExt<num::Complex<f32>>),
    C64(ArrayExt<num::Complex<f64>>),
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

impl Array {
    fn af_cast<T: af::HasAfEnum>(&self) -> ArrayExt<T> {
        use Array::*;
        match self {
            Bool(b) => b.as_type(),
            C32(c) => c.as_type(),
            C64(c) => c.as_type(),
            F32(f) => f.as_type(),
            F64(f) => f.as_type(),
            I16(i) => i.as_type(),
            I32(i) => i.as_type(),
            I64(i) => i.as_type(),
            U8(u) => u.as_type(),
            U16(u) => u.as_type(),
            U32(u) => u.as_type(),
            U64(u) => u.as_type(),
        }
    }

    pub fn constant(value: Number, len: usize) -> Array {
        let dim = dim4(len);

        use Array::*;
        match value {
            Number::Bool(b) => {
                let b: bool = b.into();
                Bool(af::constant(b, dim).into())
            }
            Number::Complex(c) => match c {
                Complex::C32(c) => C32(af::constant(c, dim).into()),
                Complex::C64(c) => C64(af::constant(c, dim).into()),
            },
            Number::Float(f) => match f {
                Float::F32(f) => F32(af::constant(f, dim).into()),
                Float::F64(f) => F64(af::constant(f, dim).into()),
            },
            Number::Int(i) => match i {
                Int::I16(i) => I16(af::constant(i, dim).into()),
                Int::I32(i) => I32(af::constant(i, dim).into()),
                Int::I64(i) => I64(af::constant(i, dim).into()),
            },
            Number::UInt(u) => match u {
                UInt::U8(i) => U8(af::constant(i, dim).into()),
                UInt::U16(u) => U16(af::constant(u, dim).into()),
                UInt::U32(u) => U32(af::constant(u, dim).into()),
                UInt::U64(u) => U64(af::constant(u, dim).into()),
            },
        }
    }

    fn try_from_bytes(data: Bytes, dtype: NumberType) -> TCResult<Array> {
        use Array::*;
        use NumberType::*;
        match dtype {
            NumberType::Bool => Ok(Array::Bool(data.try_into()?)),
            Complex(c) => match c {
                ComplexType::C32 => Ok(C32(data.try_into()?)),
                ComplexType::C64 => Ok(C64(data.try_into()?)),
            },
            Float(f) => match f {
                FloatType::F32 => Ok(F32(data.try_into()?)),
                FloatType::F64 => Ok(F64(data.try_into()?)),
            },
            Int(i) => match i {
                IntType::I16 => Ok(I16(data.try_into()?)),
                IntType::I32 => Ok(I32(data.try_into()?)),
                IntType::I64 => Ok(I64(data.try_into()?)),
            },
            UInt(u) => match u {
                UIntType::U8 => Ok(U8(data.try_into()?)),
                UIntType::U16 => Ok(U16(data.try_into()?)),
                UIntType::U32 => Ok(U32(data.try_into()?)),
                UIntType::U64 => Ok(U64(data.try_into()?)),
            },
        }
    }

    pub fn try_from_values(values: Vec<Number>, dtype: NumberType) -> TCResult<Array> {
        use Array::*;
        let chunk = match dtype {
            NumberType::Bool => {
                let values: Vec<Boolean> = vec_try_into(values)?;
                Bool(vec_into(values).into())
            }
            NumberType::Complex(c) => {
                let values: Vec<Complex> = vec_try_into(values)?;
                match c {
                    ComplexType::C32 => C32(vec_try_into(values)?.into()),
                    ComplexType::C64 => C32(vec_try_into(values)?.into()),
                }
            }
            NumberType::Float(f) => {
                let values: Vec<Float> = vec_try_into(values)?;
                match f {
                    FloatType::F32 => F32(vec_try_into(values)?.into()),
                    FloatType::F64 => F32(vec_try_into(values)?.into()),
                }
            }
            NumberType::Int(i) => {
                let values: Vec<Int> = vec_try_into(values)?;
                match i {
                    IntType::I16 => I16(vec_try_into(values)?.into()),
                    IntType::I32 => I32(vec_try_into(values)?.into()),
                    IntType::I64 => I64(vec_into(values).into()),
                }
            }
            NumberType::UInt(u) => {
                let values: Vec<UInt> = vec_try_into(values)?;
                match u {
                    UIntType::U8 => U8(vec_try_into(values)?.into()),
                    UIntType::U16 => U16(vec_try_into(values)?.into()),
                    UIntType::U32 => U32(vec_try_into(values)?.into()),
                    UIntType::U64 => U64(vec_try_into(values)?.into()),
                }
            }
        };

        Ok(chunk)
    }

    pub fn dtype(&self) -> NumberType {
        use Array::*;
        match self {
            Bool(_) => NumberType::Bool,
            C32(_) => ComplexType::C32.into(),
            C64(_) => ComplexType::C32.into(),
            F32(_) => FloatType::F32.into(),
            F64(_) => FloatType::F32.into(),
            I16(_) => IntType::I16.into(),
            I32(_) => IntType::I32.into(),
            I64(_) => IntType::I64.into(),
            U8(_) => UIntType::U16.into(),
            U16(_) => UIntType::U16.into(),
            U32(_) => UIntType::U32.into(),
            U64(_) => UIntType::U64.into(),
        }
    }

    pub fn into_type(self, dtype: NumberType) -> TCResult<Array> {
        use ComplexType::*;
        use FloatType::*;
        use IntType::*;
        use NumberType::*;
        use UIntType::*;
        let cast = match dtype {
            Bool => Self::Bool(self.af_cast()),
            Complex(ct) => match ct {
                C32 => Self::C32(self.af_cast()),
                C64 => Self::C64(self.af_cast()),
            },
            Float(ft) => match ft {
                F32 => Self::F32(self.af_cast()),
                F64 => Self::F64(self.af_cast()),
            },
            Int(it) => match it {
                I16 => Self::I16(self.af_cast()),
                I32 => Self::I32(self.af_cast()),
                I64 => Self::I64(self.af_cast()),
            },
            UInt(ut) => match ut {
                U8 => Self::U8(self.af_cast()),
                U16 => Self::U16(self.af_cast()),
                U32 => Self::U32(self.af_cast()),
                U64 => Self::U64(self.af_cast()),
            },
        };
        Ok(cast)
    }

    pub fn into_values(self) -> Vec<Number> {
        use Array::*;
        match self {
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

    pub fn abs(&self) -> TCResult<Array> {
        use Array::*;
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
        use Array::*;
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
        use Array::*;
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

    pub fn add(self, other: Array) -> TCResult<Array> {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Ok(Bool(l.or(&r))),
            (C32(l), C32(r)) => Ok(C32(l + r)),
            (C64(l), C64(r)) => Ok(C64(l + r)),
            (F32(l), F32(r)) => Ok(F32(l + r)),
            (F64(l), F64(r)) => Ok(F64(l + r)),
            (I16(l), I16(r)) => Ok(I16(l + r)),
            (I32(l), I32(r)) => Ok(I32(l + r)),
            (I64(l), I64(r)) => Ok(I64(l + r)),
            (U8(l), U8(r)) => Ok(U8(l + r)),
            (U16(l), U16(r)) => Ok(U16(l + r)),
            (U32(l), U32(r)) => Ok(U32(l + r)),
            (U64(l), U64(r)) => Ok(U64(l + r)),
            (l, r) => Err(error::internal(format!("Tried to add {} and {}", l, r))),
        }
    }

    pub fn and(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.af_cast();
        let that: ArrayExt<bool> = other.af_cast();
        Array::Bool(this.and(&that))
    }

    pub fn eq(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.eq(&other.af_cast())),
            C32(l) => Bool(l.eq(&other.af_cast())),
            C64(l) => Bool(l.eq(&other.af_cast())),
            F32(l) => Bool(l.eq(&other.af_cast())),
            F64(l) => Bool(l.eq(&other.af_cast())),
            I16(l) => Bool(l.eq(&other.af_cast())),
            I32(l) => Bool(l.eq(&other.af_cast())),
            I64(l) => Bool(l.eq(&other.af_cast())),
            U8(l) => Bool(l.eq(&other.af_cast())),
            U16(l) => Bool(l.eq(&other.af_cast())),
            U32(l) => Bool(l.eq(&other.af_cast())),
            U64(l) => Bool(l.eq(&other.af_cast())),
        }
    }

    pub fn gt(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.gt(&other.af_cast())),
            C32(l) => Bool(l.gt(&other.af_cast())),
            C64(l) => Bool(l.gt(&other.af_cast())),
            F32(l) => Bool(l.gt(&other.af_cast())),
            F64(l) => Bool(l.gt(&other.af_cast())),
            I16(l) => Bool(l.gt(&other.af_cast())),
            I32(l) => Bool(l.gt(&other.af_cast())),
            I64(l) => Bool(l.gt(&other.af_cast())),
            U8(l) => Bool(l.gt(&other.af_cast())),
            U16(l) => Bool(l.gt(&other.af_cast())),
            U32(l) => Bool(l.gt(&other.af_cast())),
            U64(l) => Bool(l.gt(&other.af_cast())),
        }
    }

    pub fn gte(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.gte(&other.af_cast())),
            C32(l) => Bool(l.gte(&other.af_cast())),
            C64(l) => Bool(l.gte(&other.af_cast())),
            F32(l) => Bool(l.gte(&other.af_cast())),
            F64(l) => Bool(l.gte(&other.af_cast())),
            I16(l) => Bool(l.gte(&other.af_cast())),
            I32(l) => Bool(l.gte(&other.af_cast())),
            I64(l) => Bool(l.gte(&other.af_cast())),
            U8(l) => Bool(l.gte(&other.af_cast())),
            U16(l) => Bool(l.gte(&other.af_cast())),
            U32(l) => Bool(l.gte(&other.af_cast())),
            U64(l) => Bool(l.gte(&other.af_cast())),
        }
    }

    pub fn lt(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.lt(&other.af_cast())),
            C32(l) => Bool(l.lt(&other.af_cast())),
            C64(l) => Bool(l.lt(&other.af_cast())),
            F32(l) => Bool(l.lt(&other.af_cast())),
            F64(l) => Bool(l.lt(&other.af_cast())),
            I16(l) => Bool(l.lt(&other.af_cast())),
            I32(l) => Bool(l.lt(&other.af_cast())),
            I64(l) => Bool(l.lt(&other.af_cast())),
            U8(l) => Bool(l.lt(&other.af_cast())),
            U16(l) => Bool(l.lt(&other.af_cast())),
            U32(l) => Bool(l.lt(&other.af_cast())),
            U64(l) => Bool(l.lt(&other.af_cast())),
        }
    }

    pub fn lte(&self, other: &Array) -> Array {
        use Array::*;
        match self {
            Bool(l) => Bool(l.lte(&other.af_cast())),
            C32(l) => Bool(l.lte(&other.af_cast())),
            C64(l) => Bool(l.lte(&other.af_cast())),
            F32(l) => Bool(l.lte(&other.af_cast())),
            F64(l) => Bool(l.lte(&other.af_cast())),
            I16(l) => Bool(l.lte(&other.af_cast())),
            I32(l) => Bool(l.lte(&other.af_cast())),
            I64(l) => Bool(l.lte(&other.af_cast())),
            U8(l) => Bool(l.lte(&other.af_cast())),
            U16(l) => Bool(l.lte(&other.af_cast())),
            U32(l) => Bool(l.lte(&other.af_cast())),
            U64(l) => Bool(l.lte(&other.af_cast())),
        }
    }

    pub fn multiply(self, other: Array) -> TCResult<Array> {
        use Array::*;
        match (self, other) {
            (Bool(l), Bool(r)) => Ok(Bool(l.and(&r))),
            (C32(l), C32(r)) => Ok(C32(l * r)),
            (C64(l), C64(r)) => Ok(C64(l * r)),
            (F32(l), F32(r)) => Ok(F32(l * r)),
            (F64(l), F64(r)) => Ok(F64(l * r)),
            (I16(l), I16(r)) => Ok(I16(l * r)),
            (I32(l), I32(r)) => Ok(I32(l * r)),
            (I64(l), I64(r)) => Ok(I64(l * r)),
            (U8(l), U8(r)) => Ok(U8(l * r)),
            (U16(l), U16(r)) => Ok(U16(l * r)),
            (U32(l), U32(r)) => Ok(U32(l * r)),
            (U64(l), U64(r)) => Ok(U64(l * r)),
            (l, r) => Err(error::internal(format!(
                "Tried to multiply {} and {}",
                l, r
            ))),
        }
    }

    pub fn not(&self) -> Array {
        let this: ArrayExt<bool> = self.af_cast();
        Array::Bool(this.not())
    }

    pub fn or(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.af_cast();
        let that: ArrayExt<bool> = other.af_cast();
        Array::Bool(this.or(&that))
    }

    pub fn product(&self) -> Number {
        use Array::*;
        match self {
            Bool(b) => Number::UInt(b.product().into()),
            C32(c) => Number::Complex(c.product().into()),
            C64(c) => Number::Complex(c.product().into()),
            F32(f) => Number::Float(f.product().into()),
            F64(f) => Number::Float(f.product().into()),
            I16(i) => Number::Int(i.product().into()),
            I32(i) => Number::Int(i.product().into()),
            I64(i) => Number::Int(i.product().into()),
            U8(u) => Number::UInt(u.product().into()),
            U16(u) => Number::UInt(u.product().into()),
            U32(u) => Number::UInt(u.product().into()),
            U64(u) => Number::UInt(u.product().into()),
        }
    }

    pub fn sum(&self) -> Number {
        use Array::*;
        match self {
            Bool(b) => Number::UInt(b.sum().into()),
            C32(c) => Number::Complex(c.sum().into()),
            C64(c) => Number::Complex(c.sum().into()),
            F32(f) => Number::Float(f.sum().into()),
            F64(f) => Number::Float(f.sum().into()),
            I16(i) => Number::Int(i.sum().into()),
            I32(i) => Number::Int(i.sum().into()),
            I64(i) => Number::Int(i.sum().into()),
            U8(u) => Number::UInt(u.sum().into()),
            U16(u) => Number::UInt(u.sum().into()),
            U32(u) => Number::UInt(u.sum().into()),
            U64(u) => Number::UInt(u.sum().into()),
        }
    }

    pub fn len(&self) -> usize {
        use Array::*;
        match self {
            Bool(b) => b.len(),
            C32(c) => c.len(),
            C64(c) => c.len(),
            F32(f) => f.len(),
            F64(f) => f.len(),
            I16(i) => i.len(),
            I32(i) => i.len(),
            I64(i) => i.len(),
            U8(u) => u.len(),
            U16(u) => u.len(),
            U32(u) => u.len(),
            U64(u) => u.len(),
        }
    }

    pub fn get_value(&self, index: usize) -> Option<Number> {
        let seq = af::Seq::new(index as f64, index as f64, 1.0f64);
        let mut indexer = af::Indexer::default();
        indexer.set_index(&seq, 0, None);
        self.get_at(indexer).pop()
    }

    pub fn get(&self, index: af::Array<u64>) -> Vec<Number> {
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, None);
        self.get_at(indexer)
    }

    fn get_at(&self, index: af::Indexer) -> Vec<Number> {
        use Array::*;
        match self {
            Bool(b) => b.get(index).into(),
            C32(c) => c.get(index).into(),
            C64(c) => c.get(index).into(),
            F32(f) => f.get(index).into(),
            F64(f) => f.get(index).into(),
            I16(i) => i.get(index).into(),
            I32(i) => i.get(index).into(),
            I64(i) => i.get(index).into(),
            U8(i) => i.get(index).into(),
            U16(i) => i.get(index).into(),
            U32(i) => i.get(index).into(),
            U64(i) => i.get(index).into(),
        }
    }

    pub fn set(&mut self, index: af::Array<u64>, other: &Array) -> TCResult<()> {
        let mut indexer = af::Indexer::default();
        indexer.set_index(&index, 0, None);
        self.set_at(indexer, other)
    }

    pub fn set_value(&mut self, offset: usize, value: Number) -> TCResult<()> {
        use Array::*;
        match self {
            Bool(b) => {
                let value: Boolean = value.try_into()?;
                b.set_at(offset, value.try_into()?);
            }
            C32(c) => {
                let value: Complex = value.try_into()?;
                c.set_at(offset, value.try_into()?)
            }
            C64(c) => {
                let value: Complex = value.try_into()?;
                c.set_at(offset, value.try_into()?)
            }
            F32(f) => {
                let value: Float = value.try_into()?;
                f.set_at(offset, value.try_into()?)
            }
            F64(f) => {
                let value: Float = value.try_into()?;
                f.set_at(offset, value.try_into()?)
            }
            I16(i) => {
                let value: Int = value.try_into()?;
                i.set_at(offset, value.try_into()?)
            }
            I32(i) => {
                let value: Int = value.try_into()?;
                i.set_at(offset, value.try_into()?)
            }
            I64(i) => {
                let value: Int = value.try_into()?;
                i.set_at(offset, value.try_into()?)
            }
            U8(u) => {
                let value: UInt = value.try_into()?;
                u.set_at(offset, value.try_into()?)
            }
            U16(u) => {
                let value: UInt = value.try_into()?;
                u.set_at(offset, value.try_into()?)
            }
            U32(u) => {
                let value: UInt = value.try_into()?;
                u.set_at(offset, value.try_into()?)
            }
            U64(u) => {
                let value: UInt = value.try_into()?;
                u.set_at(offset, value.try_into()?)
            }
        }

        Ok(())
    }

    fn set_at(&mut self, index: af::Indexer, value: &Array) -> TCResult<()> {
        use Array::*;
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

    pub fn xor(&self, other: &Array) -> Array {
        let this: ArrayExt<bool> = self.af_cast();
        let that: ArrayExt<bool> = other.af_cast();
        Array::Bool(this.xor(&that))
    }
}

impl TryFrom<Bytes> for Array {
    type Error = error::TCError;

    fn try_from(mut data: Bytes) -> TCResult<Array> {
        let array = data.split_off(2);
        let dtype: NumberType = bincode::deserialize(&data)
            .map_err(|e| error::bad_request("Unable to deserialize Tensor array data type", e))?;

        use Array::*;
        use NumberType::*;
        let array = match dtype {
            NumberType::Bool => Array::Bool(array.try_into()?),
            Complex(ct) => match ct {
                ComplexType::C32 => C32(array.try_into()?),
                ComplexType::C64 => C64(array.try_into()?),
            },
            Float(ft) => match ft {
                FloatType::F32 => F32(array.try_into()?),
                FloatType::F64 => F64(array.try_into()?),
            },
            Int(it) => match it {
                IntType::I16 => I16(array.try_into()?),
                IntType::I32 => I32(array.try_into()?),
                IntType::I64 => I64(array.try_into()?),
            },
            UInt(ut) => match ut {
                UIntType::U8 => U8(array.try_into()?),
                UIntType::U16 => U16(array.try_into()?),
                UIntType::U32 => U32(array.try_into()?),
                UIntType::U64 => U64(array.try_into()?),
            },
        };

        Ok(array)
    }
}

impl From<Array> for Bytes {
    fn from(array: Array) -> Bytes {
        let dtype = array.dtype();

        use Array::*;
        let serialized: Bytes = match array {
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
        };

        let dtype = Bytes::from(bincode::serialize(&dtype).unwrap());
        assert!(dtype.len() == 2);
        Bytes::from([dtype, serialized].concat())
    }
}

impl BlockData for Array {}

impl From<Vec<bool>> for Array {
    fn from(b: Vec<bool>) -> Array {
        let data = af::Array::new(&b, dim4(b.len()));
        Array::Bool(data.into())
    }
}

impl From<Vec<num::Complex<f32>>> for Array {
    fn from(c: Vec<num::Complex<f32>>) -> Array {
        let data = af::Array::new(&c, dim4(c.len()));
        Array::C32(data.into())
    }
}

impl From<Vec<num::Complex<f64>>> for Array {
    fn from(c: Vec<num::Complex<f64>>) -> Array {
        let data = af::Array::new(&c, dim4(c.len()));
        Array::C64(data.into())
    }
}

impl From<Vec<f32>> for Array {
    fn from(f: Vec<f32>) -> Array {
        let data = af::Array::new(&f, dim4(f.len()));
        Array::F32(data.into())
    }
}

impl From<Vec<f64>> for Array {
    fn from(f: Vec<f64>) -> Array {
        let data = af::Array::new(&f, dim4(f.len()));
        Array::F64(data.into())
    }
}

impl From<Vec<i16>> for Array {
    fn from(i: Vec<i16>) -> Array {
        let data = af::Array::new(&i, dim4(i.len()));
        Array::I16(data.into())
    }
}

impl From<Vec<i32>> for Array {
    fn from(i: Vec<i32>) -> Array {
        let data = af::Array::new(&i, dim4(i.len()));
        Array::I32(data.into())
    }
}

impl From<Vec<i64>> for Array {
    fn from(i: Vec<i64>) -> Array {
        let data = af::Array::new(&i, dim4(i.len()));
        Array::I64(data.into())
    }
}

impl From<Vec<u8>> for Array {
    fn from(u: Vec<u8>) -> Array {
        let data = af::Array::new(&u, dim4(u.len()));
        Array::U8(data.into())
    }
}

impl From<Vec<u16>> for Array {
    fn from(u: Vec<u16>) -> Array {
        let data = af::Array::new(&u, dim4(u.len()));
        Array::U16(data.into())
    }
}

impl From<Vec<u32>> for Array {
    fn from(u: Vec<u32>) -> Array {
        let data = af::Array::new(&u, dim4(u.len()));
        Array::U32(data.into())
    }
}

impl From<Vec<u64>> for Array {
    fn from(u: Vec<u64>) -> Array {
        let data = af::Array::new(&u, dim4(u.len()));
        Array::U64(data.into())
    }
}

impl From<Array> for Vec<Number> {
    fn from(chunk: Array) -> Vec<Number> {
        use Array::*;
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

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dtype = match self {
            Self::Bool(_) => "Bool",
            Self::C32(_) => "Complex::C32",
            Self::C64(_) => "Complex::C64",
            Self::F32(_) => "Float::F32",
            Self::F64(_) => "Float::F64",
            Self::I16(_) => "Int::I16",
            Self::I32(_) => "Int::I32",
            Self::I64(_) => "Int::I64",
            Self::U8(_) => "UInt::U8",
            Self::U16(_) => "UInt::U16",
            Self::U32(_) => "UInt::U32",
            Self::U64(_) => "UInt::U64",
        };

        write!(f, "ArrayInstance<{}>", dtype)
    }
}

fn dim4(size: usize) -> af::Dim4 {
    af::Dim4::new(&[size as u64, 1, 1, 1])
}

fn vec_into<D, S: Into<D>>(mut source: Vec<S>) -> Vec<D> {
    source.drain(..).map(|i| i.into()).collect()
}

fn vec_try_into<D: TryFrom<S, Error = error::TCError>, S>(mut source: Vec<S>) -> TCResult<Vec<D>> {
    source.drain(..).map(|i| i.try_into()).collect()
}

fn err_corrupt<T: fmt::Display>(info: T) -> error::TCError {
    error::internal(format!(
        "Error! This Tensor's data has been corrupted on disk: {}",
        info
    ))
}
