use std::cmp::Ordering;
use std::convert::TryInto;
use std::fmt;
use std::ops::{Add, Mul};

use serde::{Deserialize, Serialize};

use crate::class::{Class, TCResult};
use crate::error;
use crate::value::class::{ValueClass, ValueInstance, ValueType};
use crate::value::link::TCPath;
use crate::value::{label, Link};

use super::instance::{Boolean, Complex, Float, Int, Number, UInt};

pub trait NumberClass: Class + ValueClass + Into<NumberType> + Ord + Send + Sync {
    type Instance: NumberInstance;

    fn size(self) -> usize;

    fn one(&self) -> <Self as NumberClass>::Instance {
        let b: Boolean = true.into();
        b.into()
    }

    fn zero(&self) -> <Self as NumberClass>::Instance {
        let b: Boolean = false.into();
        b.into()
    }
}

pub trait NumberInstance:
    ValueInstance
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sized
    + PartialOrd
    + From<Boolean>
    + Into<Number>
{
    type Abs: NumberInstance;
    type Class: NumberClass;

    fn into_type(
        self,
        dtype: <Self as NumberInstance>::Class,
    ) -> <<Self as NumberInstance>::Class as NumberClass>::Instance;

    fn abs(self) -> Self::Abs;

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn and(self, other: Self) -> Self
    where
        Self: CastInto<Boolean>,
    {
        let this: Boolean = self.cast_into();
        let that: Boolean = other.cast_into();
        this.and(that).into()
    }

    fn eq(self, other: Self) -> Self {
        let eq: Boolean = (self == other).into();
        eq.into()
    }

    fn lt(self, other: Self) -> Self {
        let lt: Boolean = PartialOrd::lt(&self, &other).into();
        lt.into()
    }

    fn lte(self, other: Self) -> Self {
        let lte: Boolean = (self <= other).into();
        lte.into()
    }

    fn gt(self, other: Self) -> Self {
        let gt: Boolean = PartialOrd::gt(&self, &other).into();
        gt.into()
    }

    fn gte(self, other: Self) -> Self {
        let gte: Boolean = (self >= other).into();
        gte.into()
    }

    fn multiply(self, other: Self) -> Self {
        self * other
    }

    fn ne(self, other: Self) -> Self {
        let ne: Boolean = (self != other).into();
        ne.into()
    }

    fn not(self) -> Self
    where
        Self: CastInto<Boolean>,
    {
        let this: Boolean = self.cast_into();
        this.not().into()
    }

    fn or(self, other: Self) -> Self
    where
        Self: CastInto<Boolean>,
    {
        let this: Boolean = self.cast_into();
        let that: Boolean = other.cast_into();
        this.or(that).into()
    }

    fn xor(self, other: Self) -> Self
    where
        Self: CastInto<Boolean>,
    {
        let this: Boolean = self.cast_into();
        let that: Boolean = other.cast_into();
        this.xor(that).into()
    }
}

pub trait CastFrom<T> {
    fn cast_from(value: T) -> Self;
}

pub trait CastInto<T> {
    fn cast_into(self) -> T;
}

impl<T> CastFrom<T> for T {
    fn cast_from(value: T) -> Self {
        value
    }
}

impl<T, F: CastFrom<T>> CastInto<F> for T {
    fn cast_into(self) -> F {
        F::cast_from(self)
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum ComplexType {
    C32,
    C64,
}

impl Class for ComplexType {
    type Instance = Complex;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let path = path.from_path(&Self::prefix())?;

        if path.is_empty() {
            Err(error::unsupported(
                "Complex number requires a size, complex/32 or complex/64",
            ))
        } else if path.len() > 1 {
            Err(error::not_found(path))
        } else {
            match path[0].as_str() {
                "32" => Ok(ComplexType::C32),
                "64" => Ok(ComplexType::C64),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPath {
        NumberType::prefix().join(label("complex").into())
    }
}

impl ValueClass for ComplexType {
    type Instance = Complex;

    fn get(path: &TCPath, value: Complex) -> TCResult<Complex> {
        if path.is_empty() {
            Ok(value)
        } else if path.len() == 1 {
            let dtype = match path[0].as_str() {
                "32" => ComplexType::C32,
                "64" => ComplexType::C64,
                _ => return Err(error::not_found(&path[0])),
            };

            Ok(value.into_type(dtype))
        } else {
            Err(error::not_found(path))
        }
    }

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for ComplexType {
    type Instance = Complex;

    fn size(self) -> usize {
        match self {
            Self::C32 => 8,
            Self::C64 => 16,
        }
    }
}

impl Ord for ComplexType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::C32, Self::C32) => Ordering::Equal,
            (Self::C64, Self::C64) => Ordering::Equal,

            (Self::C64, Self::C32) => Ordering::Greater,
            (Self::C32, Self::C64) => Ordering::Less,
        }
    }
}

impl PartialOrd for ComplexType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<ComplexType> for NumberType {
    fn from(ct: ComplexType) -> NumberType {
        Self::Complex(ct)
    }
}

impl From<ComplexType> for Link {
    fn from(ct: ComplexType) -> Link {
        let prefix = ComplexType::prefix();

        use ComplexType::*;
        match ct {
            C32 => prefix.join(label("32").into()).into(),
            C64 => prefix.join(label("64").into()).into(),
        }
    }
}

impl fmt::Display for ComplexType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::C32 => write!(f, "C32"),
            Self::C64 => write!(f, "C64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct BooleanType;

impl Class for BooleanType {
    type Instance = Boolean;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let path = path.from_path(&Self::prefix())?;

        if path.is_empty() {
            Ok(BooleanType)
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        NumberType::prefix().join(label("bool").into())
    }
}

impl ValueClass for BooleanType {
    type Instance = Boolean;

    fn get(path: &TCPath, value: Boolean) -> TCResult<Boolean> {
        if path.is_empty() {
            Ok(value)
        } else {
            Err(error::not_found(path))
        }
    }

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for BooleanType {
    type Instance = Boolean;

    fn size(self) -> usize {
        1
    }
}

impl Ord for BooleanType {
    fn cmp(&self, _other: &Self) -> Ordering {
        Ordering::Equal
    }
}

impl PartialOrd for BooleanType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<BooleanType> for NumberType {
    fn from(_bt: BooleanType) -> NumberType {
        NumberType::Bool
    }
}

impl From<BooleanType> for Link {
    fn from(_: BooleanType) -> Link {
        BooleanType::prefix().into()
    }
}

impl fmt::Display for BooleanType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Boolean")
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum FloatType {
    F32,
    F64,
}

impl Class for FloatType {
    type Instance = Float;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let path = path.from_path(&Self::prefix())?;

        if path.is_empty() {
            Err(error::unsupported(
                "Float requires a size, float/32 or float/64",
            ))
        } else if path.len() > 1 {
            Err(error::not_found(path))
        } else {
            match path[0].as_str() {
                "32" => Ok(FloatType::F32),
                "64" => Ok(FloatType::F64),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPath {
        NumberType::prefix().join(label("float").into())
    }
}

impl ValueClass for FloatType {
    type Instance = Float;

    fn get(path: &TCPath, value: Float) -> TCResult<Float> {
        if path.is_empty() {
            Ok(value)
        } else if path.len() == 1 {
            let dtype = match path[0].as_str() {
                "32" => FloatType::F32,
                "64" => FloatType::F64,
                _ => return Err(error::not_found(&path[0])),
            };

            Ok(value.into_type(dtype))
        } else {
            Err(error::not_found(path))
        }
    }

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for FloatType {
    type Instance = Float;

    fn size(self) -> usize {
        match self {
            FloatType::F32 => 4,
            FloatType::F64 => 8,
        }
    }
}

impl Ord for FloatType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::F32, Self::F32) => Ordering::Equal,
            (Self::F64, Self::F64) => Ordering::Equal,

            (Self::F64, Self::F32) => Ordering::Greater,
            (Self::F32, Self::F64) => Ordering::Less,
        }
    }
}

impl PartialOrd for FloatType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<FloatType> for NumberType {
    fn from(ft: FloatType) -> NumberType {
        NumberType::Float(ft)
    }
}

impl From<FloatType> for Link {
    fn from(ft: FloatType) -> Link {
        let prefix = FloatType::prefix();

        use FloatType::*;
        match ft {
            F32 => prefix.join(label("32").into()).into(),
            F64 => prefix.join(label("64").into()).into(),
        }
    }
}

impl fmt::Display for FloatType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use FloatType::*;
        match self {
            F32 => write!(f, "F32"),
            F64 => write!(f, "F64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum IntType {
    I16,
    I32,
    I64,
}

impl Class for IntType {
    type Instance = Int;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let path = path.from_path(&Self::prefix())?;

        if path.is_empty() {
            Err(error::unsupported(
                "Int requires a size, int/16 or int/32 or int/64",
            ))
        } else if path.len() > 1 {
            Err(error::not_found(path))
        } else {
            match path[0].as_str() {
                "16" => Ok(IntType::I16),
                "32" => Ok(IntType::I32),
                "64" => Ok(IntType::I64),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPath {
        NumberType::prefix().join(label("int").into())
    }
}

impl ValueClass for IntType {
    type Instance = Int;

    fn get(path: &TCPath, value: Int) -> TCResult<Int> {
        if path.is_empty() {
            Ok(value)
        } else if path.len() == 1 {
            let dtype = match path[0].as_str() {
                "16" => IntType::I16,
                "32" => IntType::I32,
                "64" => IntType::I64,
                _ => return Err(error::not_found(&path[0])),
            };

            Ok(value.into_type(dtype))
        } else {
            Err(error::not_found(path))
        }
    }

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for IntType {
    type Instance = Int;

    fn size(self) -> usize {
        match self {
            IntType::I16 => 2,
            IntType::I32 => 4,
            IntType::I64 => 8,
        }
    }
}

impl Ord for IntType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::I16, Self::I16) => Ordering::Equal,
            (Self::I32, Self::I32) => Ordering::Equal,
            (Self::I64, Self::I64) => Ordering::Equal,

            (Self::I64, _) => Ordering::Greater,
            (_, Self::I64) => Ordering::Less,
            (Self::I16, _) => Ordering::Less,
            (_, Self::I16) => Ordering::Greater,
        }
    }
}

impl PartialOrd for IntType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<IntType> for NumberType {
    fn from(it: IntType) -> NumberType {
        NumberType::Int(it)
    }
}

impl From<IntType> for Link {
    fn from(it: IntType) -> Link {
        let prefix = IntType::prefix();

        use IntType::*;
        match it {
            I16 => prefix.join(label("16").into()).into(),
            I32 => prefix.join(label("32").into()).into(),
            I64 => prefix.join(label("64").into()).into(),
        }
    }
}

impl fmt::Display for IntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use IntType::*;
        match self {
            I16 => write!(f, "I16"),
            I32 => write!(f, "I32"),
            I64 => write!(f, "I64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum UIntType {
    U8,
    U16,
    U32,
    U64,
}

impl Class for UIntType {
    type Instance = UInt;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let path = path.from_path(&Self::prefix())?;

        if path.is_empty() {
            Err(error::unsupported(
                "UInt requires a size, uint/8 or uint/16 or uint/32 or uint/64",
            ))
        } else if path.len() > 1 {
            Err(error::not_found(path))
        } else {
            match path[0].as_str() {
                "8" => Ok(UIntType::U8),
                "16" => Ok(UIntType::U16),
                "32" => Ok(UIntType::U32),
                "64" => Ok(UIntType::U64),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPath {
        NumberType::prefix().join(label("uint").into())
    }
}

impl ValueClass for UIntType {
    type Instance = UInt;

    fn get(path: &TCPath, value: UInt) -> TCResult<UInt> {
        if path.is_empty() {
            Ok(value)
        } else if path.len() == 1 {
            let dtype = match path[0].as_str() {
                "8" => UIntType::U8,
                "16" => UIntType::U16,
                "32" => UIntType::U32,
                "64" => UIntType::U64,
                _ => return Err(error::not_found(&path[0])),
            };

            Ok(value.into_type(dtype))
        } else {
            Err(error::not_found(path))
        }
    }

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for UIntType {
    type Instance = UInt;

    fn size(self) -> usize {
        match self {
            UIntType::U8 => 1,
            UIntType::U16 => 2,
            UIntType::U32 => 4,
            UIntType::U64 => 8,
        }
    }
}

impl Ord for UIntType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::U16, Self::U16) => Ordering::Equal,
            (Self::U32, Self::U32) => Ordering::Equal,
            (Self::U64, Self::U64) => Ordering::Equal,

            (Self::U8, _) => Ordering::Less,
            (_, Self::U8) => Ordering::Greater,
            (Self::U64, _) => Ordering::Greater,
            (_, Self::U64) => Ordering::Less,
            (Self::U32, Self::U16) => Ordering::Greater,
            (Self::U16, Self::U32) => Ordering::Less,
        }
    }
}

impl PartialOrd for UIntType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<UIntType> for NumberType {
    fn from(ut: UIntType) -> NumberType {
        NumberType::UInt(ut)
    }
}

impl From<UIntType> for Link {
    fn from(ut: UIntType) -> Link {
        let prefix = UIntType::prefix();

        use UIntType::*;
        match ut {
            U8 => prefix.join(label("8").into()).into(),
            U16 => prefix.join(label("16").into()).into(),
            U32 => prefix.join(label("32").into()).into(),
            U64 => prefix.join(label("64").into()).into(),
        }
    }
}

impl fmt::Display for UIntType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use UIntType::*;
        match self {
            U8 => write!(f, "U8"),
            U16 => write!(f, "U16"),
            U32 => write!(f, "U32"),
            U64 => write!(f, "U64"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum NumberType {
    Bool,
    Complex(ComplexType),
    Float(FloatType),
    Int(IntType),
    UInt(UIntType),
}

impl NumberType {
    pub fn uint64() -> Self {
        NumberType::UInt(UIntType::U64)
    }
}

impl Class for NumberType {
    type Instance = Number;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            Err(error::unsupported("You must specify a type of Number"))
        } else if suffix.len() == 1 && suffix[0].as_str() == "bool" {
            Ok(NumberType::Bool)
        } else if suffix.len() > 1 {
            match suffix[0].as_str() {
                "complex" => ComplexType::from_path(path).map(NumberType::Complex),
                "float" => FloatType::from_path(path).map(NumberType::Float),
                "int" => IntType::from_path(path).map(NumberType::Int),
                "uint" => UIntType::from_path(path).map(NumberType::UInt),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        ValueType::prefix().join(label("number").into())
    }
}

impl ValueClass for NumberType {
    type Instance = Number;

    fn get(path: &TCPath, value: Number) -> TCResult<Number> {
        if path.is_empty() {
            return Err(error::bad_request(
                "You must specify a type of Number to GET",
                "",
            ));
        }

        match path[0].as_str() {
            "int" if path.len() > 1 => {
                Int::get(&path.slice_from(1), value.try_into()?).map(Number::Int)
            }
            other => Err(error::not_found(other)),
        }
    }

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for NumberType {
    type Instance = Number;

    fn size(self) -> usize {
        use NumberType::*;
        match self {
            Bool => 1,
            Complex(ct) => NumberClass::size(ct),
            Float(ft) => NumberClass::size(ft),
            Int(it) => NumberClass::size(it),
            UInt(ut) => NumberClass::size(ut),
        }
    }
}

impl Ord for NumberType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::Bool, Self::Bool) => Ordering::Equal,
            (Self::Complex(l), Self::Complex(r)) => l.cmp(r),
            (Self::Float(l), Self::Float(r)) => l.cmp(r),
            (Self::Int(l), Self::Int(r)) => l.cmp(r),
            (Self::UInt(l), Self::UInt(r)) => l.cmp(r),

            (Self::Bool, _) => Ordering::Less,
            (_, Self::Bool) => Ordering::Greater,
            (Self::Complex(_), _) => Ordering::Greater,
            (_, Self::Complex(_)) => Ordering::Less,
            (Self::UInt(_), Self::Int(_)) => Ordering::Less,
            (Self::UInt(_), Self::Float(_)) => Ordering::Less,
            (Self::Int(_), Self::UInt(_)) => Ordering::Greater,
            (Self::Float(_), Self::UInt(_)) => Ordering::Greater,
            (Self::Int(_), Self::Float(_)) => Ordering::Less,
            (Self::Float(_), Self::Int(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for NumberType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<NumberType> for Link {
    fn from(nt: NumberType) -> Link {
        use NumberType::*;
        match nt {
            Bool => BooleanType.into(),
            Complex(ct) => ct.into(),
            Float(ft) => ft.into(),
            Int(it) => it.into(),
            UInt(ut) => ut.into(),
        }
    }
}

impl fmt::Display for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use NumberType::*;
        match self {
            Bool => write!(f, "Bool"),
            Complex(ct) => write!(f, "Complex: {}", ct),
            Float(ft) => write!(f, "Float: {}", ft),
            Int(it) => write!(f, "Int: {}", it),
            UInt(ut) => write!(f, "UInt: {}", ut),
        }
    }
}
