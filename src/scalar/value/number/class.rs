use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::{Add, Mul, Sub};

use serde::{Deserialize, Serialize};

use crate::class::{Class, NativeClass, TCResult, TCType};
use crate::error;
use crate::scalar::{
    label, CastInto, Link, PathSegment, Scalar, ScalarClass, ScalarType, TCPathBuf, TryCastFrom,
    Value, ValueClass, ValueInstance, ValueType,
};

use super::instance::{Boolean, Complex, Float, Int, Number, UInt};

pub trait NumberClass: Class + ValueClass + Into<NumberType> + Ord + Send {
    type Instance: NumberInstance;

    fn size(self) -> usize;

    fn one(&self) -> <Self as NumberClass>::Instance;

    fn zero(&self) -> <Self as NumberClass>::Instance;
}

pub trait NumberInstance:
    ValueInstance
    + Add<Output = Self>
    + Mul<Output = Self>
    + Sub<Output = Self>
    + Sized
    + PartialOrd
    + From<Boolean>
    + Into<Number>
    + Default
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

    fn sub(self, other: Self) -> Self {
        self - other
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

#[derive(Clone, Copy, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub enum ComplexType {
    C32,
    C64,
}

impl Class for ComplexType {
    type Instance = Complex;
}

impl NativeClass for ComplexType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::unsupported(
                "Complex number requires a size, complex/32 or complex/64",
            ))
        } else if suffix.len() > 1 {
            Err(error::path_not_found(suffix))
        } else {
            match suffix[0].as_str() {
                "32" => Ok(ComplexType::C32),
                "64" => Ok(ComplexType::C64),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPathBuf {
        NumberType::prefix().append(label("complex"))
    }
}

impl ScalarClass for ComplexType {
    type Instance = Complex;

    fn try_cast<S>(&self, scalar: S) -> TCResult<Complex>
    where
        Scalar: From<S>,
    {
        let v = Value::try_from(Scalar::from(scalar))?;
        let n = Number::try_from(v)?;
        let n = n.into_type(NumberType::Complex(*self));
        n.try_into()
    }
}

impl ValueClass for ComplexType {
    type Instance = Complex;

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

    fn one(&self) -> Complex {
        match self {
            Self::C32 => num::Complex::<f32>::new(1f32, 0f32).into(),
            Self::C64 => num::Complex::<f64>::new(1f64, 0f64).into(),
        }
    }

    fn zero(&self) -> Complex {
        match self {
            Self::C32 => num::Complex::<f32>::new(0f32, 0f32).into(),
            Self::C64 => num::Complex::<f64>::new(0f64, 0f64).into(),
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
        use ComplexType::*;
        let suffix = match ct {
            C32 => label("32"),
            C64 => label("64"),
        };

        ComplexType::prefix().append(suffix).into()
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
}

impl NativeClass for BooleanType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Ok(BooleanType)
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        NumberType::prefix().append(label("bool"))
    }
}

impl ScalarClass for BooleanType {
    type Instance = Boolean;

    fn try_cast<S>(&self, scalar: S) -> TCResult<Boolean>
    where
        Scalar: From<S>,
    {
        let s = Scalar::from(scalar);
        let v = Value::try_from(s)?;
        let n = Number::try_from(v)?;
        let n = n.into_type(NumberType::Bool);
        n.try_into()
    }
}

impl ValueClass for BooleanType {
    type Instance = Boolean;

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for BooleanType {
    type Instance = Boolean;

    fn size(self) -> usize {
        1
    }

    fn one(&self) -> Boolean {
        true.into()
    }

    fn zero(&self) -> Boolean {
        false.into()
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
}

impl NativeClass for FloatType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::unsupported(
                "Float requires a size, float/32 or float/64",
            ))
        } else if suffix.len() > 1 {
            Err(error::path_not_found(suffix))
        } else {
            match suffix[0].as_str() {
                "32" => Ok(FloatType::F32),
                "64" => Ok(FloatType::F64),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPathBuf {
        NumberType::prefix().append(label("float"))
    }
}

impl ScalarClass for FloatType {
    type Instance = Float;

    fn try_cast<S>(&self, scalar: S) -> TCResult<Float>
    where
        Scalar: From<S>,
    {
        let s = Scalar::from(scalar);
        let v = Value::try_from(s)?;
        let n = Number::try_from(v)?;
        let n = n.into_type(NumberType::Float(*self));
        n.try_into()
    }
}

impl ValueClass for FloatType {
    type Instance = Float;

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for FloatType {
    type Instance = Float;

    fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn one(&self) -> Float {
        match self {
            Self::F32 => 1f32.into(),
            Self::F64 => 1f64.into(),
        }
    }

    fn zero(&self) -> Float {
        match self {
            Self::F32 => 0f32.into(),
            Self::F64 => 0f64.into(),
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
        use FloatType::*;
        let suffix = match ft {
            F32 => label("32"),
            F64 => label("64"),
        };

        FloatType::prefix().append(suffix).into()
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
}

impl NativeClass for IntType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::unsupported(
                "Int requires a size, int/16 or int/32 or int/64",
            ))
        } else if suffix.len() > 1 {
            Err(error::path_not_found(path))
        } else {
            match suffix[0].as_str() {
                "16" => Ok(IntType::I16),
                "32" => Ok(IntType::I32),
                "64" => Ok(IntType::I64),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPathBuf {
        NumberType::prefix().append(label("int"))
    }
}

impl ScalarClass for IntType {
    type Instance = Int;

    fn try_cast<S>(&self, scalar: S) -> TCResult<Int>
    where
        Scalar: From<S>,
    {
        let v = Value::try_from(Scalar::from(scalar))?;
        let n = Number::try_from(v)?;
        let n = n.into_type(NumberType::Int(*self));
        n.try_into()
    }
}

impl ValueClass for IntType {
    type Instance = Int;

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

impl NumberClass for IntType {
    type Instance = Int;

    fn size(self) -> usize {
        match self {
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
        }
    }

    fn one(&self) -> Int {
        match self {
            Self::I16 => 1i16.into(),
            Self::I32 => 1i32.into(),
            Self::I64 => 1i64.into(),
        }
    }

    fn zero(&self) -> Int {
        match self {
            Self::I16 => 0i16.into(),
            Self::I32 => 0i32.into(),
            Self::I64 => 0i64.into(),
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
        use IntType::*;
        let suffix = match it {
            I16 => label("16"),
            I32 => label("32"),
            I64 => label("64"),
        };

        IntType::prefix().append(suffix).into()
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
}

impl NativeClass for UIntType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::unsupported(
                "UInt requires a size, uint/8 or uint/16 or uint/32 or uint/64",
            ))
        } else if suffix.len() > 1 {
            Err(error::path_not_found(path))
        } else {
            match suffix[0].as_str() {
                "8" => Ok(UIntType::U8),
                "16" => Ok(UIntType::U16),
                "32" => Ok(UIntType::U32),
                "64" => Ok(UIntType::U64),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPathBuf {
        NumberType::prefix().append(label("uint"))
    }
}

impl ScalarClass for UIntType {
    type Instance = UInt;

    fn try_cast<S>(&self, scalar: S) -> TCResult<UInt>
    where
        Scalar: From<S>,
    {
        let v = Value::try_from(Scalar::from(scalar))?;
        let n = Number::try_from(v)?;
        let n = n.into_type(NumberType::UInt(*self));
        n.try_into()
    }
}

impl ValueClass for UIntType {
    type Instance = UInt;

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

    fn one(&self) -> UInt {
        match self {
            Self::U8 => 1u8.into(),
            Self::U16 => 1u16.into(),
            Self::U32 => 1u32.into(),
            Self::U64 => 1u64.into(),
        }
    }

    fn zero(&self) -> UInt {
        match self {
            Self::U8 => 0u8.into(),
            Self::U16 => 0u16.into(),
            Self::U32 => 0u32.into(),
            Self::U64 => 0u64.into(),
        }
    }
}

impl Ord for UIntType {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Self::U8, Self::U8) => Ordering::Equal,
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
        use UIntType::*;
        let suffix = match ut {
            U8 => label("8"),
            U16 => label("16"),
            U32 => label("32"),
            U64 => label("64"),
        };

        UIntType::prefix().append(suffix).into()
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
    Number,
}

impl NumberType {
    pub fn uint64() -> Self {
        NumberType::UInt(UIntType::U64)
    }
}

impl Class for NumberType {
    type Instance = Number;
}

impl NativeClass for NumberType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Ok(NumberType::Number)
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
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        ValueType::prefix().append(label("number"))
    }
}

impl ScalarClass for NumberType {
    type Instance = Number;

    fn try_cast<S>(&self, scalar: S) -> TCResult<Number>
    where
        Scalar: From<S>,
    {
        let value = Value::try_from(Scalar::from(scalar))?;
        Number::try_from(value).map(|n| n.into_type(*self))
    }
}

impl ValueClass for NumberType {
    type Instance = Number;

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

            // a generic Number still has a distinct maximum size
            Number => NumberClass::size(ComplexType::C64),
        }
    }

    fn one(&self) -> Number {
        use NumberType::*;
        match self {
            Bool | Number => true.into(),
            Complex(ct) => ct.one().into(),
            Float(ft) => ft.one().into(),
            Int(it) => it.one().into(),
            UInt(ut) => ut.one().into(),
        }
    }

    fn zero(&self) -> Number {
        use NumberType::*;
        match self {
            Bool | Number => false.into(),
            Complex(ct) => ct.zero().into(),
            Float(ft) => ft.zero().into(),
            Int(it) => it.zero().into(),
            UInt(ut) => ut.zero().into(),
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

            (Self::Number, Self::Number) => Ordering::Equal,
            (Self::Number, _) => Ordering::Greater,
            (_, Self::Number) => Ordering::Less,

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
            Number => NumberType::prefix().into(),
        }
    }
}

impl TryFrom<TCType> for NumberType {
    type Error = error::TCError;

    fn try_from(class: TCType) -> TCResult<NumberType> {
        match class {
            TCType::Scalar(ScalarType::Value(ValueType::Number(nt))) => Ok(nt),
            other => Err(error::bad_request("Expected ValueType, found", other)),
        }
    }
}

impl TryCastFrom<Link> for NumberType {
    fn can_cast_from(link: &Link) -> bool {
        NumberType::from_path(&link.path()[..]).is_ok()
    }

    fn opt_cast_from(link: Link) -> Option<NumberType> {
        NumberType::from_path(&link.path()[..]).ok()
    }
}

impl TryCastFrom<TCType> for NumberType {
    fn can_cast_from(class: &TCType) -> bool {
        if let TCType::Scalar(ScalarType::Value(ValueType::Number(_))) = class {
            true
        } else {
            false
        }
    }

    fn opt_cast_from(class: TCType) -> Option<NumberType> {
        if let TCType::Scalar(ScalarType::Value(ValueType::Number(nt))) = class {
            Some(nt)
        } else {
            None
        }
    }
}

impl fmt::Display for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use NumberType::*;
        match self {
            Bool => write!(f, "type Bool"),
            Complex(ct) => write!(f, "Complex: {}", ct),
            Float(ft) => write!(f, "Float: {}", ft),
            Int(it) => write!(f, "Int: {}", it),
            UInt(ut) => write!(f, "UInt: {}", ut),
            Number => write!(f, "type Number"),
        }
    }
}
