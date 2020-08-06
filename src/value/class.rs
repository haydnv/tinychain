use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul};

use serde::{Deserialize, Serialize};

use crate::error;

use super::number::{Boolean, Complex, Float, Int, Number, UInt};
use super::string::TCString;
use super::{TCResult, Value};

pub trait Class: Clone + Eq + fmt::Display {
    type Instance: Instance;
}

pub trait ValueClass: Class {
    type Instance: ValueInstance;

    fn size(self) -> Option<usize>;
}

pub trait NumberClass: Class + Into<NumberType> + Ord + Send + Sync {
    type Instance: NumberInstance + Into<Number>;

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

impl<T: NumberClass> ValueClass for T {
    type Instance = <Self as NumberClass>::Instance;

    fn size(self) -> Option<usize> {
        Some(NumberClass::size(self))
    }
}

pub trait Instance {
    type Class: Class;

    fn class(&self) -> Self::Class;

    fn is_a(&self, dtype: Self::Class) -> bool {
        self.class() == dtype
    }

    fn expect<M: fmt::Display>(&self, dtype: Self::Class, context_msg: M) -> TCResult<()> {
        if self.is_a(dtype.clone()) {
            Ok(())
        } else {
            Err(error::TCError::of(
                error::Code::BadRequest,
                format!(
                    "Expected {} but found {} {}",
                    self.class(),
                    dtype,
                    context_msg
                ),
            ))
        }
    }
}

pub trait ValueInstance: Instance + Serialize {
    type Class: ValueClass;
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

impl Class for NumberType {
    type Instance = Number;
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

impl From<NumberType> for ValueType {
    fn from(nt: NumberType) -> ValueType {
        ValueType::Number(nt)
    }
}

impl fmt::Display for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use NumberType::*;
        match self {
            Bool => write!(f, "Bool"),
            Complex(c) => write!(f, "Complex: {}", c),
            Float(ft) => write!(f, "Float: {}", ft),
            Int(i) => write!(f, "Int: {}", i),
            UInt(u) => write!(f, "UInt: {}", u),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum StringType {
    Id,
    Link,
    Ref,
    r#String,
}

impl Class for StringType {
    type Instance = TCString;
}

impl From<StringType> for ValueType {
    fn from(st: StringType) -> ValueType {
        ValueType::TCString(st)
    }
}

impl fmt::Display for StringType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use StringType::*;
        match self {
            Id => write!(f, "type Id"),
            Link => write!(f, "type Link"),
            Ref => write!(f, "type Ref"),
            r#String => write!(f, "type String"),
        }
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub enum ValueType {
    Bytes,
    None,
    Number(NumberType),
    TCString(StringType),
    Op,
    Vector,
}

impl Class for ValueType {
    type Instance = Value;
}

impl ValueClass for ValueType {
    type Instance = Value;

    fn size(self) -> Option<usize> {
        use ValueType::*;
        match self {
            None => Some(1),
            Number(nt) => ValueClass::size(nt),
            _ => Option::None,
        }
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use ValueType::*;
        match self {
            None => write!(f, "type None"),
            Bytes => write!(f, "type Bytes"),
            Number(n) => write!(f, "type Number: {}", n),
            TCString(s) => write!(f, "type String: {}", s),
            Op => write!(f, "type Op"),
            Vector => write!(f, "type Vector"),
        }
    }
}
