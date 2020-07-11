use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::hash::Hash;
use std::str::FromStr;

use bytes::Bytes;
use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use uuid::Uuid;

use crate::error;

use super::link::{Link, TCPath};
use super::op::Op;
use super::*;

const RESERVED_CHARS: [&str; 21] = [
    "/", "..", "~", "$", "`", "^", "&", "|", "=", "^", "{", "}", "<", ">", "'", "\"", "?", ":",
    "//", "@", "#",
];

fn validate_id(id: &str) -> TCResult<()> {
    if id.is_empty() {
        return Err(error::bad_request("ValueId cannot be empty", id));
    }

    let filtered: &str = &id.chars().filter(|c| *c as u8 > 32).collect::<String>();
    if filtered != id {
        return Err(error::bad_request(
            "This value ID contains an ASCII control character",
            filtered,
        ));
    }

    for pattern in &RESERVED_CHARS {
        if id.contains(pattern) {
            return Err(error::bad_request(
                "A value ID may not contain this pattern",
                pattern,
            ));
        }
    }

    if let Some(w) = Regex::new(r"\s").unwrap().find(id) {
        return Err(error::bad_request(
            "A value ID may not contain whitespace",
            format!("{:?}", w),
        ));
    }

    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ValueId {
    id: String,
}

impl ValueId {
    pub fn as_str(&self) -> &str {
        self.id.as_str()
    }

    pub fn starts_with(&self, prefix: &str) -> bool {
        self.id.starts_with(prefix)
    }
}

impl From<Uuid> for ValueId {
    fn from(id: Uuid) -> ValueId {
        id.to_hyphenated().to_string().parse().unwrap()
    }
}

impl From<u64> for ValueId {
    fn from(i: u64) -> ValueId {
        i.to_string().parse().unwrap()
    }
}

impl fmt::Display for ValueId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

impl<'de> serde::Deserialize<'de> for ValueId {
    fn deserialize<D>(deserializer: D) -> Result<ValueId, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        s.parse().map_err(de::Error::custom)
    }
}

impl Serialize for ValueId {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&self.id)
    }
}

impl PartialEq<&str> for ValueId {
    fn eq(&self, other: &&str) -> bool {
        &self.id == other
    }
}

impl FromStr for ValueId {
    type Err = error::TCError;

    fn from_str(id: &str) -> TCResult<ValueId> {
        validate_id(id)?;
        Ok(ValueId { id: id.to_string() })
    }
}

impl TryFrom<TCPath> for ValueId {
    type Error = error::TCError;

    fn try_from(path: TCPath) -> TCResult<ValueId> {
        ValueId::try_from(&path)
    }
}

impl TryFrom<&TCPath> for ValueId {
    type Error = error::TCError;

    fn try_from(path: &TCPath) -> TCResult<ValueId> {
        if path.len() == 1 {
            Ok(path[0].clone())
        } else {
            Err(error::bad_request("Expected a ValueId, found", path))
        }
    }
}

impl From<&ValueId> for String {
    fn from(value_id: &ValueId) -> String {
        value_id.id.to_string()
    }
}

#[derive(Clone, PartialEq)]
pub enum Complex {
    C32(num::Complex<f32>),
    C64(num::Complex<f64>),
}

impl TypeImpl for Complex {
    type DType = ComplexType;

    fn dtype(&self) -> ComplexType {
        match self {
            Complex::C32(_) => ComplexType::C32,
            Complex::C64(_) => ComplexType::C64,
        }
    }
}

impl NumberTypeImpl for Complex {
    type DType = ComplexType;
}

impl Eq for Complex {}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, other: Complex) -> Self {
        match (self, other) {
            (Self::C32(l), Self::C32(r)) => Self::C32(l * r),
            (Self::C64(l), Self::C64(r)) => Self::C64(l * r),
            (Self::C64(l), r) => {
                let r: num::Complex<f64> = r.into();
                Self::C64(l * r)
            }
            (l, r) => r * l,
        }
    }
}

impl PartialOrd for Complex {
    fn partial_cmp(&self, other: &Complex) -> Option<Ordering> {
        match (self, other) {
            (Complex::C32(l), Complex::C32(r)) => l.norm_sqr().partial_cmp(&r.norm_sqr()),
            (Complex::C64(l), Complex::C64(r)) => l.norm_sqr().partial_cmp(&r.norm_sqr()),
            _ => None,
        }
    }
}

impl From<Complex> for num::Complex<f64> {
    fn from(c: Complex) -> Self {
        match c {
            Complex::C32(c) => num::Complex::new(c.re as f64, c.im as f64),
            Complex::C64(c64) => c64,
        }
    }
}

impl From<Float> for Complex {
    fn from(f: Float) -> Self {
        match f {
            Float::F64(f) => Self::C64(num::Complex::new(f, 0.0f64)),
            Float::F32(f) => Self::C32(num::Complex::new(f, 0.0f32)),
        }
    }
}

impl From<Int> for Complex {
    fn from(i: Int) -> Self {
        match i {
            Int::I64(i) => Self::C64(num::Complex::new(i as f64, 0.0f64)),
            Int::I32(i) => Self::C32(num::Complex::new(i as f32, 0.0f32)),
            Int::I16(i) => Self::C32(num::Complex::new(i as f32, 0.0f32)),
        }
    }
}

impl From<UInt> for Complex {
    fn from(u: UInt) -> Self {
        match u {
            UInt::U64(u) => Self::C64(num::Complex::new(u as f64, 0.0f64)),
            UInt::U32(u) => Self::C32(num::Complex::new(u as f32, 0.0f32)),
            UInt::U16(u) => Self::C32(num::Complex::new(u as f32, 0.0f32)),
            UInt::U8(u) => Self::C32(num::Complex::new(u as f32, 0.0f32)),
        }
    }
}

impl From<bool> for Complex {
    fn from(b: bool) -> Self {
        if b {
            Self::C32(num::Complex::new(1.0f32, 0.0f32))
        } else {
            Self::C64(num::Complex::new(1.0f64, 0.0f64))
        }
    }
}

impl From<num::Complex<f32>> for Complex {
    fn from(c: num::Complex<f32>) -> Complex {
        Complex::C32(c)
    }
}

impl From<num::Complex<f64>> for Complex {
    fn from(c: num::Complex<f64>) -> Complex {
        Complex::C64(c)
    }
}

impl TryFrom<Complex> for num::Complex<f32> {
    type Error = error::TCError;

    fn try_from(c: Complex) -> TCResult<num::Complex<f32>> {
        match c {
            Complex::C32(c) => Ok(c),
            other => Err(error::bad_request("Expected C32 but found", other)),
        }
    }
}

impl Serialize for Complex {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Complex::C32(c) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/number/complex/32", &[[c.re, c.im]])?;
                map.end()
            }
            Complex::C64(c) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/number/complex/64", &[[c.re, c.im]])?;
                map.end()
            }
        }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Complex::C32(c) => write!(f, "C32({})", c),
            Complex::C64(c) => write!(f, "C64({})", c),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Float {
    F32(f32),
    F64(f64),
}

impl TypeImpl for Float {
    type DType = FloatType;

    fn dtype(&self) -> FloatType {
        match self {
            Float::F32(_) => FloatType::F32,
            Float::F64(_) => FloatType::F64,
        }
    }
}

impl NumberTypeImpl for Float {
    type DType = FloatType;
}

impl Eq for Float {}

impl Mul for Float {
    type Output = Self;

    fn mul(self, other: Float) -> Self {
        match (self, other) {
            (Self::F32(l), Self::F32(r)) => Self::F32(l * r),
            (Self::F64(l), Self::F64(r)) => Self::F64(l * r),
            (Self::F64(l), Self::F32(r)) => Self::F64(l * r as f64),
            (l, r) => (r * l),
        }
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Float) -> Option<Ordering> {
        match (self, other) {
            (Float::F32(l), Float::F32(r)) => l.partial_cmp(r),
            (Float::F64(l), Float::F64(r)) => l.partial_cmp(r),
            _ => None,
        }
    }
}

impl From<f32> for Float {
    fn from(f: f32) -> Self {
        Self::F32(f)
    }
}

impl From<f64> for Float {
    fn from(f: f64) -> Self {
        Self::F64(f)
    }
}

impl From<Int> for Float {
    fn from(i: Int) -> Self {
        match i {
            Int::I64(i) => Self::F64(i as f64),
            Int::I32(i) => Self::F32(i as f32),
            Int::I16(i) => Self::F32(i as f32),
        }
    }
}

impl From<UInt> for Float {
    fn from(u: UInt) -> Self {
        match u {
            UInt::U64(u) => Self::F64(u as f64),
            UInt::U32(u) => Self::F32(u as f32),
            UInt::U16(u) => Self::F32(u as f32),
            UInt::U8(u) => Self::F32(u as f32),
        }
    }
}

impl From<bool> for Float {
    fn from(b: bool) -> Self {
        if b {
            Self::F32(1.0f32)
        } else {
            Self::F32(0.0f32)
        }
    }
}

impl TryFrom<Float> for f32 {
    type Error = error::TCError;

    fn try_from(f: Float) -> TCResult<f32> {
        match f {
            Float::F32(f) => Ok(f),
            other => Err(error::bad_request("Expected F32 but found", other)),
        }
    }
}

impl From<Float> for f64 {
    fn from(f: Float) -> f64 {
        match f {
            Float::F32(f) => f as f64,
            Float::F64(f) => f,
        }
    }
}

impl Serialize for Float {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Float::F32(f) => s.serialize_f32(*f),
            Float::F64(f) => s.serialize_f64(*f),
        }
    }
}

impl fmt::Display for Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Float::F32(i) => write!(f, "F32({})", i),
            Float::F64(i) => write!(f, "F64({})", i),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Int {
    I16(i16),
    I32(i32),
    I64(i64),
}

impl TypeImpl for Int {
    type DType = IntType;

    fn dtype(&self) -> IntType {
        match self {
            Int::I16(_) => IntType::I16,
            Int::I32(_) => IntType::I32,
            Int::I64(_) => IntType::I64,
        }
    }
}

impl NumberTypeImpl for Int {
    type DType = IntType;
}

impl Eq for Int {}

impl Mul for Int {
    type Output = Self;

    fn mul(self, other: Int) -> Self {
        match (self, other) {
            (Self::I64(l), Self::I64(r)) => Self::I64(l * r),
            (Self::I64(l), Self::I32(r)) => Self::I64(l * r as i64),
            (Self::I64(l), Self::I16(r)) => Self::I64(l * r as i64),
            (Self::I32(l), Self::I32(r)) => Self::I32(l * r),
            (Self::I32(l), Self::I16(r)) => Self::I32(l * r as i32),
            (Self::I16(l), Self::I16(r)) => Self::I16(l * r),
            (l, r) => r * l,
        }
    }
}

impl PartialOrd for Int {
    fn partial_cmp(&self, other: &Int) -> Option<Ordering> {
        match (self, other) {
            (Int::I16(l), Int::I16(r)) => l.partial_cmp(r),
            (Int::I32(l), Int::I32(r)) => l.partial_cmp(r),
            (Int::I64(l), Int::I64(r)) => l.partial_cmp(r),
            _ => None,
        }
    }
}

impl From<i16> for Int {
    fn from(i: i16) -> Int {
        Int::I16(i)
    }
}

impl From<i32> for Int {
    fn from(i: i32) -> Int {
        Int::I32(i)
    }
}

impl From<i64> for Int {
    fn from(i: i64) -> Int {
        Int::I64(i)
    }
}

impl From<UInt> for Int {
    fn from(u: UInt) -> Int {
        match u {
            UInt::U64(u) => Int::I64(u as i64),
            UInt::U32(u) => Int::I32(u as i32),
            UInt::U16(u) => Int::I16(u as i16),
            UInt::U8(u) => Int::I16(u as i16),
        }
    }
}

impl From<bool> for Int {
    fn from(b: bool) -> Int {
        if b {
            Int::I16(1)
        } else {
            Int::I16(0)
        }
    }
}
impl TryFrom<Int> for i16 {
    type Error = error::TCError;

    fn try_from(i: Int) -> TCResult<i16> {
        match i {
            Int::I16(i) => Ok(i),
            other => Err(error::bad_request("Expected I16 but found", other)),
        }
    }
}

impl TryFrom<Int> for i32 {
    type Error = error::TCError;

    fn try_from(i: Int) -> TCResult<i32> {
        match i {
            Int::I32(i) => Ok(i),
            other => Err(error::bad_request("Expected I32 but found", other)),
        }
    }
}

impl From<Int> for i64 {
    fn from(i: Int) -> i64 {
        match i {
            Int::I16(i) => i as i64,
            Int::I32(i) => i as i64,
            Int::I64(i) => i,
        }
    }
}

impl Serialize for Int {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Int::I16(i) => s.serialize_i16(*i),
            Int::I32(i) => s.serialize_i32(*i),
            Int::I64(i) => s.serialize_i64(*i),
        }
    }
}

impl fmt::Display for Int {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Int::I16(i) => write!(f, "I16: {}", i),
            Int::I32(i) => write!(f, "I32: {}", i),
            Int::I64(i) => write!(f, "I64: {}", i),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum UInt {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
}

impl TypeImpl for UInt {
    type DType = UIntType;

    fn dtype(&self) -> UIntType {
        match self {
            UInt::U8(_) => UIntType::U8,
            UInt::U16(_) => UIntType::U16,
            UInt::U32(_) => UIntType::U32,
            UInt::U64(_) => UIntType::U64,
        }
    }
}

impl NumberTypeImpl for UInt {
    type DType = UIntType;
}

impl Mul for UInt {
    type Output = Self;

    fn mul(self, other: UInt) -> Self {
        match (self, other) {
            (UInt::U64(l), UInt::U64(r)) => UInt::U64(l * r),
            (UInt::U64(l), UInt::U32(r)) => UInt::U64(l * r as u64),
            (UInt::U64(l), UInt::U16(r)) => UInt::U64(l * r as u64),
            (UInt::U64(l), UInt::U8(r)) => UInt::U64(l * r as u64),
            (UInt::U32(l), UInt::U32(r)) => UInt::U32(l * r),
            (UInt::U32(l), UInt::U16(r)) => UInt::U32(l * r as u32),
            (UInt::U32(l), UInt::U8(r)) => UInt::U32(l * r as u32),
            (UInt::U16(l), UInt::U16(r)) => UInt::U16(l * r),
            (UInt::U16(l), UInt::U8(r)) => UInt::U16(l * r as u16),
            (UInt::U8(l), UInt::U8(r)) => UInt::U8(l * r),
            (l, r) => r * l,
        }
    }
}

impl Eq for UInt {}

impl Ord for UInt {
    fn cmp(&self, other: &UInt) -> Ordering {
        match (self, other) {
            (UInt::U64(l), UInt::U64(r)) => l.cmp(r),
            (UInt::U64(l), UInt::U32(r)) => l.cmp(&r.clone().into()),
            (UInt::U64(l), UInt::U16(r)) => l.cmp(&r.clone().into()),
            (UInt::U64(l), UInt::U8(r)) => l.cmp(&r.clone().into()),
            (UInt::U32(l), UInt::U32(r)) => l.cmp(r),
            (UInt::U32(l), UInt::U16(r)) => l.cmp(&r.clone().into()),
            (UInt::U32(l), UInt::U8(r)) => l.cmp(&r.clone().into()),
            (UInt::U16(l), UInt::U16(r)) => l.cmp(r),
            (UInt::U16(l), UInt::U8(r)) => l.cmp(&r.clone().into()),
            (UInt::U8(l), UInt::U8(r)) => l.cmp(r),
            (l, r) => match r.cmp(l) {
                Ordering::Greater => Ordering::Less,
                Ordering::Less => Ordering::Greater,
                Ordering::Equal => Ordering::Equal,
            },
        }
    }
}

impl PartialOrd for UInt {
    fn partial_cmp(&self, other: &UInt) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<bool> for UInt {
    fn from(b: bool) -> UInt {
        if b {
            UInt::U8(1)
        } else {
            UInt::U8(0)
        }
    }
}

impl From<u8> for UInt {
    fn from(u: u8) -> UInt {
        UInt::U8(u)
    }
}

impl From<u16> for UInt {
    fn from(u: u16) -> UInt {
        UInt::U16(u)
    }
}

impl From<u32> for UInt {
    fn from(u: u32) -> UInt {
        UInt::U32(u)
    }
}

impl From<u64> for UInt {
    fn from(u: u64) -> UInt {
        UInt::U64(u)
    }
}

impl TryFrom<UInt> for u8 {
    type Error = error::TCError;

    fn try_from(u: UInt) -> TCResult<u8> {
        match u {
            UInt::U8(u) => Ok(u),
            other => Err(error::bad_request("Expected a UInt8 but found", other)),
        }
    }
}

impl TryFrom<UInt> for u16 {
    type Error = error::TCError;

    fn try_from(u: UInt) -> TCResult<u16> {
        match u {
            UInt::U16(u) => Ok(u),
            other => Err(error::bad_request("Expected a UInt16 but found", other)),
        }
    }
}

impl TryFrom<UInt> for u32 {
    type Error = error::TCError;

    fn try_from(u: UInt) -> TCResult<u32> {
        match u {
            UInt::U32(u) => Ok(u),
            other => Err(error::bad_request("Expected a UInt32 but found", other)),
        }
    }
}

impl TryFrom<UInt> for u64 {
    type Error = error::TCError;

    fn try_from(u: UInt) -> TCResult<u64> {
        match u {
            UInt::U64(u) => Ok(u),
            other => Err(error::bad_request("Expected a UInt64 but found", other)),
        }
    }
}

impl Serialize for UInt {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            UInt::U8(u) => s.serialize_u8(*u),
            UInt::U16(u) => s.serialize_u16(*u),
            UInt::U32(u) => s.serialize_u32(*u),
            UInt::U64(u) => s.serialize_u64(*u),
        }
    }
}

impl fmt::Display for UInt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UInt::U8(u) => write!(f, "U8: {}", u),
            UInt::U16(u) => write!(f, "UInt16: {}", u),
            UInt::U32(u) => write!(f, "UInt32: {}", u),
            UInt::U64(u) => write!(f, "UInt64: {}", u),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Number {
    Bool(bool),
    Complex(Complex),
    Float(Float),
    Int(Int),
    UInt(UInt),
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        match (self, other) {
            (Self::Complex(l), Self::Complex(r)) => l.partial_cmp(r),
            (Self::Complex(l), Self::Float(r)) => l.partial_cmp(&r.clone().into()),
            (Self::Complex(l), Self::Int(r)) => l.partial_cmp(&r.clone().into()),
            (Self::Complex(l), Self::UInt(r)) => l.partial_cmp(&r.clone().into()),
            (Self::Complex(l), Self::Bool(r)) => l.partial_cmp(&r.clone().into()),
            (Self::Float(l), Self::Float(r)) => l.partial_cmp(r),
            (Self::Float(l), Self::Int(r)) => l.partial_cmp(&r.clone().into()),
            (Self::Float(l), Self::UInt(r)) => l.partial_cmp(&r.clone().into()),
            (Self::Float(l), Self::Bool(r)) => l.partial_cmp(&r.clone().into()),
            (Self::Int(l), Self::Int(r)) => l.partial_cmp(r),
            (Self::Int(l), Self::UInt(r)) => l.partial_cmp(&r.clone().into()),
            (Self::UInt(l), Self::UInt(r)) => l.partial_cmp(r),
            (Self::Bool(l), Self::Bool(r)) => l.partial_cmp(r),
            (l, r) => match r.partial_cmp(l) {
                Some(Ordering::Greater) => Some(Ordering::Less),
                Some(Ordering::Less) => Some(Ordering::Greater),
                Some(Ordering::Equal) => Some(Ordering::Equal),
                None => None,
            },
        }
    }
}

impl TypeImpl for Number {
    type DType = NumberType;

    fn dtype(&self) -> NumberType {
        use NumberType::*;
        match self {
            Self::Bool(_) => Bool,
            Self::Complex(c) => Complex(c.dtype()),
            Self::Float(f) => Float(f.dtype()),
            Self::Int(i) => Int(i.dtype()),
            Self::UInt(u) => UInt(u.dtype()),
        }
    }
}

impl NumberTypeImpl for Number {
    type DType = NumberType;
}

impl Mul for Number {
    type Output = Self;

    fn mul(self, other: Number) -> Self {
        match (self, other) {
            (Self::Bool(false), r) => r.dtype().zero(),
            (Self::Bool(true), r) => r,

            (Self::Complex(l), Self::Complex(r)) => Self::Complex(l * r),
            (Self::Float(l), Self::Float(r)) => Self::Float(l * r),
            (Self::Int(l), Self::Int(r)) => Self::Int(l * r),
            (Self::UInt(l), Self::UInt(r)) => Self::UInt(l * r),
            (Self::Complex(l), Self::Float(r)) => {
                let r: Complex = r.into();
                Self::Complex(l * r)
            }
            (Self::Complex(l), Self::Int(r)) => {
                let r: Complex = r.into();
                Self::Complex(l * r)
            }
            (Self::Complex(l), Self::UInt(r)) => {
                let r: Complex = r.into();
                Self::Complex(l * r)
            }
            (Self::Float(l), Self::Int(r)) => {
                let r: Float = r.into();
                Self::Float(l * r)
            }
            (Self::Float(l), Self::UInt(r)) => {
                let r: Float = r.into();
                Self::Float(l * r)
            }
            (Self::Int(l), Self::UInt(r)) => {
                let r: Int = r.into();
                Self::Int(l * r)
            }
            (l, r) => r * l,
        }
    }
}

impl From<bool> for Number {
    fn from(b: bool) -> Number {
        Number::Bool(b)
    }
}

pub trait Numeric {}

impl From<Complex> for Number {
    fn from(c: Complex) -> Number {
        Number::Complex(c)
    }
}

impl From<Float> for Number {
    fn from(f: Float) -> Number {
        Number::Float(f)
    }
}

impl From<Int> for Number {
    fn from(i: Int) -> Number {
        Number::Int(i)
    }
}

impl From<UInt> for Number {
    fn from(u: UInt) -> Number {
        Number::UInt(u)
    }
}

impl TryFrom<Number> for bool {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<bool> {
        match n {
            Number::Bool(b) => Ok(b),
            other => Err(error::bad_request("Expected Bool but found", other)),
        }
    }
}

impl TryFrom<Number> for Complex {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<Complex> {
        match n {
            Number::Complex(c) => Ok(c),
            other => Err(error::bad_request("Expected Complex but found", other)),
        }
    }
}

impl TryFrom<Number> for Float {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<Float> {
        match n {
            Number::Float(f) => Ok(f),
            other => Err(error::bad_request("Expected Float but found", other)),
        }
    }
}

impl TryFrom<Number> for Int {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<Int> {
        match n {
            Number::Int(i) => Ok(i),
            other => Err(error::bad_request("Expected Int but found", other)),
        }
    }
}

impl TryFrom<Number> for UInt {
    type Error = error::TCError;

    fn try_from(n: Number) -> TCResult<UInt> {
        match n {
            Number::UInt(u) => Ok(u),
            other => Err(error::bad_request("Expected UInt but found", other)),
        }
    }
}

impl Serialize for Number {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Number::Bool(b) => s.serialize_bool(*b),
            Number::Complex(c) => c.serialize(s),
            Number::Float(f) => f.serialize(s),
            Number::Int(i) => i.serialize(s),
            Number::UInt(u) => u.serialize(s),
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Number::Bool(b) => write!(f, "Bool({})", b),
            Number::Complex(c) => write!(f, "Complex({})", c),
            Number::Float(n) => write!(f, "Float({})", n),
            Number::Int(i) => write!(f, "Int({})", i),
            Number::UInt(u) => write!(f, "UInt({})", u),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum TCString {
    Id(ValueId),
    Link(Link),
    Ref(TCRef),
    r#String(String),
}

impl TypeImpl for TCString {
    type DType = StringType;

    fn dtype(&self) -> StringType {
        match self {
            TCString::Id(_) => StringType::Id,
            TCString::Link(_) => StringType::Link,
            TCString::Ref(_) => StringType::Ref,
            TCString::r#String(_) => StringType::r#String,
        }
    }
}

impl From<Link> for TCString {
    fn from(l: Link) -> TCString {
        TCString::Link(l)
    }
}

impl From<TCPath> for TCString {
    fn from(path: TCPath) -> TCString {
        TCString::Link(path.into())
    }
}

impl From<ValueId> for TCString {
    fn from(id: ValueId) -> TCString {
        TCString::Id(id)
    }
}

impl From<TCRef> for TCString {
    fn from(r: TCRef) -> TCString {
        TCString::Ref(r)
    }
}

impl From<String> for TCString {
    fn from(s: String) -> TCString {
        TCString::r#String(s)
    }
}

impl TryFrom<TCString> for Link {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<Link> {
        match s {
            TCString::Link(l) => Ok(l),
            other => Err(error::bad_request("Expected Link but found", other)),
        }
    }
}

impl TryFrom<TCString> for String {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<String> {
        match s {
            TCString::r#String(s) => Ok(s),
            other => Err(error::bad_request("Expected a String but found", other)),
        }
    }
}

impl TryFrom<TCString> for TCPath {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<TCPath> {
        match s {
            TCString::Link(l) => {
                if l.host().is_none() {
                    Ok(l.path().clone())
                } else {
                    Err(error::bad_request("Expected Path but found Link", l))
                }
            }
            other => Err(error::bad_request("Expected Path but found", other)),
        }
    }
}

impl TryFrom<TCString> for ValueId {
    type Error = error::TCError;

    fn try_from(s: TCString) -> TCResult<ValueId> {
        let s: String = s.try_into()?;
        s.parse()
    }
}

impl Serialize for TCString {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use TCString::*;
        match self {
            Id(i) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/id", &[i.as_str()])?;
                map.end()
            }
            Link(l) => l.serialize(s),
            Ref(r) => r.serialize(s),
            r#String(v) => s.serialize_str(v),
        }
    }
}

impl fmt::Display for TCString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCString::Id(id) => write!(f, "ValueId: {}", id),
            TCString::Link(l) => write!(f, "Link: {}", l),
            TCString::Ref(r) => write!(f, "Ref: {}", r),
            TCString::r#String(s) => write!(f, "String: {}", s),
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum Value {
    None,
    Bytes(Bytes),
    Number(Number),
    TCString(TCString),
    Op(Box<Op>),
    Vector(Vec<Value>),
}

impl TypeImpl for Value {
    type DType = TCType;

    fn dtype(&self) -> TCType {
        match self {
            Value::None => TCType::None,
            Value::Bytes(_) => TCType::Bytes,
            Value::Number(n) => TCType::Number(n.dtype()),
            Value::TCString(s) => TCType::TCString(s.dtype()),
            Value::Op(_) => TCType::Op,
            Value::Vector(_) => TCType::Vector,
        }
    }
}

impl From<()> for Value {
    fn from(_: ()) -> Value {
        Value::None
    }
}

impl From<&'static [u8]> for Value {
    fn from(b: &'static [u8]) -> Value {
        Value::Bytes(Bytes::from(b))
    }
}

impl From<Bytes> for Value {
    fn from(b: Bytes) -> Value {
        Value::Bytes(b)
    }
}

impl From<Number> for Value {
    fn from(n: Number) -> Value {
        Value::Number(n)
    }
}

impl From<TCString> for Value {
    fn from(s: TCString) -> Value {
        Value::TCString(s)
    }
}

impl From<Op> for Value {
    fn from(op: Op) -> Value {
        Value::Op(Box::new(op))
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Value {
        match opt {
            Some(val) => val.into(),
            None => Value::None,
        }
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(mut v: Vec<T>) -> Value {
        Value::Vector(v.drain(..).map(|i| i.into()).collect())
    }
}

impl TryFrom<Value> for Bytes {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Bytes> {
        match v {
            Value::Bytes(b) => Ok(b),
            other => Err(error::bad_request("Expected Bytes but found", other)),
        }
    }
}

impl TryFrom<Value> for Number {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Number> {
        match v {
            Value::Number(n) => Ok(n),
            other => Err(error::bad_request("Expected Number but found", other)),
        }
    }
}

impl<E: Into<error::TCError>, T: TryFrom<Value, Error = E>> TryFrom<Value> for Vec<T> {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Vec<T>> {
        match v {
            Value::Vector(mut v) => v
                .drain(..)
                .map(|i| i.try_into().map_err(|e: E| e.into()))
                .collect(),
            other => Err(error::bad_request("Expected a Vector but found", other)),
        }
    }
}

struct ValueVisitor;

impl ValueVisitor {
    fn visit_float<F: Into<Float>>(&self, f: F) -> TCResult<Value> {
        self.visit_number(f.into())
    }

    fn visit_int<I: Into<Int>>(&self, i: I) -> TCResult<Value> {
        self.visit_number(i.into())
    }

    fn visit_number<N: Into<Number>>(&self, n: N) -> TCResult<Value> {
        Ok(Value::Number(n.into()))
    }
}

impl<'de> de::Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Value, e.g. \"foo\" or 123 or {\"$object_ref: [\"slice_id\", \"$state\"]\"}")
    }

    fn visit_f32<E>(self, value: f32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_float(value).map_err(de::Error::custom)
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_float(value).map_err(de::Error::custom)
    }

    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_int(value).map_err(de::Error::custom)
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_int(value).map_err(de::Error::custom)
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::TCString(TCString::r#String(value.to_string())))
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        if let Some(key) = access.next_key::<&str>()? {
            let mut value: Vec<Value> = access.next_value()?;

            if key.starts_with('$') {
                let subject = key.parse::<TCRef>().map_err(de::Error::custom)?;
                match value.len() {
                    0 => Ok(Value::TCString(TCString::Ref(subject))),
                    1 => Ok(Op::Get(subject.into(), value.remove(0)).into()),
                    2 => Ok(Op::Put(subject.into(), value.remove(0), value.remove(0)).into()),
                    _ => Err(de::Error::custom(format!(
                        "Expected a Get or Put op, found {}",
                        Value::Vector(value)
                    ))),
                }
            } else if let Ok(link) = key.parse::<Link>() {
                Ok(Value::TCString(TCString::Link(link)))
            } else {
                panic!("NOT IMPLEMENTED")
            }
        } else {
            Err(de::Error::custom("Unable to parse map entry"))
        }
    }
}

impl<'de> de::Deserialize<'de> for Value {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        d.deserialize_any(ValueVisitor)
    }
}

impl Serialize for Value {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Value::None => s.serialize_none(),
            Value::Bytes(b) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/bytes", &[base64::encode(b)])?;
                map.end()
            }
            Value::Number(n) => n.serialize(s),
            Value::Op(op) => {
                let mut map = s.serialize_map(Some(1))?;
                match &**op {
                    Op::Get(subject, selector) => {
                        map.serialize_entry(&subject.to_string(), &[selector])?
                    }
                    Op::Put(subject, selector, value) => {
                        map.serialize_entry(&subject.to_string(), &[selector, value])?
                    }
                }
                map.end()
            }
            Value::TCString(tc_string) => tc_string.serialize(s),
            Value::Vector(v) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for item in v {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
        }
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::None => write!(f, "None"),
            Value::Bytes(b) => write!(f, "Bytes({})", b.len()),
            Value::Number(n) => write!(f, "Number({})", n),
            Value::TCString(s) => write!(f, "String({})", s),
            Value::Op(op) => write!(f, "Op: {}", op),
            Value::Vector(v) => write!(
                f,
                "[{}]",
                v.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}
