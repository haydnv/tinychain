//! A generic [`Value`] which supports collation

use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::iter::FromIterator;
use std::ops::Deref;
use std::str::FromStr;

use async_hash::Hash;
use async_trait::async_trait;
use bytes::Bytes;
use destream::de;
use destream::de::Error as DestreamError;
use destream::de::Visitor as DestreamVisitor;
use destream::en;
use log::debug;
use safecast::{CastFrom, CastInto, TryCastFrom, TryCastInto};
use serde::de::{Deserialize, Deserializer, Error as SerdeError};
use serde::ser::{Serialize, SerializeMap, Serializer};
use sha2::digest::generic_array::GenericArray;
use sha2::digest::{Digest, Output};
use uuid::Uuid;

use tc_error::*;
use tcgeneric::*;

use super::{Link, TCString, Version};

pub use number_general::{
    Boolean, BooleanType, Complex, ComplexType, Float, FloatInstance, FloatType, Int, IntType,
    Number, NumberClass, NumberCollator, NumberInstance, NumberType, Trigonometry, UInt, UIntType,
};

const EMPTY_SEQ: [u8; 0] = [];
const EXPECTING: &'static str = "a TinyChain value, e.g. 1 or \"two\" or [3]";
const PREFIX: PathLabel = path_label(&["state", "scalar", "value"]);

/// The class of a [`Value`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueType {
    Bytes,
    Id,
    Link,
    None,
    Number(NumberType),
    String,
    Tuple,
    Value,
    Version,
}

impl ValueType {
    pub fn size(&self) -> Option<usize> {
        match self {
            Self::Number(nt) => Some(nt.size()),
            _ => None,
        }
    }

    pub fn try_cast<V>(&self, value: V) -> TCResult<Value>
    where
        Value: From<V>,
    {
        let on_err = |v: &Value| TCError::bad_request(format!("cannot cast into {} from", self), v);
        let value = Value::from(value);

        match self {
            Self::Bytes => match value {
                Value::Bytes(bytes) => Ok(Value::Bytes(bytes)),
                Value::Number(_) => Err(TCError::not_implemented("cast into Bytes from Number")),
                Value::String(s) => s
                    .try_cast_into(|s| TCError::bad_request("cannot cast into Bytes from", s))
                    .map(Value::Bytes),
                other => Err(TCError::bad_request("cannot cast into Bytes from", other)),
            },
            Self::Id => value.try_cast_into(on_err).map(Value::Id),
            Self::Link => value.try_cast_into(on_err).map(Value::String),
            Self::None => Ok(Value::None),
            Self::Number(nt) => value
                .try_cast_into(|v| TCError::bad_request("cannot cast into Number from", v))
                .map(|n| nt.cast(n))
                .map(Value::Number),
            Self::String => value.try_cast_into(on_err).map(Value::String),
            Self::Tuple => value.try_cast_into(on_err).map(Value::Tuple),
            Self::Value => Ok(value),
            Self::Version => match value {
                Value::String(s) => s.parse().map(Value::Version),
                Value::Tuple(t) => {
                    let (maj, min, rev) =
                        t.try_cast_into(|t| TCError::bad_request("invalid semantic version", t))?;
                    Ok(Value::Version(Version::from((maj, min, rev))))
                }
                other => Err(TCError::bad_request("cannot cast into Version from", other)),
            },
        }
    }
}

impl Default for ValueType {
    fn default() -> Self {
        Self::Value
    }
}

impl Class for ValueType {}

impl NativeClass for ValueType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        debug!("ValueType::from_path {}", TCPath::from(path));

        use ComplexType as CT;
        use FloatType as FT;
        use IntType as IT;
        use NumberType as NT;
        use UIntType as UT;

        if path.len() >= 3 && &path[..3] == &PREFIX[..] {
            if path.len() == 3 {
                Some(Self::default())
            } else if path.len() == 4 {
                match path[3].as_str() {
                    "bytes" => Some(Self::Bytes),
                    "id" => Some(Self::Id),
                    "link" => Some(Self::Link),
                    "number" => Some(Self::Number(NT::Number)),
                    "none" => Some(Self::None),
                    "string" => Some(Self::String),
                    "tuple" => Some(Self::Tuple),
                    "version" => Some(Self::Version),
                    _ => None,
                }
            } else if path.len() == 5 && &path[3] == "number" {
                match path[4].as_str() {
                    "bool" => Some(Self::Number(NT::Bool)),
                    "complex" => Some(Self::Number(NT::Complex(CT::Complex))),
                    "float" => Some(Self::Number(NT::Float(FT::Float))),
                    "int" => Some(Self::Number(NT::Int(IT::Int))),
                    "uint" => Some(Self::Number(NT::UInt(UT::UInt))),
                    _ => None,
                }
            } else if path.len() == 6 && &path[3] == "number" {
                match path[4].as_str() {
                    "complex" => match path[5].as_str() {
                        "32" => Some(Self::Number(NT::Complex(CT::C32))),
                        "64" => Some(Self::Number(NT::Complex(CT::C64))),
                        _ => None,
                    },
                    "float" => match path[5].as_str() {
                        "32" => Some(Self::Number(NT::Float(FT::F32))),
                        "64" => Some(Self::Number(NT::Float(FT::F64))),
                        _ => None,
                    },
                    "int" => match path[5].as_str() {
                        "16" => Some(Self::Number(NT::Int(IT::I16))),
                        "32" => Some(Self::Number(NT::Int(IT::I32))),
                        "64" => Some(Self::Number(NT::Int(IT::I64))),
                        _ => None,
                    },
                    "uint" => match path[5].as_str() {
                        "8" => Some(Self::Number(NT::UInt(UT::U8))),
                        "16" => Some(Self::Number(NT::UInt(UT::U16))),
                        "32" => Some(Self::Number(NT::UInt(UT::U32))),
                        "64" => Some(Self::Number(NT::UInt(UT::U64))),
                        _ => None,
                    },
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let prefix = TCPathBuf::from(PREFIX);

        match self {
            Self::Bytes => prefix.append(label("bytes")),
            Self::Id => prefix.append(label("id")),
            Self::Link => prefix.append(label("link")),
            Self::None => prefix.append(label("none")),
            Self::Number(nt) => {
                const N8: Label = label("8");
                const N16: Label = label("16");
                const N32: Label = label("32");
                const N64: Label = label("64");

                let prefix = prefix.append(label("number"));
                use NumberType as NT;
                match nt {
                    NT::Bool => prefix.append(label("bool")),
                    NT::Complex(ct) => {
                        let prefix = prefix.append(label("complex"));
                        use ComplexType as CT;
                        match ct {
                            CT::C32 => prefix.append(N32),
                            CT::C64 => prefix.append(N64),
                            CT::Complex => prefix,
                        }
                    }
                    NT::Float(ft) => {
                        let prefix = prefix.append(label("float"));
                        use FloatType as FT;
                        match ft {
                            FT::F32 => prefix.append(N32),
                            FT::F64 => prefix.append(N64),
                            FT::Float => prefix,
                        }
                    }
                    NT::Int(it) => {
                        let prefix = prefix.append(label("int"));
                        use IntType as IT;
                        match it {
                            IT::I8 => prefix.append(N16),
                            IT::I16 => prefix.append(N16),
                            IT::I32 => prefix.append(N32),
                            IT::I64 => prefix.append(N64),
                            IT::Int => prefix,
                        }
                    }
                    NT::UInt(ut) => {
                        let prefix = prefix.append(label("uint"));
                        use UIntType as UT;
                        match ut {
                            UT::U8 => prefix.append(N8),
                            UT::U16 => prefix.append(N16),
                            UT::U32 => prefix.append(N32),
                            UT::U64 => prefix.append(N64),
                            UT::UInt => prefix,
                        }
                    }
                    NT::Number => prefix,
                }
            }
            Self::String => prefix.append(label("string")),
            Self::Tuple => prefix.append(label("tuple")),
            Self::Value => prefix,
            Self::Version => prefix.append(label("version")),
        }
    }
}

impl Ord for ValueType {
    fn cmp(&self, other: &Self) -> Ordering {
        use Ordering::*;

        match (self, other) {
            (Self::Number(l), Self::Number(r)) => l.cmp(r),
            (l, r) if l == r => Equal,

            (Self::Value, _) => Greater,
            (_, Self::Value) => Less,

            (Self::Tuple, _) => Greater,
            (_, Self::Tuple) => Less,

            (Self::Link, _) => Greater,
            (_, Self::Link) => Less,

            (Self::String, _) => Greater,
            (_, Self::String) => Less,

            (Self::Id, _) => Greater,
            (_, Self::Id) => Less,

            (Self::Bytes, _) => Greater,
            (_, Self::Bytes) => Less,

            (Self::Number(_), _) => Greater,
            (_, Self::Number(_)) => Less,

            (Self::Version, _) => Greater,
            (_, Self::Version) => Less,

            (Self::None, Self::None) => Equal,
        }
    }
}

impl PartialOrd for ValueType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<NumberType> for ValueType {
    fn from(nt: NumberType) -> ValueType {
        ValueType::Number(nt)
    }
}

impl TryCastFrom<Value> for ValueType {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Link(l) if l.host().is_none() => Self::from_path(l.path()).is_some(),
            Value::String(s) => {
                if let Ok(path) = TCPathBuf::from_str(s.as_str()) {
                    Self::from_path(&path).is_some()
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        if let Some(path) = TCPathBuf::opt_cast_from(value) {
            Self::from_path(&path)
        } else {
            None
        }
    }
}

impl TryFrom<ValueType> for NumberType {
    type Error = TCError;

    fn try_from(vt: ValueType) -> TCResult<NumberType> {
        match vt {
            ValueType::Number(nt) => Ok(nt),
            other => Err(TCError::bad_request("expected a Number type, not", other)),
        }
    }
}

#[async_trait]
impl de::FromStream for ValueType {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        debug!("ValueType::from_stream");
        decoder.decode_any(ValueTypeVisitor).await
    }
}

impl<'en> en::IntoStream<'en> for ValueType {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.path(), Map::<Value>::default())?;
        map.end()
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bytes => f.write_str("type Bytes"),
            Self::Id => f.write_str("type Id"),
            Self::Link => f.write_str("type Link"),
            Self::None => f.write_str("type None"),
            Self::Number(nt) => write!(f, "type {}", nt),
            Self::String => f.write_str("type String"),
            Self::Tuple => f.write_str("type Tuple<Value>"),
            Self::Value => f.write_str("type Value"),
            Self::Version => f.write_str("type Version"),
        }
    }
}

struct ValueTypeVisitor;

impl ValueTypeVisitor {
    fn visit_path<E: DestreamError>(self, path: TCPathBuf) -> Result<ValueType, E> {
        ValueType::from_path(&path)
            .ok_or_else(|| DestreamError::invalid_value(path, Self::expecting()))
    }
}

#[async_trait]
impl DestreamVisitor for ValueTypeVisitor {
    type Value = ValueType;

    fn expecting() -> &'static str {
        "a Value type"
    }

    fn visit_string<E: DestreamError>(self, v: String) -> Result<Self::Value, E> {
        let path: TCPathBuf = v.parse().map_err(DestreamError::custom)?;
        self.visit_path(path)
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        if let Some(path) = access.next_key::<TCPathBuf>(()).await? {
            if let Ok(list) = access.next_value::<Tuple<Value>>(()).await {
                if list.is_empty() {
                    self.visit_path(path)
                } else {
                    Err(DestreamError::invalid_value(list, "empty list"))
                }
            } else if let Ok(map) = access.next_value::<Map<Value>>(()).await {
                if map.is_empty() {
                    self.visit_path(path)
                } else {
                    Err(DestreamError::invalid_value(map, "empty map"))
                }
            } else {
                Err(de::Error::custom("invalid Value class"))
            }
        } else {
            Err(DestreamError::invalid_length(0, Self::expecting()))
        }
    }
}

/// A generic value enum
#[derive(Clone, Eq, PartialEq)]
pub enum Value {
    Bytes(Bytes),
    Link(Link),
    Id(Id),
    None,
    Number(Number),
    String(TCString),
    Tuple(Tuple<Self>),
    Version(Version),
}

impl Value {
    /// Return a [`TCError`] if this `Value` is not none.
    #[inline]
    pub fn expect_none(&self) -> TCResult<()> {
        if self.is_none() {
            Ok(())
        } else {
            Err(TCError::bad_request("expected None but found", self))
        }
    }

    /// Return `true` if this `Value` is a default [`Link`], `Value::None`, or empty [`Tuple`].
    pub fn is_none(&self) -> bool {
        match self {
            Self::Link(link) => link.host().is_none() && link.path() == &TCPathBuf::default(),
            Self::None => true,
            Self::Tuple(tuple) => tuple.is_empty(),
            _ => false,
        }
    }

    /// Return `true` if this value is *not* a default [`Link`], `Value::None`, or empty [`Tuple`].
    pub fn is_some(&self) -> bool {
        !self.is_none()
    }

    /// Return `true` if this `Value` variant is a [`Tuple`].
    pub fn is_tuple(&self) -> bool {
        match self {
            Self::Tuple(_) => true,
            _ => false,
        }
    }

    pub fn into_type(self, class: ValueType) -> Option<Self> {
        if self.class() == class {
            return Some(self);
        }

        use ValueType as VT;

        match class {
            VT::Bytes => self.opt_cast_into().map(Self::Bytes),
            VT::Id => self.opt_cast_into().map(Self::Id),
            VT::Link => self.opt_cast_into().map(Self::Link),
            VT::None => Some(Self::None),
            VT::Number(nt) => Number::opt_cast_from(self)
                .map(|n| n.into_type(nt))
                .map(Self::Number),
            VT::String => Some(Value::String(self.to_string().into())),
            VT::Tuple => match self {
                Self::Tuple(tuple) => Some(Self::Tuple(tuple)),
                _ => None,
            },
            VT::Value => Some(self),
            VT::Version => self.opt_cast_into().map(Self::Version),
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Self::None
    }
}

impl Instance for Value {
    type Class = ValueType;

    fn class(&self) -> ValueType {
        use ValueType as VT;
        match self {
            Self::Bytes(_) => VT::Bytes,
            Self::Id(_) => VT::Id,
            Self::Link(_) => VT::Link,
            Self::None => VT::None,
            Self::Number(n) => VT::Number(n.class()),
            Self::String(_) => VT::String,
            Self::Tuple(_) => VT::Tuple,
            Self::Version(_) => VT::Version,
        }
    }
}

impl<D: Digest> Hash<D> for Value {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self)
    }
}

impl<'a, D: Digest> Hash<D> for &'a Value {
    fn hash(self) -> Output<D> {
        match self {
            Value::Bytes(bytes) => D::digest(bytes),
            Value::Id(id) => Hash::<D>::hash(id),
            Value::Link(link) => Hash::<D>::hash(link),
            Value::None => GenericArray::default(),
            Value::Number(n) => Hash::<D>::hash(*n),
            Value::String(s) => Hash::<D>::hash(s.as_str()),
            Value::Tuple(tuple) => Hash::<D>::hash(tuple.deref()),
            Value::Version(v) => Hash::<D>::hash(*v),
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(ValueVisitor)
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Bytes(bytes) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(&self.class().path().to_string(), &base64::encode(&bytes))?;
                map.end()
            }
            Self::Id(id) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(id.as_str(), &EMPTY_SEQ)?;
                map.end()
            }
            Self::Link(link) => {
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry(link, &EMPTY_SEQ)?;
                map.end()
            }
            Self::None => serializer.serialize_unit(),
            Self::Number(n) => match n {
                Number::Complex(c) => {
                    let mut map = serializer.serialize_map(Some(1))?;
                    map.serialize_entry(&self.class().path().to_string(), &Number::Complex(*c))?;
                    map.end()
                }
                n => n.serialize(serializer),
            },
            Self::String(s) => s.serialize(serializer),
            Self::Tuple(t) => t.as_slice().serialize(serializer),
            Self::Version(v) => v.serialize(serializer),
        }
    }
}

impl<T> FromIterator<T> for Value
where
    Value: From<T>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let tuple = Tuple::<Value>::from_iter(iter.into_iter().map(Value::from));
        Self::Tuple(tuple)
    }
}

#[async_trait]
impl de::FromStream for Value {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_context: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_any(ValueVisitor).await
    }
}

impl<'en> en::ToStream<'en> for Value {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;

        match self {
            Self::Bytes(bytes) => encoder.encode_bytes(bytes),
            Self::Id(id) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(id, &EMPTY_SEQ)?;
                map.end()
            }
            Self::Link(link) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(link, &EMPTY_SEQ)?;
                map.end()
            }
            Self::None => encoder.encode_unit(),
            Self::Number(n) => match n {
                Number::Complex(c) => {
                    let mut map = encoder.encode_map(Some(1))?;
                    map.encode_entry(self.class().path().to_string(), Number::Complex(*c))?;
                    map.end()
                }
                n => n.to_stream(encoder),
            },
            Self::String(s) => s.to_stream(encoder),
            Self::Tuple(t) => t.to_stream(encoder),
            Self::Version(v) => v.to_stream(encoder),
        }
    }
}

impl<'en> en::IntoStream<'en> for Value {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;

        match self {
            Self::Bytes(bytes) => encoder.encode_bytes(&bytes),
            Self::Id(id) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(id, &EMPTY_SEQ)?;
                map.end()
            }
            Self::Link(link) => {
                let mut map = encoder.encode_map(Some(1))?;
                map.encode_entry(link, &EMPTY_SEQ)?;
                map.end()
            }
            Self::None => encoder.encode_unit(),
            Self::Number(n) => match n {
                Number::Complex(c) => {
                    let mut map = encoder.encode_map(Some(1))?;
                    map.encode_entry(self.class().path().to_string(), Number::Complex(c))?;
                    map.end()
                }
                n => n.into_stream(encoder),
            },
            Self::String(s) => s.into_stream(encoder),
            Self::Tuple(t) => t.into_inner().into_stream(encoder),
            Self::Version(v) => v.into_stream(encoder),
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Self::Number(Number::from(b))
    }
}

impl From<Bytes> for Value {
    fn from(bytes: Bytes) -> Self {
        Self::Bytes(bytes)
    }
}

impl From<Id> for Value {
    fn from(id: Id) -> Self {
        Self::Id(id)
    }
}

impl From<Link> for Value {
    fn from(link: Link) -> Self {
        Self::Link(link)
    }
}

impl From<Number> for Value {
    fn from(n: Number) -> Self {
        Self::Number(n)
    }
}

impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(value) => value.into(),
            None => Self::None,
        }
    }
}

impl From<TCPathBuf> for Value {
    fn from(path: TCPathBuf) -> Value {
        Value::Link(path.into())
    }
}

impl From<Tuple<Value>> for Value {
    fn from(tuple: Tuple<Value>) -> Self {
        Self::Tuple(tuple)
    }
}

impl From<Vec<Value>> for Value {
    fn from(tuple: Vec<Value>) -> Self {
        Self::Tuple(tuple.into())
    }
}

impl From<usize> for Value {
    fn from(n: usize) -> Self {
        Self::Number((n as u64).into())
    }
}

impl From<u64> for Value {
    fn from(n: u64) -> Self {
        Self::Number(n.into())
    }
}

impl<T1: CastInto<Value>, T2: CastInto<Value>> CastFrom<(T1, T2)> for Value {
    fn cast_from(value: (T1, T2)) -> Self {
        Value::Tuple(vec![value.0.cast_into(), value.1.cast_into()].into())
    }
}

impl TryFrom<Value> for Number {
    type Error = TCError;

    fn try_from(value: Value) -> TCResult<Self> {
        match value {
            Value::Number(number) => Ok(number),
            other => Err(TCError::bad_request("expected Number but found", other)),
        }
    }
}

impl TryFrom<Value> for Tuple<Value> {
    type Error = TCError;

    fn try_from(value: Value) -> TCResult<Self> {
        match value {
            Value::Tuple(tuple) => Ok(tuple),
            other => Err(TCError::bad_request("expected Tuple but found", other)),
        }
    }
}

impl TryCastFrom<Value> for Bytes {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Bytes(_) => true,
            Value::Tuple(tuple) => Vec::<u8>::can_cast_from(tuple),
            Value::String(s) => Self::can_cast_from(s),
            Value::None => true,
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Bytes> {
        match value {
            Value::Bytes(bytes) => Some(bytes),
            Value::Tuple(tuple) => Vec::<u8>::opt_cast_from(tuple).map(Bytes::from),
            Value::String(s) => Self::opt_cast_from(s),
            Value::None => Some(Bytes::new()),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for Id {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Id(_) => true,
            Value::String(s) => Self::can_cast_from(s),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Id(id) => Some(id),
            Value::String(s) => Self::opt_cast_from(s),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for Number {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Bytes(_) => false,
            Value::Id(id) => f64::from_str(id.as_str()).is_ok(),
            Value::Link(_) => false,
            Value::None => true,
            Value::Number(_) => true,
            Value::Tuple(t) if t.len() == 1 => Self::can_cast_from(&t[0]),
            Value::Tuple(t) if t.len() == 2 => {
                Self::can_cast_from(&t[0]) && Self::can_cast_from(&t[1])
            }
            Value::Tuple(_) => false,
            Value::String(s) => f64::from_str(&s).is_ok(),
            Value::Version(_) => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Bytes(_) => None,
            Value::Id(id) => f64::from_str(id.as_str())
                .map(Float::F64)
                .map(Self::Float)
                .ok(),
            Value::Link(_) => None,
            Value::None => Some(false.into()),
            Value::Number(n) => Some(n),
            Value::Tuple(mut t) if t.len() == 1 => Self::opt_cast_from(t.pop().unwrap()),
            Value::Tuple(mut t) if t.len() == 2 => {
                let im = Self::opt_cast_from(t.pop().unwrap());
                let re = Self::opt_cast_from(t.pop().unwrap());
                match (re, im) {
                    (Some(Number::Float(re)), Some(Number::Float(im))) => match (re, im) {
                        (Float::F32(re), Float::F32(im)) => {
                            let c = Complex::from([re, im]);
                            Some(Self::Complex(c))
                        }
                        (re, im) => {
                            let c = Complex::from([f64::cast_from(re), f64::cast_from(im)]);
                            Some(Self::Complex(c))
                        }
                    },
                    _ => None,
                }
            }
            Value::Tuple(_) => None,
            Value::String(s) => Number::from_str(&s).ok(),
            Value::Version(_) => None,
        }
    }
}

impl TryCastFrom<Value> for TCString {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Link(_) => true,
            Value::Id(_) => true,
            Value::Number(_) => true,
            Value::String(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Link(link) => Self::opt_cast_from(link),
            Value::Id(id) => Self::opt_cast_from(id),
            Value::Number(n) => Self::opt_cast_from(n),
            Value::String(s) => Some(s),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for Uuid {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Bytes(bytes) => bytes.len() == 16,
            Value::Id(id) => Uuid::from_str(id.as_str()).is_ok(),
            Value::String(s) => Uuid::from_str(s).is_ok(),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Bytes(bytes) if bytes.len() == 16 => {
                let bytes = bytes.to_vec().try_into().expect("16-byte UUID");
                Some(Uuid::from_bytes(bytes))
            }
            Value::Id(id) => id.as_str().parse().ok(),
            Value::String(s) => s.parse().ok(),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for Version {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::String(s) => Version::from_str(s).is_ok(),
            Value::Tuple(t) => <(u32, u32, u32) as TryCastFrom<Tuple<Value>>>::can_cast_from(t),
            Value::Version(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::String(s) => Version::from_str(&s).ok(),

            Value::Tuple(t) => t
                .opt_cast_into()
                .map(|tuple: (u32, u32, u32)| Version::from(tuple)),

            Value::Version(version) => Some(version),

            _ => None,
        }
    }
}

impl TryCastFrom<Value> for bool {
    fn can_cast_from(value: &Value) -> bool {
        Number::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        Number::opt_cast_from(value).map(|n| n.cast_into())
    }
}

impl TryCastFrom<Value> for u8 {
    fn can_cast_from(value: &Value) -> bool {
        Number::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        Number::opt_cast_from(value).map(|n| n.cast_into())
    }
}

impl TryCastFrom<Value> for usize {
    fn can_cast_from(value: &Value) -> bool {
        Number::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        Number::opt_cast_from(value).map(|n| n.cast_into())
    }
}

impl TryCastFrom<Value> for u32 {
    fn can_cast_from(value: &Value) -> bool {
        Number::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        Number::opt_cast_from(value).map(|n| n.cast_into())
    }
}

impl TryCastFrom<Value> for u64 {
    fn can_cast_from(value: &Value) -> bool {
        Number::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        Number::opt_cast_from(value).map(|n| n.cast_into())
    }
}

impl TryCastFrom<Value> for i64 {
    fn can_cast_from(value: &Value) -> bool {
        Number::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        Number::opt_cast_from(value).map(|n| n.cast_into())
    }
}

impl TryCastFrom<Value> for TCPathBuf {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Id(_) => true,
            Value::Link(link) => link.host().is_none(),
            Value::String(s) => Self::from_str(s).is_ok(),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Id(id) => Some(Self::from(id)),
            Value::Link(link) if link.host().is_none() => Some(link.into_path()),
            Value::String(s) => Self::from_str(&s).ok(),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<Value>> TryCastFrom<Value> for (T,) {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<Value>, T2: TryCastFrom<Value>> TryCastFrom<Value> for (T1, T2) {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<Value>, T2: TryCastFrom<Value>, T3: TryCastFrom<Value>> TryCastFrom<Value>
    for (T1, T2, T3)
{
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T: Clone + TryCastFrom<Value>> TryCastFrom<Value> for Map<T> {
    fn can_cast_from(value: &Value) -> bool {
        Vec::<(Id, T)>::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        Vec::<(Id, T)>::opt_cast_from(value).map(|entries| entries.into_iter().collect())
    }
}

impl<T: TryCastFrom<Value>> TryCastFrom<Value> for Vec<T> {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T: Clone + TryCastFrom<Value>> TryCastFrom<Value> for Tuple<T> {
    fn can_cast_from(value: &Value) -> bool {
        Vec::<T>::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        Vec::<T>::opt_cast_from(value).map(Tuple::from)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bytes(bytes) => write!(f, "({} bytes)", bytes.len()),
            Self::Id(id) => write!(f, "{}: {:?}", ValueType::Id, id.as_str()),
            Self::Link(link) => write!(f, "{}: {:?}", ValueType::Link, link),
            Self::None => f.write_str("None"),
            Self::Number(n) => fmt::Debug::fmt(n, f),
            Self::String(s) => write!(f, "{}: {}", ValueType::String, s),
            Self::Tuple(t) => write!(
                f,
                "{}: ({})",
                ValueType::Tuple,
                t.iter()
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Self::Version(v) => fmt::Debug::fmt(v, f),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Bytes(bytes) => write!(f, "({} bytes)", bytes.len()),
            Self::Id(id) => f.write_str(id.as_str()),
            Self::Link(link) => fmt::Display::fmt(link, f),
            Self::None => f.write_str("None"),
            Self::Number(n) => fmt::Display::fmt(n, f),
            Self::String(s) => f.write_str(s),
            Self::Tuple(t) => write!(
                f,
                "({})",
                t.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Self::Version(v) => fmt::Display::fmt(v, f),
        }
    }
}

/// A struct for deserializing a [`Value`] which implements [`destream::de::Visitor`]
/// and [`serde::de::Visitor`].
#[derive(Default)]
pub struct ValueVisitor;

impl ValueVisitor {
    fn visit_number<E, N>(self, n: N) -> Result<Value, E>
    where
        Number: CastFrom<N>,
    {
        Ok(Value::Number(Number::cast_from(n)))
    }

    pub fn visit_map_value<'de, A: serde::de::MapAccess<'de>>(
        class: ValueType,
        mut map: A,
    ) -> Result<Value, A::Error> {
        use ValueType as VT;

        return match class {
            VT::Bytes => {
                let encoded = map.next_value::<&str>()?;
                base64::decode(encoded)
                    .map(Bytes::from)
                    .map(Value::Bytes)
                    .map_err(serde::de::Error::custom)
            }
            VT::Id => {
                let id: &str = map.next_value()?;
                Id::from_str(id).map(Value::Id).map_err(A::Error::custom)
            }
            VT::Link => {
                let link = map.next_value()?;
                Ok(Value::Link(link))
            }
            VT::None => {
                let _ = map.next_value::<()>()?;
                Ok(Value::None)
            }
            VT::Number(nt) => {
                let n = map.next_value::<Number>()?;
                Ok(Value::Number(n.into_type(nt)))
            }
            VT::String => {
                let s = map.next_value()?;
                Ok(Value::String(s))
            }
            VT::Tuple => {
                let t = map.next_value::<Vec<Value>>()?;
                Ok(Value::Tuple(t.into()))
            }
            VT::Value => {
                let v = map.next_value()?;
                Ok(v)
            }
            VT::Version => {
                let v = map.next_value()?;
                Ok(Value::Version(v))
            }
        };
    }

    pub async fn visit_map_value_async<A: destream::MapAccess>(
        class: ValueType,
        map: &mut A,
    ) -> Result<Value, A::Error> {
        use ValueType as VT;

        if let Ok(map) = map.next_value::<Map<Value>>(()).await {
            return if map.is_empty() {
                Ok(Value::Link(class.path().into()))
            } else {
                Err(de::Error::invalid_value(map, "a Value class"))
            };
        }

        return match class {
            VT::Bytes => {
                let bytes = map.next_value(()).await?;
                Ok(Value::Bytes(bytes))
            }
            VT::Id => {
                let id = map.next_value(()).await?;
                Ok(Value::Id(id))
            }
            VT::Link => {
                let link = map.next_value(()).await?;
                Ok(Value::Link(link))
            }
            VT::None => {
                let _ = map.next_value::<()>(()).await?;
                Ok(Value::None)
            }
            VT::Number(nt) => {
                let n = map.next_value::<Number>(()).await?;
                Ok(Value::Number(n.into_type(nt)))
            }
            VT::String => {
                let s = map.next_value(()).await?;
                Ok(Value::String(s))
            }
            VT::Tuple => {
                let t = map.next_value::<Vec<Value>>(()).await?;
                Ok(Value::Tuple(t.into()))
            }
            VT::Value => {
                let v = map.next_value(()).await?;
                Ok(v)
            }
            VT::Version => {
                let v = map.next_value(()).await?;
                Ok(Value::Version(v))
            }
        };
    }
}

impl<'de> serde::de::Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(EXPECTING)
    }

    fn visit_bool<E: SerdeError>(self, b: bool) -> Result<Self::Value, E> {
        self.visit_number(b)
    }

    fn visit_i8<E: SerdeError>(self, i: i8) -> Result<Self::Value, E> {
        self.visit_number(i as i16)
    }

    fn visit_i16<E: SerdeError>(self, i: i16) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_i32<E: SerdeError>(self, i: i32) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_i64<E: SerdeError>(self, i: i64) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_u8<E: SerdeError>(self, u: u8) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u16<E: SerdeError>(self, u: u16) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u32<E: SerdeError>(self, u: u32) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u64<E: SerdeError>(self, u: u64) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_f32<E: SerdeError>(self, f: f32) -> Result<Self::Value, E> {
        self.visit_number(f)
    }

    fn visit_f64<E: SerdeError>(self, f: f64) -> Result<Self::Value, E> {
        self.visit_number(f)
    }

    fn visit_str<E: SerdeError>(self, s: &str) -> Result<Self::Value, E> {
        Ok(Value::String(s.to_string().into()))
    }

    fn visit_borrowed_str<E: SerdeError>(self, s: &'de str) -> Result<Self::Value, E> {
        Ok(Value::String(s.to_string().into()))
    }

    fn visit_string<E: SerdeError>(self, s: String) -> Result<Self::Value, E> {
        Ok(Value::String(s.into()))
    }

    fn visit_byte_buf<E: SerdeError>(self, buf: Vec<u8>) -> Result<Self::Value, E> {
        Ok(Value::Bytes(buf.into()))
    }

    fn visit_unit<E: SerdeError>(self) -> Result<Self::Value, E> {
        Ok(Value::None)
    }

    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut value = if let Some(len) = seq.size_hint() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };

        while let Some(element) = seq.next_element()? {
            value.push(element)
        }

        Ok(Value::Tuple(value.into()))
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        if let Some(key) = map.next_key::<String>()? {
            if let Ok(link) = Link::from_str(&key) {
                if link.host().is_none() {
                    use ValueType as VT;
                    if let Some(class) = VT::from_path(link.path()) {
                        return Self::visit_map_value(class, map);
                    }
                }

                let value = map.next_value::<Vec<Value>>()?;
                if value.is_empty() {
                    Ok(Value::Link(link))
                } else {
                    Err(A::Error::invalid_length(value.len(), &"an empty list"))
                }
            } else if let Ok(id) = Id::from_str(&key) {
                let value = map.next_value::<Vec<Value>>()?;
                if value.is_empty() {
                    Ok(Value::Id(id))
                } else {
                    Err(A::Error::invalid_length(value.len(), &"an empty list"))
                }
            } else {
                Err(A::Error::custom(format!(
                    "expected a Link but found {}",
                    key
                )))
            }
        } else {
            Err(A::Error::custom("expected a Link but found an empty map"))
        }
    }
}

#[async_trait]
impl destream::de::Visitor for ValueVisitor {
    type Value = Value;

    fn expecting() -> &'static str {
        EXPECTING
    }

    fn visit_bool<E: DestreamError>(self, b: bool) -> Result<Self::Value, E> {
        self.visit_number(b)
    }

    fn visit_i8<E: DestreamError>(self, i: i8) -> Result<Self::Value, E> {
        self.visit_number(i as i16)
    }

    fn visit_i16<E: DestreamError>(self, i: i16) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_i32<E: DestreamError>(self, i: i32) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_i64<E: DestreamError>(self, i: i64) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_u8<E: DestreamError>(self, u: u8) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u16<E: DestreamError>(self, u: u16) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u32<E: DestreamError>(self, u: u32) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u64<E: DestreamError>(self, u: u64) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_f32<E: DestreamError>(self, f: f32) -> Result<Self::Value, E> {
        self.visit_number(f)
    }

    fn visit_f64<E: DestreamError>(self, f: f64) -> Result<Self::Value, E> {
        self.visit_number(f)
    }

    fn visit_string<E: DestreamError>(self, s: String) -> Result<Self::Value, E> {
        Ok(Value::String(s.into()))
    }

    fn visit_byte_buf<E: DestreamError>(self, buf: Vec<u8>) -> Result<Self::Value, E> {
        Ok(Value::Bytes(buf.into()))
    }

    fn visit_unit<E: DestreamError>(self) -> Result<Self::Value, E> {
        Ok(Value::None)
    }

    fn visit_none<E: DestreamError>(self) -> Result<Self::Value, E> {
        Ok(Value::None)
    }

    async fn visit_map<A: destream::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        use ValueType as VT;

        if let Some(key) = map.next_key::<String>(()).await? {
            if let Ok(link) = Link::from_str(&key) {
                if link.host().is_none() {
                    if let Some(class) = VT::from_path(link.path()) {
                        return Self::visit_map_value_async(class, &mut map).await;
                    }
                }

                let value = map.next_value::<Vec<Value>>(()).await?;
                if value.is_empty() {
                    Ok(Value::Link(link))
                } else {
                    Err(A::Error::invalid_length(value.len(), "empty sequence"))
                }
            } else if let Ok(id) = Id::from_str(&key) {
                let value = map.next_value::<Vec<Value>>(()).await?;
                if value.is_empty() {
                    Ok(Value::Id(id))
                } else {
                    Err(A::Error::invalid_length(value.len(), "empty sequence"))
                }
            } else {
                Err(DestreamError::invalid_value(key, "a Link"))
            }
        } else {
            Err(DestreamError::invalid_value("empty map", "a Link"))
        }
    }

    async fn visit_seq<A: destream::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut value = if let Some(len) = seq.size_hint() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };

        while let Some(element) = seq.next_element(()).await? {
            value.push(element)
        }

        Ok(Value::Tuple(value.into()))
    }
}
