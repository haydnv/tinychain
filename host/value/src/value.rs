use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::Error as DestreamError;
use destream::{Decoder, EncodeMap, Encoder, FromStream, IntoStream, ToStream};
use log::debug;
use safecast::{CastFrom, TryCastFrom, TryCastInto};
use serde::de::{Deserialize, Deserializer, Error as SerdeError};
use serde::ser::{Serialize, SerializeMap, Serializer};

use generic::*;

use super::Link;

pub use number_general::*;

const EXPECTING: &'static str = "a Tinychain value, e.g. 1 or \"two\" or [3]";
const PREFIX: PathLabel = path_label(&["state", "scalar", "value"]);

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ValueType {
    Link,
    None,
    Number(NumberType),
    String,
    Tuple,
    Value,
}

impl Default for ValueType {
    fn default() -> Self {
        Self::Value
    }
}

impl Class for ValueType {
    type Instance = Value;
}

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
                    "link" => Some(Self::Link),
                    "number" => Some(Self::Number(NT::Number)),
                    "none" => Some(Self::None),
                    "string" => Some(Self::String),
                    "tuple" => Some(Self::Tuple),
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

            (Self::None, _) => Less,
            (_, Self::None) => Greater,
        }
    }
}

impl PartialOrd for ValueType {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Link => f.write_str("Link"),
            Self::None => f.write_str("None"),
            Self::Number(nt) => fmt::Display::fmt(nt, f),
            Self::String => f.write_str("String"),
            Self::Tuple => f.write_str("Tuple<Value>"),
            Self::Value => f.write_str("Value"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Value {
    Link(Link),
    None,
    Number(Number),
    String(String),
    Tuple(Tuple<Self>),
}

impl Value {
    pub fn is_none(&self) -> bool {
        match self {
            Self::Link(link) => link == &TCPathBuf::default(),
            Self::None => true,
            Self::Tuple(tuple) => tuple.is_empty(),
            _ => false,
        }
    }

    pub fn is_some(&self) -> bool {
        !self.is_none()
    }

    pub fn into_type(self, class: ValueType) -> Option<Self> {
        if self.class() == class {
            return Some(self);
        }

        use ValueType as VT;

        match class {
            VT::Link => self.opt_cast_into().map(Self::Link),
            VT::None => Some(Self::None),
            VT::Number(nt) => Number::opt_cast_from(self)
                .map(|n| n.into_type(nt))
                .map(Self::Number),
            VT::String => Some(Value::String(self.to_string())),
            VT::Tuple => match self {
                Self::Tuple(tuple) => Some(Self::Tuple(tuple)),
                _ => None,
            },
            VT::Value => Some(self),
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
            Self::Link(_) => VT::Link,
            Self::None => VT::None,
            Self::Number(n) => VT::Number(n.class()),
            Self::String(_) => VT::String,
            Self::Tuple(_) => VT::Tuple,
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
            Self::Link(link) => link.serialize(serializer),
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
        }
    }
}

#[async_trait]
impl FromStream for Value {
    type Context = ();

    async fn from_stream<D: Decoder>(_context: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_any(ValueVisitor).await
    }
}

impl<'en> ToStream<'en> for Value {
    fn to_stream<E: Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Link(link) => link.to_stream(encoder),
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
        }
    }
}

impl<'en> IntoStream<'en> for Value {
    fn into_stream<E: Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Link(link) => link.into_stream(encoder),
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
        }
    }
}

impl CastFrom<Value> for String {
    fn cast_from(value: Value) -> String {
        value.to_string()
    }
}

impl TryCastFrom<Value> for Id {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::String(s) => Self::can_cast_from(s),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::String(s) => Self::opt_cast_from(s),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for Number {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Link(_) => false,
            Value::None => true,
            Value::Number(_) => true,
            Value::Tuple(t) if t.is_empty() => true,
            Value::Tuple(t) if t.len() == 1 => Self::can_cast_from(&t[0]),
            Value::Tuple(t) if t.len() == 2 => {
                Self::can_cast_from(&t[0]) && Self::can_cast_from(&t[1])
            }
            Value::Tuple(_) => false,
            Value::String(s) => f64::from_str(&s).is_ok(),
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Link(_) => None,
            Value::None => Some(false.into()),
            Value::Number(n) => Some(n),
            Value::Tuple(t) if t.is_empty() => Some(false.into()),
            Value::Tuple(mut t) if t.len() == 1 => Self::opt_cast_from(t.pop().unwrap()),
            Value::Tuple(mut t) if t.len() == 2 => {
                let im = Self::opt_cast_from(t.pop().unwrap());
                let re = Self::opt_cast_from(t.pop().unwrap());
                match (re, im) {
                    (Some(re), Some(im)) => {
                        let c = Complex::cast_from((re, im));
                        Some(Self::Complex(c))
                    }
                    _ => None,
                }
            }
            Value::Tuple(_) => None,
            Value::String(s) => f64::from_str(&s).map(Float::F64).map(Self::Float).ok(),
        }
    }
}

impl TryCastFrom<Value> for TCPathBuf {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::String(s) = value {
            Self::from_str(s).is_ok()
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        if let Value::String(s) = value {
            Self::from_str(&s).ok()
        } else {
            None
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

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Link(link) => fmt::Display::fmt(link, f),
            Self::None => f.write_str("none"),
            Self::Number(n) => fmt::Display::fmt(n, f),
            Self::String(s) => f.write_str(s),
            Self::Tuple(t) => write!(
                f,
                "[{}]",
                t.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}

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
        };
    }

    pub async fn visit_map_value_async<A: destream::MapAccess>(
        class: ValueType,
        map: &mut A,
    ) -> Result<Value, A::Error> {
        use ValueType as VT;

        return match class {
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
        Ok(Value::String(s.to_string()))
    }

    fn visit_borrowed_str<E: SerdeError>(self, s: &'de str) -> Result<Self::Value, E> {
        Ok(Value::String(s.to_string()))
    }

    fn visit_string<E: SerdeError>(self, s: String) -> Result<Self::Value, E> {
        Ok(Value::String(s))
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
        Ok(Value::String(s))
    }

    fn visit_byte_buf<E: DestreamError>(self, _buf: Vec<u8>) -> Result<Self::Value, E> {
        unimplemented!()
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
