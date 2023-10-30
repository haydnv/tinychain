use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use destream::de::Error as DestreamError;
use destream::de::Visitor as DestreamVisitor;
use destream::{de, en};
use email_address_parser::EmailAddress;
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tcgeneric::*;

use super::number::{ComplexType, FloatType, IntType, NumberClass, NumberType, UIntType};
use super::value::Value;
use super::version::Version;

const PREFIX: PathLabel = path_label(&["state", "scalar", "value"]);

/// The class of a [`Value`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueType {
    Bytes,
    Email,
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
        let on_err = |v: &Value| bad_request!("cannot cast into {} from {}", self, v);
        let value = Value::from(value);

        match self {
            Self::Bytes => match value {
                Value::Bytes(bytes) => Ok(Value::Bytes(bytes)),
                Value::Number(_) => Err(not_implemented!("cast into Bytes from Number")),
                Value::String(s) => {
                    Bytes::try_cast_from(s, |s| bad_request!("cannot cast into Bytes from {}", s))
                        .map(Vec::from)
                        .map(Arc::from)
                        .map(Value::Bytes)
                }
                other => Err(bad_request!("cannot cast into Bytes from {}", other)),
            },
            Self::Email => match value {
                Value::Email(email) => Ok(Value::Email(email)),

                Value::Id(id) => parse_email(id.as_str())
                    .map(Arc::new)
                    .map(Value::Email)
                    .ok_or_else(|| bad_request!("cannot cast into an email address from {}", id)),

                Value::String(s) => parse_email(s.as_str())
                    .map(Arc::new)
                    .map(Value::Email)
                    .ok_or_else(|| bad_request!("cannot cast into an email address from {}", s)),

                other => Err(bad_request!(
                    "cannot cast into an email address from {}",
                    other
                )),
            },
            Self::Id => value.try_cast_into(on_err).map(Value::Id),
            Self::Link => value.try_cast_into(on_err).map(Value::String),
            Self::None => Ok(Value::None),
            Self::Number(nt) => value
                .try_cast_into(|v| bad_request!("cannot cast into Number from {}", v))
                .map(|n| nt.cast(n))
                .map(Value::Number),

            Self::String => value.try_cast_into(on_err).map(Value::String),
            Self::Tuple => value.try_cast_into(on_err).map(Value::Tuple),
            Self::Value => Ok(value),
            Self::Version => match value {
                Value::Id(id) => id.as_str().parse().map(Value::Version),
                Value::String(s) => s.as_str().parse().map(Value::Version),
                Value::Tuple(t) => {
                    let (maj, min, rev) =
                        t.try_cast_into(|t| bad_request!("invalid semantic version {}", t))?;

                    Ok(Value::Version(Version::from((maj, min, rev))))
                }
                Value::Version(v) => Ok(Value::Version(v)),
                other => Err(bad_request!("cannot cast into Version from {}", other)),
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
                    "email" => Some(Self::Email),
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
            Self::Email => prefix.append(label("email")),
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

            (Self::Email, _) => Greater,
            (_, Self::Email) => Less,

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
            other => Err(TCError::unexpected(other, "a Number class")),
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

impl<'en> en::ToStream<'en> for ValueType {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
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
            Self::Email => f.write_str("type Email"),
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
            if let Ok(map) = access.next_value::<Map<Value>>(()).await {
                if map.is_empty() {
                    self.visit_path(path)
                } else {
                    Err(de::Error::custom(format!(
                        "invalid specification {:?} for Value class {}",
                        map, path
                    )))
                }
            } else if let Ok(list) = access.next_value::<Tuple<Value>>(()).await {
                if list.is_empty() {
                    self.visit_path(path)
                } else {
                    Err(de::Error::custom(format!(
                        "invalid specification {:?} for Value class {}",
                        list, path
                    )))
                }
            } else {
                Err(de::Error::custom(format!(
                    "invalid specification for Value class {}",
                    path
                )))
            }
        } else {
            Err(DestreamError::invalid_length(0, Self::expecting()))
        }
    }
}

fn parse_email(s: &str) -> Option<EmailAddress> {
    EmailAddress::parse(s.as_ref(), None)
}
