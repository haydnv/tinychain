use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::{
    de, Decoder, EncodeMap, Encoder, FromStream, IntoStream, MapAccess, SeqAccess, ToStream,
};
use number_general::*;
use safecast::{CastFrom, TryCastFrom};

use generic::*;

pub mod link;

pub use link::*;

const PREFIX: PathLabel = path_label(&["sbin", "value"]);

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ValueType {
    Link,
    None,
    Number(NumberType),
    String,
    Tuple,
    Value,
}

impl Class for ValueType {
    type Instance = Value;

    fn from_path(path: &[PathSegment]) -> Option<Self> {
        use ComplexType as CT;
        use NumberType as NT;

        if path.len() >= 2 && &path[..2] == &PREFIX[..] {
            if path.len() == 2 {
                Some(Self::Value)
            } else if path.len() == 3 {
                match path[2].as_str() {
                    "link" => Some(Self::Link),
                    "none" => Some(Self::None),
                    "string" => Some(Self::String),
                    "tuple" => Some(Self::Tuple),
                    _ => None,
                }
            } else if path.len() == 4 {
                match (path[2].as_str(), path[3].as_str()) {
                    ("number", "complex") => Some(Self::Number(NT::Complex(CT::Complex))),
                    _ => None,
                }
            } else if path.len() == 5 {
                match (path[2].as_str(), path[3].as_str(), path[4].as_str()) {
                    ("number", "complex", "32") => Some(Self::Number(NT::Complex(CT::C32))),
                    ("number", "complex", "64") => Some(Self::Number(NT::Complex(CT::C64))),
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
            Self::None => true,
            Self::Tuple(tuple) => tuple.is_empty(),
            _ => false,
        }
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

#[async_trait]
impl FromStream for Value {
    async fn from_stream<D: Decoder>(decoder: &mut D) -> Result<Self, D::Error> {
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
            Self::Tuple(t) => t.as_slice().into_stream(encoder),
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

struct ValueVisitor;

impl ValueVisitor {
    fn visit_number<E: de::Error, N>(self, n: N) -> Result<Value, E>
    where
        Number: CastFrom<N>,
    {
        Ok(Value::Number(Number::cast_from(n)))
    }
}

#[async_trait]
impl de::Visitor for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "a Tinychain value, e.g. 1 or \"two\" or [3]")
    }

    fn visit_bool<E: de::Error>(self, b: bool) -> Result<Self::Value, E> {
        self.visit_number(b)
    }

    fn visit_i8<E: de::Error>(self, i: i8) -> Result<Self::Value, E> {
        self.visit_number(i as i16)
    }

    fn visit_i16<E: de::Error>(self, i: i16) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_i32<E: de::Error>(self, i: i32) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_i64<E: de::Error>(self, i: i64) -> Result<Self::Value, E> {
        self.visit_number(i)
    }

    fn visit_u8<E: de::Error>(self, u: u8) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u16<E: de::Error>(self, u: u16) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u32<E: de::Error>(self, u: u32) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_u64<E: de::Error>(self, u: u64) -> Result<Self::Value, E> {
        self.visit_number(u)
    }

    fn visit_f32<E: de::Error>(self, f: f32) -> Result<Self::Value, E> {
        self.visit_number(f)
    }

    fn visit_f64<E: de::Error>(self, f: f64) -> Result<Self::Value, E> {
        self.visit_number(f)
    }

    fn visit_string<E: de::Error>(self, s: String) -> Result<Self::Value, E> {
        Ok(Value::String(s))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(Value::None)
    }

    async fn visit_map<A: MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        if let Some(key) = map.next_key::<String>().await? {
            if let Ok(link) = Link::from_str(&key) {
                if link.host().is_none() {
                    use ValueType as VT;
                    if let Some(class) = VT::from_path(link.path()) {
                        return match class {
                            VT::Link => {
                                let link = map.next_value().await?;
                                Ok(Value::Link(link))
                            }
                            VT::None => {
                                let _ = map.next_value::<()>().await?;
                                Ok(Value::None)
                            }
                            VT::Number(nt) => {
                                let n = map.next_value::<Number>().await?;
                                Ok(Value::Number(n.into_type(nt)))
                            }
                            VT::String => {
                                let s = map.next_value().await?;
                                Ok(Value::String(s))
                            }
                            VT::Tuple => {
                                let t = map.next_value::<Vec<Value>>().await?;
                                Ok(Value::Tuple(t.into()))
                            }
                            VT::Value => {
                                let v = map.next_value().await?;
                                Ok(v)
                            }
                        };
                    }
                }

                if let Some(key) = map.next_key::<String>().await? {
                    Err(de::Error::custom(format!(
                        "the end of the map specifying this Value by type, not a second key \"{}\"",
                        key
                    )))
                } else {
                    Ok(Value::Link(link))
                }
            } else {
                Err(de::Error::invalid_value(key, &"a Link"))
            }
        } else {
            Err(de::Error::invalid_value("empty map", &"a Link"))
        }
    }

    async fn visit_seq<A: SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut value = if let Some(len) = seq.size_hint() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };

        while let Some(element) = seq.next_element().await? {
            value.push(element)
        }

        Ok(Value::Tuple(value.into()))
    }
}
