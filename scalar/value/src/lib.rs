use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::{de, Decoder, Encoder, FromStream, IntoStream, MapAccess, SeqAccess, ToStream};
use number_general::{Number, NumberInstance, NumberType};
use safecast::CastFrom;

use generic::Tuple;

pub mod link;

pub use link::*;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ValueType {
    None,
    Link,
    Number(NumberType),
    String,
    Tuple,
    Value,
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
    pub fn class(&self) -> ValueType {
        use ValueType as VT;
        match self {
            Self::Link(_) => VT::Link,
            Self::None => VT::None,
            Self::Number(n) => VT::Number(n.class()),
            Self::String(_) => VT::String,
            Self::Tuple(_) => VT::Tuple,
        }
    }

    pub fn is_none(&self) -> bool {
        match self {
            Self::None => true,
            Self::Tuple(tuple) => tuple.is_empty(),
            _ => false,
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
                Number::Complex(_c) => unimplemented!(),
                n => n.to_stream(encoder),
            },
            Self::String(s) => s.to_stream(encoder),
            Self::Tuple(t) => t.as_slice().into_stream(encoder),
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
            let value: Value = map.next_value().await?;

            if let Some(k) = map.next_key::<String>().await? {
                return Err(de::Error::invalid_value(k, &Self));
            }

            if let Ok(link) = Link::from_str(&key) {
                if value.is_none() {
                    // TODO: support Number::Complex
                    Ok(Value::Link(link))
                } else {
                    Err(de::Error::invalid_value(value, &"empty list"))
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
