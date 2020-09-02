use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Bound;

use bytes::Bytes;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

use crate::class::{Class, Instance, TCResult, TCType};
use crate::error;

pub mod class;
pub mod link;
pub mod number;
pub mod op;
pub mod reference;
pub mod string;
pub mod version;

pub type Label = string::Label;
pub type Link = link::Link;
pub type Number = number::instance::Number;
pub type Op = op::Op;
pub type TCPath = link::TCPath;
pub type TCString = string::TCString;
pub type TCRef = reference::TCRef;
pub type ValueId = string::ValueId;
pub type ValueType = class::ValueType;

pub const fn label(id: &'static str) -> string::Label {
    string::label(id)
}

#[derive(Clone, PartialEq)]
pub enum Value {
    None,
    Bound(Bound<Box<Value>>),
    Bytes(Bytes),
    Class(TCType),
    Number(Number),
    TCString(TCString),
    Op(Box<op::Op>),
    Tuple(Vec<Value>),
}

impl Instance for Value {
    type Class = class::ValueType;

    fn class(&self) -> class::ValueType {
        use class::ValueType;
        match self {
            Value::None => ValueType::None,
            Value::Bound(_) => ValueType::Bound,
            Value::Bytes(_) => ValueType::Bytes,
            Value::Class(_) => ValueType::Class,
            Value::Number(n) => ValueType::Number(n.class()),
            Value::TCString(s) => ValueType::TCString(s.class()),
            Value::Op(_) => ValueType::Op,
            Value::Tuple(_) => ValueType::Tuple,
        }
    }
}

impl class::ValueInstance for Value {
    type Class = class::ValueType;
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

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Number(b.into())
    }
}

impl From<Bound<Value>> for Value {
    fn from(b: Bound<Value>) -> Value {
        match b {
            Bound::Included(v) => Value::Bound(Bound::Included(Box::new(v))),
            Bound::Excluded(v) => Value::Bound(Bound::Excluded(Box::new(v))),
            Bound::Unbounded => Value::Bound(Bound::Unbounded),
        }
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

impl From<u64> for Value {
    fn from(u: u64) -> Value {
        let u: number::instance::UInt = u.into();
        let n: Number = u.into();
        n.into()
    }
}

impl From<TCString> for Value {
    fn from(s: TCString) -> Value {
        Value::TCString(s)
    }
}

impl From<TCRef> for Value {
    fn from(r: TCRef) -> Value {
        let s: TCString = r.into();
        s.into()
    }
}

impl From<ValueId> for Value {
    fn from(v: ValueId) -> Value {
        let s: TCString = v.into();
        s.into()
    }
}

impl From<op::Op> for Value {
    fn from(op: op::Op) -> Value {
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
        Value::Tuple(v.drain(..).map(|i| i.into()).collect())
    }
}

impl TryFrom<Value> for bool {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<bool> {
        match v {
            Value::Number(n) => n.try_into(),
            other => Err(error::bad_request("Expected bool but found", other)),
        }
    }
}

impl TryFrom<Value> for Bound<Value> {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Bound<Value>> {
        match v {
            Value::Bound(b) => match b {
                Bound::Included(b) => Ok(Bound::Included(*b)),
                Bound::Excluded(b) => Ok(Bound::Excluded(*b)),
                Bound::Unbounded => Ok(Bound::Unbounded),
            },
            other => Err(error::bad_request("Expected Bound but found", other)),
        }
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

impl<'a> TryFrom<&'a Value> for &'a Number {
    type Error = error::TCError;

    fn try_from(v: &'a Value) -> TCResult<&'a Number> {
        match v {
            Value::Number(n) => Ok(n),
            other => Err(error::bad_request("Expected Number but found", other)),
        }
    }
}

impl TryFrom<Value> for number::NumberType {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<number::NumberType> {
        match v {
            Value::Class(t) => t.try_into(),
            Value::TCString(TCString::Link(l)) if l.host().is_none() => {
                number::NumberType::from_path(l.path())
            }
            other => Err(error::bad_request("Expected NumberType, found", other)),
        }
    }
}

impl TryFrom<Value> for usize {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<usize> {
        let n: Number = v.try_into()?;
        n.try_into()
    }
}

impl TryFrom<Value> for u64 {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<u64> {
        let n: Number = v.try_into()?;
        n.try_into()
    }
}

impl<'a> TryFrom<&'a Value> for &'a String {
    type Error = error::TCError;

    fn try_from(v: &'a Value) -> TCResult<&'a String> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected String but found", other)),
        }
    }
}

impl TryFrom<Value> for TCType {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<TCType> {
        match v {
            Value::Class(c) => Ok(c),
            other => Err(error::bad_request("Expected Class, found", other)),
        }
    }
}

impl TryFrom<Value> for ValueType {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<ValueType> {
        match v {
            Value::Class(t) => t.try_into(),
            Value::TCString(TCString::Link(l)) if l.host().is_none() => {
                ValueType::from_path(l.path())
            }
            other => Err(error::bad_request("Expected ValueType, found", other)),
        }
    }
}

impl TryFrom<Value> for ValueId {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<ValueId> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected ValueId, found", other)),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a ValueId {
    type Error = error::TCError;

    fn try_from(v: &'a Value) -> TCResult<&'a ValueId> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected ValueId but found", other)),
        }
    }
}

impl TryFrom<Value> for Vec<Value> {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Vec<Value>> {
        match v {
            Value::Tuple(t) => Ok(t),
            other => Err(error::bad_request("Expected Tuple, found", other)),
        }
    }
}

impl<T: TryFrom<Value, Error = error::TCError>> TryFrom<Value> for Vec<T> {
    type Error = error::TCError;

    fn try_from(source: Value) -> TCResult<Vec<T>> {
        let mut source: Vec<Value> = source.try_into()?;
        let mut values = Vec::with_capacity(source.len());
        for value in source.drain(..) {
            values.push(value.try_into()?);
        }
        Ok(values)
    }
}

impl<
        E1: Into<error::TCError>,
        T1: TryFrom<Value, Error = E1>,
        E2: Into<error::TCError>,
        T2: TryFrom<Value, Error = E2>,
    > TryFrom<Value> for (T1, T2)
{
    type Error = error::TCError;

    fn try_from(source: Value) -> TCResult<(T1, T2)> {
        match source {
            Value::Tuple(mut source) if source.len() == 2 => {
                let second: T2 = source.pop().unwrap().try_into().map_err(|e: E2| e.into())?;
                let first: T1 = source.pop().unwrap().try_into().map_err(|e: E1| e.into())?;
                Ok((first, second))
            }
            other => Err(error::bad_request("Expected a 2-Tuple but found", other)),
        }
    }
}

impl<
        E1: Into<error::TCError>,
        T1: TryFrom<Value, Error = E1>,
        E2: Into<error::TCError>,
        T2: TryFrom<Value, Error = E2>,
        E3: Into<error::TCError>,
        T3: TryFrom<Value, Error = E3>,
    > TryFrom<Value> for (T1, T2, T3)
{
    type Error = error::TCError;

    fn try_from(source: Value) -> TCResult<(T1, T2, T3)> {
        match source {
            Value::Tuple(mut source) if source.len() == 3 => {
                let third: T3 = source.pop().unwrap().try_into().map_err(|e: E3| e.into())?;
                let second: T2 = source.pop().unwrap().try_into().map_err(|e: E2| e.into())?;
                let first: T1 = source.pop().unwrap().try_into().map_err(|e: E1| e.into())?;
                Ok((first, second, third))
            }
            other => Err(error::bad_request("Expected a 3-Tuple but found", other)),
        }
    }
}

struct ValueVisitor;

impl ValueVisitor {
    fn visit_float<F: Into<number::instance::Float>>(&self, f: F) -> TCResult<Value> {
        self.visit_number(f.into())
    }

    fn visit_int<I: Into<number::instance::Int>>(&self, i: I) -> TCResult<Value> {
        self.visit_number(i.into())
    }

    fn visit_uint<U: Into<number::instance::UInt>>(&self, u: U) -> TCResult<Value> {
        self.visit_number(u.into())
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

    fn visit_i16<E>(self, value: i16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_int(value).map_err(de::Error::custom)
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

    fn visit_u8<E>(self, value: u8) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_uint(value).map_err(de::Error::custom)
    }

    fn visit_u16<E>(self, value: u16) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_uint(value).map_err(de::Error::custom)
    }

    fn visit_u32<E>(self, value: u32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_uint(value).map_err(de::Error::custom)
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        self.visit_uint(value).map_err(de::Error::custom)
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
                    1 => Ok(op::Op::Get(subject.into(), value.remove(0)).into()),
                    2 => Ok(op::Op::Put(subject.into(), value.remove(0), value.remove(0)).into()),
                    _ => Err(de::Error::custom(format!(
                        "Expected a Get or Put op, found {}",
                        Value::Tuple(value)
                    ))),
                }
            } else if let Ok(link) = key.parse::<link::Link>() {
                if value.is_empty() {
                    Ok(Value::TCString(TCString::Link(link)))
                } else if value.len() == 1 {
                    Ok(Value::Op(Box::new(Op::Get(
                        link.into(),
                        value.pop().unwrap(),
                    ))))
                } else if value.len() == 2 {
                    Ok(Value::Op(Box::new(Op::Put(
                        link.into(),
                        value.pop().unwrap(),
                        value.pop().unwrap(),
                    ))))
                } else {
                    Err(de::Error::custom(
                        "This functionality is not yet implemented",
                    ))
                }
            } else if let Ok(value_id) = key.parse::<string::ValueId>() {
                if value.is_empty() {
                    Ok(Value::TCString(TCString::Id(value_id)))
                } else {
                    Err(de::Error::custom(
                        "This functionality is not yet implemented",
                    ))
                }
            } else {
                Err(de::Error::custom(
                    "This functionality is not yet implemented",
                ))
            }
        } else {
            Err(de::Error::custom("Unable to parse map entry"))
        }
    }

    fn visit_seq<L>(self, mut access: L) -> Result<Self::Value, L::Error>
    where
        L: de::SeqAccess<'de>,
    {
        let mut items: Vec<Value> = if let Some(size) = access.size_hint() {
            Vec::with_capacity(size)
        } else {
            vec![]
        };

        while let Some(value) = access.next_element()? {
            items.push(value)
        }

        Ok(Value::Tuple(items))
    }

    fn visit_str<E>(self, s: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::TCString(TCString::UString(s.to_string())))
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
            Value::Bound(b) => b.serialize(s),
            Value::Bytes(b) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry("/sbin/value/bytes", &[base64::encode(b)])?;
                map.end()
            }
            Value::Class(c) => {
                let c: link::Link = c.clone().into();
                c.serialize(s)
            }
            Value::Number(n) => n.serialize(s),
            Value::Op(op) => {
                let mut map = s.serialize_map(Some(1))?;
                match &**op {
                    op::Op::If(cond, then, or_else) => map.serialize_entry(
                        "/sbin/value/op/if",
                        &[&Value::from(cond.clone()), then, or_else],
                    )?,
                    op::Op::Get(subject, selector) => {
                        map.serialize_entry(&subject.to_string(), &[selector])?
                    }
                    op::Op::Put(subject, selector, value) => {
                        map.serialize_entry(&subject.to_string(), &[selector, value])?
                    }
                }
                map.end()
            }
            Value::TCString(tc_string) => tc_string.serialize(s),
            Value::Tuple(v) => {
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
            Value::Bound(b) => write!(f, "Bound({:?})", b),
            Value::Class(c) => write!(f, "Class: {}", c),
            Value::Number(n) => write!(f, "Number({})", n),
            Value::TCString(s) => write!(f, "String({})", s),
            Value::Op(op) => write!(f, "Op: {}", op),
            Value::Tuple(v) => write!(
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
