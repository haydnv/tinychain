use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use bytes::Bytes;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

use crate::class::{Instance, NativeClass, State, TCType};
use crate::error;
use crate::general::{TCResult, TryCastFrom, TryCastInto, Tuple};
use crate::handler::{Handler, Route};
use crate::transaction::Txn;

use super::{Scalar, ScalarClass, ScalarInstance};

pub mod class;
pub mod link;
pub mod number;
pub mod string;
pub mod version;

use crate::scalar::MethodType;
pub use class::*;
pub use link::*;
pub use number::*;
use std::ops::Deref;
pub use string::*;
pub use version::*;

#[derive(Clone, Eq, PartialEq)]
pub enum Value {
    None,
    Bytes(Bytes),
    Class(TCType),
    Number(Number),
    TCString(TCString),
    Tuple(Tuple<Value>),
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
    type Class = class::ValueType;

    fn class(&self) -> class::ValueType {
        match self {
            Value::None => ValueType::None,
            Value::Bytes(_) => ValueType::Bytes,
            Value::Class(_) => ValueType::Class,
            Value::Number(n) => ValueType::Number(n.class()),
            Value::TCString(s) => ValueType::TCString(s.class()),
            Value::Tuple(_) => ValueType::Tuple,
        }
    }
}

impl ScalarInstance for Value {
    type Class = ValueType;
}

impl ValueInstance for Value {
    type Class = ValueType;
}

impl Route for Value {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        if method != MethodType::Get || path.is_empty() {
            None
        } else if &path[0] == "eq" {
            return Some(Box::new(EqHandler { value: self }));
        } else {
            match self {
                Self::Number(n) => n.route(method, path),
                Self::Tuple(tuple) => tuple.route(method, path),
                _ => None,
            }
        }
    }
}

impl Route for Tuple<Value> {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        if method != MethodType::Get || path.is_empty() {
            None
        } else if usize::can_cast_from(&path[0]) {
            let i = usize::opt_cast_from(path[0].clone()).unwrap();
            if let Some(value) = self.deref().get(i) {
                value.route(method, path)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl Default for Value {
    fn default() -> Value {
        Value::None
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

impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Number(b.into())
    }
}

impl From<Bytes> for Value {
    fn from(b: Bytes) -> Value {
        Value::Bytes(b)
    }
}

impl From<Link> for Value {
    fn from(l: Link) -> Value {
        Value::TCString(l.into())
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

impl From<Id> for Value {
    fn from(v: Id) -> Value {
        let s: TCString = v.into();
        s.into()
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

impl<T: Into<Value>> FromIterator<T> for Value {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Value {
        Value::Tuple(Tuple::from_iter(iter.into_iter().map(T::into)))
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Value {
        Self::from_iter(v)
    }
}

impl<T1: Into<Value>, T2: Into<Value>> From<(T1, T2)> for Value {
    fn from(tuple: (T1, T2)) -> Value {
        Value::Tuple(vec![tuple.0.into(), tuple.1.into()].into())
    }
}

impl TryCastFrom<State> for Value {
    fn can_cast_from(state: &State) -> bool {
        if let State::Scalar(scalar) = state {
            Value::can_cast_from(scalar)
        } else {
            false
        }
    }

    fn opt_cast_from(state: State) -> Option<Value> {
        if let State::Scalar(scalar) = state {
            Value::opt_cast_from(scalar)
        } else {
            None
        }
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

impl TryFrom<Value> for Bytes {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Bytes> {
        match v {
            Value::Bytes(b) => Ok(b),
            other => Err(error::bad_request("Expected Bytes but found", other)),
        }
    }
}

impl TryFrom<Value> for Link {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Link> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected Link but found", other)),
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

impl<'a> TryFrom<&'a Value> for Number {
    type Error = error::TCError;

    fn try_from(v: &'a Value) -> TCResult<Number> {
        match v {
            Value::Number(n) => Ok(*n),
            other => Err(error::bad_request("Expected Number but found", other)),
        }
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

impl TryFrom<Value> for String {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<String> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected String but found", other)),
        }
    }
}

impl TryFrom<Value> for TCPathBuf {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<TCPathBuf> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected Path but found", other)),
        }
    }
}

impl TryFrom<Value> for TCString {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<TCString> {
        match v {
            Value::TCString(s) => Ok(s),
            other => Err(error::bad_request("Expected String but found", other)),
        }
    }
}

impl TryFrom<Value> for Id {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Id> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected Id, found", other)),
        }
    }
}

impl<'a> TryFrom<&'a Value> for &'a Id {
    type Error = error::TCError;

    fn try_from(v: &'a Value) -> TCResult<&'a Id> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected Id but found", other)),
        }
    }
}

impl TryFrom<Value> for Tuple<Value> {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Tuple<Value>> {
        match v {
            Value::Tuple(t) => Ok(t.into()),
            other => Err(error::bad_request("Expected Tuple, found", other)),
        }
    }
}

impl TryCastFrom<Tuple<Scalar>> for Value {
    fn can_cast_from(tuple: &Tuple<Scalar>) -> bool {
        Vec::<Value>::can_cast_from(tuple)
    }

    fn opt_cast_from(tuple: Tuple<Scalar>) -> Option<Value> {
        Vec::<Value>::opt_cast_from(tuple)
            .map(Tuple::from)
            .map(Value::Tuple)
    }
}

impl TryCastFrom<Scalar> for Tuple<Value> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => tuple.iter().all(Value::can_cast_from),
            Scalar::Value(Value::Tuple(_)) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Tuple<Value>> {
        match scalar {
            Scalar::Tuple(tuple) => Vec::<Value>::opt_cast_from(tuple).map(Tuple::from),
            Scalar::Value(Value::Tuple(tuple)) => Some(tuple),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for Link {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::TCString(s) = value {
            Link::can_cast_from(s)
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<Link> {
        if let Value::TCString(s) = value {
            Link::opt_cast_from(s)
        } else {
            None
        }
    }
}

impl TryCastFrom<Value> for Number {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::Number(_) = value {
            true
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<Number> {
        if let Value::Number(n) = value {
            Some(n)
        } else {
            None
        }
    }
}

impl TryCastFrom<Value> for i64 {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::Number(n) = value {
            i64::can_cast_from(n)
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<i64> {
        if let Value::Number(n) = value {
            i64::opt_cast_from(n)
        } else {
            None
        }
    }
}

impl TryCastFrom<Value> for u64 {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::Number(n) = value {
            u64::can_cast_from(n)
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<u64> {
        if let Value::Number(n) = value {
            u64::opt_cast_from(n)
        } else {
            None
        }
    }
}

impl TryCastFrom<Value> for usize {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::Number(n) = value {
            usize::can_cast_from(n)
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<usize> {
        if let Value::Number(n) = value {
            usize::opt_cast_from(n)
        } else {
            None
        }
    }
}

impl TryCastFrom<Value> for number::NumberType {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Class(class) => number::NumberType::can_cast_from(class),
            Value::TCString(TCString::Link(link)) => number::NumberType::can_cast_from(link),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<number::NumberType> {
        match value {
            Value::Class(class) => class.opt_cast_into(),
            Value::TCString(TCString::Link(link)) => link.opt_cast_into(),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for TCString {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::TCString(_) = value {
            true
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<TCString> {
        match value {
            Value::TCString(s) => Some(s),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for Id {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::TCString(s) = value {
            Id::can_cast_from(s)
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<Id> {
        if let Value::TCString(s) = value {
            Id::opt_cast_from(s)
        } else {
            None
        }
    }
}

impl TryCastFrom<Value> for ValueType {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Class(class) => ValueType::can_cast_from(class),
            Value::TCString(TCString::Link(link)) => ValueType::can_cast_from(link),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<ValueType> {
        match value {
            Value::Class(class) => class.opt_cast_into(),
            Value::TCString(TCString::Link(link)) => link.opt_cast_into(),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<Value>> TryCastFrom<Value> for Vec<T> {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::Tuple(values) = value {
            Self::can_cast_from(values)
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<Vec<T>> {
        if let Value::Tuple(values) = value {
            Self::opt_cast_from(values)
        } else {
            None
        }
    }
}

impl<T1: TryCastFrom<Value>, T2: TryCastFrom<Value>> TryCastFrom<Value> for (T1, T2) {
    fn can_cast_from(source: &Value) -> bool {
        if let Value::Tuple(source) = source {
            Self::can_cast_from(source)
        } else {
            false
        }
    }

    fn opt_cast_from(source: Value) -> Option<Self> {
        if let Value::Tuple(source) = source {
            Self::opt_cast_from(source)
        } else {
            None
        }
    }
}

impl<T1: TryCastFrom<Value>, T2: TryCastFrom<Value>, T3: TryCastFrom<Value>> TryCastFrom<Value>
    for (T1, T2, T3)
{
    fn can_cast_from(source: &Value) -> bool {
        if let Value::Tuple(source) = source {
            Self::can_cast_from(source)
        } else {
            false
        }
    }

    fn opt_cast_from(source: Value) -> Option<Self> {
        if let Value::Tuple(source) = source {
            Self::opt_cast_from(source)
        } else {
            None
        }
    }
}

pub struct ValueVisitor;

impl ValueVisitor {
    fn visit_float<F: Into<number::instance::Float>, E: de::Error>(
        &self,
        f: F,
    ) -> Result<Value, E> {
        self.visit_number(f.into())
    }

    fn visit_int<I: Into<number::instance::Int>, E: de::Error>(&self, i: I) -> Result<Value, E> {
        self.visit_number(i.into())
    }

    fn visit_uint<U: Into<number::instance::UInt>, E: de::Error>(&self, u: U) -> Result<Value, E> {
        self.visit_number(u.into())
    }

    fn visit_number<N: Into<Number>, E: de::Error>(&self, n: N) -> Result<Value, E> {
        Ok(Value::Number(n.into()))
    }
}

impl<'de> de::Visitor<'de> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Value, e.g. \"foo\" or 123 or {\"$object_ref: [\"slice_id\", \"$state\"]\"}")
    }

    fn visit_i16<E: de::Error>(self, value: i16) -> Result<Self::Value, E> {
        self.visit_int(value)
    }

    fn visit_i32<E: de::Error>(self, value: i32) -> Result<Self::Value, E> {
        self.visit_int(value)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        self.visit_int(value)
    }

    fn visit_u8<E: de::Error>(self, value: u8) -> Result<Self::Value, E> {
        self.visit_uint(value)
    }

    fn visit_u16<E: de::Error>(self, value: u16) -> Result<Self::Value, E> {
        self.visit_uint(value)
    }

    fn visit_u32<E: de::Error>(self, value: u32) -> Result<Self::Value, E> {
        self.visit_uint(value)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        self.visit_uint(value)
    }

    fn visit_f32<E: de::Error>(self, value: f32) -> Result<Self::Value, E> {
        self.visit_float(value)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        self.visit_float(value)
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<Self::Value, E> {
        Ok(Value::TCString(TCString::UString(s.to_string())))
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(Value::None)
    }

    fn visit_seq<L: de::SeqAccess<'de>>(self, mut access: L) -> Result<Self::Value, L::Error> {
        let mut items: Vec<Value> = if let Some(size) = access.size_hint() {
            Vec::with_capacity(size)
        } else {
            vec![]
        };

        while let Some(value) = access.next_element()? {
            items.push(value)
        }

        Ok(Value::Tuple(items.into()))
    }

    fn visit_map<M: de::MapAccess<'de>>(self, mut access: M) -> Result<Self::Value, M::Error> {
        if let Some(key) = access.next_key::<&str>()? {
            let value: Value = access.next_value()?;

            if key.starts_with('$') {
                Err(de::Error::custom(format!(
                    "Expected Value but found Ref: {}",
                    key
                )))
            } else if let Ok(link) = key.parse::<link::Link>() {
                let path = link.path();

                if value == Value::None || value == Value::Tuple(Tuple::default()) {
                    Ok(Value::TCString(link.into()))
                } else if link.host().is_none() && path[..].starts_with(&ValueType::prefix()[..]) {
                    if let Value::Tuple(mut tuple) = value {
                        if path.len() == 3 && &path[2] == "tuple" {
                            Ok(Value::Tuple(tuple))
                        } else {
                            let dtype = ValueType::from_path(&link.path()[..])
                                .map_err(de::Error::custom)?;

                            if tuple.len() == 1 {
                                let key = tuple.pop().unwrap();
                                dtype.try_cast(key).map_err(de::Error::custom)
                            } else {
                                dtype.try_cast(tuple).map_err(de::Error::custom)
                            }
                        }
                    } else {
                        let dtype =
                            ValueType::from_path(&link.path()[..]).map_err(de::Error::custom)?;

                        dtype.try_cast(value).map_err(de::Error::custom)
                    }
                } else {
                    Err(de::Error::custom(format!("Support for {}", link)))
                }
            } else {
                Err(de::Error::custom("Not implemented"))
            }
        } else {
            Err(de::Error::custom(
                "Empty map is not a valid Tinychain datatype",
            ))
        }
    }
}

impl<'de> de::Deserialize<'de> for Value {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(ValueVisitor)
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::None => s.serialize_none(),
            Value::Bytes(b) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(
                    &Link::from(ValueType::Bytes).into_path(),
                    &base64::encode(b),
                )?;
                map.end()
            }
            Value::Class(c) => {
                let c: link::Link = c.clone().into();
                c.serialize(s)
            }
            Value::Number(n) => n.serialize(s),
            Value::TCString(tc_string) => tc_string.serialize(s),
            Value::Tuple(v) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for item in v.iter() {
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
            Value::Class(c) => write!(f, "Class: {}", c),
            Value::Number(n) => write!(f, "Number({})", n),
            Value::TCString(s) => write!(f, "String({})", s),
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

pub struct EqHandler<'a> {
    value: &'a Value,
}

#[async_trait]
impl<'a> Handler for EqHandler<'a> {
    fn subject(&self) -> TCType {
        self.value.class().into()
    }

    async fn handle_get(self: Box<Self>, _txn: &Txn, other: Value) -> TCResult<State> {
        Ok(State::from(Value::Number(Number::Bool(Boolean::from(
            self.value == &other,
        )))))
    }
}
