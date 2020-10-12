use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::str::FromStr;

use bytes::Bytes;
use serde::de;
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

use crate::class::{Class, Instance, TCResult, TCType};
use crate::error;

use super::{Scalar, ScalarClass, ScalarInstance, TryCastFrom, TryCastInto};

pub mod class;
pub mod link;
pub mod number;
pub mod reference;
pub mod string;
pub mod version;

pub use class::*;
pub use link::*;
pub use number::*;
pub use reference::*;
pub use string::*;
pub use version::*;

pub const fn label(id: &'static str) -> string::Label {
    string::label(id)
}

#[derive(Clone, Eq, PartialEq)]
pub enum Value {
    None,
    Bytes(Bytes),
    Class(TCType),
    Number(Number),
    TCString(TCString),
    Tuple(Vec<Value>),
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
    type Class = class::ValueType;
}

impl class::ValueInstance for Value {
    type Class = class::ValueType;

    fn get(&self, path: TCPath, key: Value) -> TCResult<Value> {
        match self {
            Value::None => Err(error::not_found(path)),
            Value::Number(number) => number.get(path, key),
            Value::TCString(string) => string.get(path, key),
            other => Err(error::method_not_allowed(format!("GET {}", other.class()))),
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

impl<T1: Into<Value>, T2: Into<Value>> From<(T1, T2)> for Value {
    fn from(tuple: (T1, T2)) -> Value {
        Value::Tuple(vec![tuple.0.into(), tuple.1.into()])
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

impl<'a> TryFrom<&'a Value> for &'a Number {
    type Error = error::TCError;

    fn try_from(v: &'a Value) -> TCResult<&'a Number> {
        match v {
            Value::Number(n) => Ok(n),
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

impl TryFrom<Value> for TCPath {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<TCPath> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected Path but found", other)),
        }
    }
}

impl TryFrom<Value> for TCRef {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<TCRef> {
        match v {
            Value::TCString(s) => s.try_into(),
            other => Err(error::bad_request("Expected Ref but found", other)),
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

impl TryCastFrom<Vec<Scalar>> for Value {
    fn can_cast_from(tuple: &Vec<Scalar>) -> bool {
        for s in tuple {
            match s {
                Scalar::Value(_) => {}
                _ => return false,
            }
        }

        true
    }

    fn opt_cast_from(mut tuple: Vec<Scalar>) -> Option<Value> {
        let mut values = Vec::with_capacity(tuple.len());
        for s in tuple.drain(..) {
            match s {
                Scalar::Value(value) => values.push(value),
                _ => return None,
            }
        }

        Some(Value::Tuple(values))
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

impl TryCastFrom<Value> for TCPath {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::TCString(TCString::Link(link)) = value {
            if link.host().is_none() {
                return true;
            }
        }

        false
    }

    fn opt_cast_from(value: Value) -> Option<TCPath> {
        if let Value::TCString(TCString::Link(link)) = value {
            if link.host().is_none() {
                return Some(link.into_path());
            }
        }

        None
    }
}

impl TryCastFrom<Value> for TCRef {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::TCString(s) => TCRef::can_cast_from(s),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<TCRef> {
        match value {
            Value::TCString(s) => TCRef::opt_cast_from(s),
            _ => None,
        }
    }
}

impl TryCastFrom<Value> for TCString {
    fn can_cast_from(_value: &Value) -> bool {
        true
    }

    fn opt_cast_from(value: Value) -> Option<TCString> {
        match value {
            Value::TCString(s) => Some(s),
            other => Some(TCString::UString(other.to_string())),
        }
    }
}

impl TryCastFrom<Value> for ValueId {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::TCString(s) = value {
            ValueId::can_cast_from(s)
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<ValueId> {
        if let Value::TCString(s) = value {
            ValueId::opt_cast_from(s)
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
            values.iter().all(T::can_cast_from)
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<Vec<T>> {
        if let Value::Tuple(mut values) = value {
            let mut cast: Vec<T> = Vec::with_capacity(values.len());
            for val in values.drain(..) {
                if let Some(val) = val.opt_cast_into() {
                    cast.push(val)
                } else {
                    return None;
                }
            }

            Some(cast)
        } else {
            None
        }
    }
}

impl<T1: TryCastFrom<Value>, T2: TryCastFrom<Value>> TryCastFrom<Value> for (T1, T2) {
    fn can_cast_from(source: &Value) -> bool {
        if let Value::Tuple(source) = source {
            if source.len() == 2 && T1::can_cast_from(&source[0]) && T2::can_cast_from(&source[1]) {
                return true;
            }
        }

        false
    }

    fn opt_cast_from(source: Value) -> Option<(T1, T2)> {
        if let Value::Tuple(mut source) = source {
            if source.len() == 2 {
                let second: Option<T2> = source.pop().unwrap().opt_cast_into();
                let first: Option<T1> = source.pop().unwrap().opt_cast_into();
                return match (first, second) {
                    (Some(first), Some(second)) => Some((first, second)),
                    _ => None,
                };
            }
        }

        None
    }
}

impl<T1: TryCastFrom<Value>, T2: TryCastFrom<Value>, T3: TryCastFrom<Value>> TryCastFrom<Value>
    for (T1, T2, T3)
{
    fn can_cast_from(source: &Value) -> bool {
        if let Value::Tuple(source) = source {
            if source.len() == 3
                && T1::can_cast_from(&source[0])
                && T2::can_cast_from(&source[1])
                && T3::can_cast_from(&source[2])
            {
                return true;
            }
        }

        false
    }

    fn opt_cast_from(source: Value) -> Option<(T1, T2, T3)> {
        if let Value::Tuple(mut source) = source {
            if source.len() == 3 {
                let third: Option<T3> = source.pop().unwrap().opt_cast_into();
                let second: Option<T2> = source.pop().unwrap().opt_cast_into();
                let first: Option<T1> = source.pop().unwrap().opt_cast_into();
                return match (first, second, third) {
                    (Some(first), Some(second), Some(third)) => Some((first, second, third)),
                    _ => None,
                };
            }
        }

        None
    }
}

pub struct ValueVisitor;

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
            let value: Value = access.next_value()?;

            if key.starts_with('$') {
                let subject = TCRef::from_str(key).map_err(de::Error::custom)?;

                if value == Value::Tuple(vec![]) || value == Value::None {
                    Ok(Value::TCString(TCString::Ref(subject)))
                } else {
                    Err(de::Error::custom(format!(
                        "Expected Value but found MethodRef: {}",
                        value
                    )))
                }
            } else if let Ok(link) = key.parse::<link::Link>() {
                if link.host().is_none() && link.path().starts_with(&ValueType::prefix()) {
                    let dtype = ValueType::from_path(link.path()).map_err(de::Error::custom)?;
                    dtype.try_cast(value).map_err(de::Error::custom)
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

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::None)
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
            Value::Bytes(b) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(Link::from(ValueType::Bytes).path(), &[base64::encode(b)])?;
                map.end()
            }
            Value::Class(c) => {
                let c: link::Link = c.clone().into();
                c.serialize(s)
            }
            Value::Number(n) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(Link::from(n.class()).path(), &[n])?;
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
