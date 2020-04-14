use std::convert::{TryFrom, TryInto};
use std::fmt;

use regex::Regex;
use serde::de;
use serde::ser::{Serialize, SerializeSeq, Serializer};

use crate::context::TCResult;
use crate::error;
use crate::state::TCState;

mod link;
mod op;
mod reference;

pub type Link = link::Link;
pub type Op = op::Op;
pub type TCRef = reference::TCRef;
pub type Subject = op::Subject;
pub type ValueId = String;

pub trait TCValueTryInto: TryInto<TCValue, Error = error::TCError> {}
pub trait TCValueTryFrom: TryFrom<TCValue, Error = error::TCError> {}

const RESERVED_CHARS: [&str; 17] = [
    "..", "~", "$", "&", "?", "|", "{", "}", "//", ":", "=", "^", ">", "<", "'", "`", "\"",
];

fn validate_id(id: &str) -> TCResult<()> {
    let mut eot_char = [0];
    let eot_char = (4 as char).encode_utf8(&mut eot_char);

    let reserved = [&RESERVED_CHARS[..], &[eot_char]].concat();

    for pattern in reserved.iter() {
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

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum TCValue {
    None,
    Int32(i32),
    Link(Link),
    Op(Op),
    r#String(String),
    Vector(Vec<TCValue>),
    Ref(TCRef),
}

impl TCValueTryFrom for String {}

impl From<()> for TCValue {
    fn from(_: ()) -> TCValue {
        TCValue::None
    }
}

impl From<Link> for TCValue {
    fn from(link: Link) -> TCValue {
        TCValue::Link(link)
    }
}

impl From<Op> for TCValue {
    fn from(op: Op) -> TCValue {
        TCValue::Op(op)
    }
}

impl From<Option<TCValue>> for TCValue {
    fn from(opt: Option<TCValue>) -> TCValue {
        match opt {
            Some(val) => val,
            None => TCValue::None,
        }
    }
}

impl From<TCRef> for TCValue {
    fn from(r: TCRef) -> TCValue {
        TCValue::Ref(r)
    }
}

impl From<String> for TCValue {
    fn from(s: String) -> TCValue {
        TCValue::r#String(s)
    }
}

impl From<Vec<TCValue>> for TCValue {
    fn from(v: Vec<TCValue>) -> TCValue {
        TCValue::Vector(v)
    }
}

impl From<Vec<(ValueId, TCValue)>> for TCValue {
    fn from(values: Vec<(ValueId, TCValue)>) -> TCValue {
        let mut result: Vec<TCValue> = vec![];
        for (id, val) in values {
            let mut r_item: Vec<TCValue> = vec![];
            r_item.push(id.into());
            r_item.push(val);
            result.push(r_item.into());
        }
        result.into()
    }
}

impl TryFrom<TCValue> for Link {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<Link> {
        match v {
            TCValue::Link(l) => Ok(l),
            other => Err(error::bad_request("Expected Link but found", other)),
        }
    }
}

impl TryFrom<TCValue> for String {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<String> {
        match v {
            TCValue::r#String(s) => Ok(s),
            other => Err(error::bad_request("Expected a String but found", other)),
        }
    }
}

impl TryFrom<TCValue> for Vec<TCValue> {
    type Error = error::TCError;

    fn try_from(v: TCValue) -> TCResult<Vec<TCValue>> {
        match v {
            TCValue::Vector(v) => Ok(v.to_vec()),
            other => Err(error::bad_request("Expected Vector but found", other)),
        }
    }
}

impl<T: Into<TCValue>> std::iter::FromIterator<T> for TCValue {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v: Vec<TCValue> = vec![];
        for item in iter {
            v.push(item.into());
        }

        v.into()
    }
}

impl<T1: TCValueTryFrom, T2: TCValueTryFrom> TryFrom<TCValue> for Vec<(T1, T2)> {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Vec<(T1, T2)>> {
        let value: Vec<TCValue> = value.try_into()?;
        let mut v: Vec<(T1, T2)> = Vec::with_capacity(value.len());
        for item in value {
            let item: (T1, T2) = item.try_into()?;
            v.push(item);
        }
        Ok(v)
    }
}

impl<T1: TCValueTryFrom, T2: TCValueTryFrom> TryFrom<TCValue> for (T1, T2) {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<(T1, T2)> {
        let v: Vec<TCValue> = value.clone().try_into()?;
        if v.len() == 2 {
            Ok((v[0].clone().try_into()?, v[1].clone().try_into()?))
        } else {
            Err(error::bad_request("Expected 2-tuple but found", value))
        }
    }
}

impl<T: TCValueTryFrom> TryFrom<TCValue> for (T, TCValue) {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<(T, TCValue)> {
        let value: Vec<TCValue> = value.try_into()?;
        if value.len() != 2 {
            return Err(error::bad_request(
                "Expected 2-tuple but found",
                format!("{:?}", value),
            ));
        }

        let item: T = value[0].clone().try_into()?;
        Ok((item, value[1].clone()))
    }
}

impl TryFrom<TCState> for TCValue {
    type Error = error::TCError;

    fn try_from(state: TCState) -> TCResult<TCValue> {
        match state {
            TCState::Value(value) => Ok(value),
            other => Err(error::bad_request("Expected a Value but found", other)),
        }
    }
}

struct TCValueVisitor;

impl<'de> de::Visitor<'de> for TCValueVisitor {
    type Value = TCValue;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("A Tinychain Value, e.g. \"foo\" or 123 or {\"$object_ref: [\"slice_id\", \"$state\"]\"}")
    }

    fn visit_i32<E>(self, value: i32) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(TCValue::Int32(value))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(TCValue::r#String(value.to_string()))
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        if let Some(key) = access.next_key::<String>()? {
            if key.starts_with('/') {
                let value = access.next_value::<Vec<(String, TCValue)>>()?;

                let link = Link::to(&key).map_err(de::Error::custom)?;

                if value.is_empty() {
                    Ok(link.into())
                } else {
                    Ok(Op::post(None, link, value).into())
                }
            } else if key.starts_with('$') {
                if key.contains('/') {
                    let key: Vec<&str> = key.split('/').collect();
                    let subject = TCRef::to(&key[0][1..]).map_err(de::Error::custom)?;
                    let method =
                        Link::to(&format!("/{}", key[1..].join("/"))).map_err(de::Error::custom)?;
                    let requires = access.next_value::<Vec<(String, TCValue)>>()?;

                    Ok(Op::post(subject.into(), method, requires).into())
                } else {
                    let subject = TCRef::to(&key[1..].to_string()).map_err(de::Error::custom)?;
                    let value = access.next_value::<Vec<TCValue>>()?;

                    if value.is_empty() {
                        Ok(subject.into())
                    } else if value.len() == 1 {
                        Ok(Op::get(subject.into(), value[0].clone()).into())
                    } else if value.len() == 2 {
                        Ok(Op::put(subject.into(), value[0].clone(), value[1].clone()).into())
                    } else {
                        Err(de::Error::custom(format!(
                            "Expected a list of 0, 1, or 2 Values for {}",
                            key
                        )))
                    }
                }
            } else {
                Err(de::Error::custom(format!(
                    "Expected a Link starting with '/' or a Ref starting with '$', found {}",
                    key
                )))
            }
        } else {
            Err(de::Error::custom("Unable to parse map entry"))
        }
    }

    fn visit_seq<S>(self, mut access: S) -> Result<Self::Value, S::Error>
    where
        S: de::SeqAccess<'de>,
    {
        let mut seq: Vec<TCValue> = vec![];

        while let Some(e) = access.next_element()? {
            seq.push(e);
        }

        Ok(TCValue::Vector(seq))
    }
}

impl<'de> de::Deserialize<'de> for TCValue {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        d.deserialize_any(TCValueVisitor)
    }
}

impl Serialize for TCValue {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            TCValue::None => s.serialize_none(),
            TCValue::Int32(i) => s.serialize_i32(*i),
            TCValue::Link(l) => l.serialize(s),
            TCValue::Op(o) => o.serialize(s),
            TCValue::Ref(r) => r.serialize(s),
            TCValue::r#String(v) => s.serialize_str(v),
            TCValue::Vector(v) => {
                let mut seq = s.serialize_seq(Some(v.len()))?;
                for element in v {
                    seq.serialize_element(element)?;
                }
                seq.end()
            }
        }
    }
}

impl fmt::Debug for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCValue::None => write!(f, "None"),
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: {}", l),
            TCValue::Op(o) => write!(f, "Op: {}", o),
            TCValue::Ref(r) => write!(f, "Ref: {}", r),
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector: {:?}", v),
        }
    }
}
