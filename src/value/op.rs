use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fmt;

use serde::de::{Deserializer, Error, MapAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde::Deserialize;

use crate::error;
use crate::value::{PathSegment, TCPath, TCRef, TCResult, TCValue, ValueId};

#[derive(Clone, Hash, Eq, PartialEq, Deserialize)]
pub enum Subject {
    Path(TCPath),
    Ref(TCRef),
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Subject::Path(p) => write!(f, "{}", p),
            Subject::Ref(r) => write!(f, "{}", r),
        }
    }
}

impl From<TCPath> for Subject {
    fn from(p: TCPath) -> Subject {
        Subject::Path(p)
    }
}

impl From<TCRef> for Subject {
    fn from(r: TCRef) -> Subject {
        Subject::Ref(r)
    }
}

impl Serialize for Subject {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&format!("{}", self))
    }
}

pub struct Args(HashMap<ValueId, TCValue>);

impl Args {
    pub fn take<T: TryFrom<TCValue, Error = error::TCError>>(&mut self, name: &str) -> TCResult<T> {
        if let Some(arg) = self.0.remove(&name.parse()?) {
            arg.try_into()
        } else {
            Err(error::bad_request("Required argument not provided", name))
        }
    }

    pub fn take_or<
        E1: Into<error::TCError>,
        E2: Into<error::TCError>,
        T1: TryFrom<TCValue, Error = E1>,
        T2: TryInto<T1, Error = E2>,
    >(
        &mut self,
        name: &str,
        else_val: T2,
    ) -> TCResult<T1> {
        if let Some(arg) = self.0.remove(&name.parse()?) {
            arg.try_into().map_err(|e: E1| e.into())
        } else {
            else_val.try_into().map_err(|e: E2| e.into())
        }
    }
}

impl From<HashMap<ValueId, TCValue>> for Args {
    fn from(v: HashMap<ValueId, TCValue>) -> Args {
        Args(v)
    }
}

impl From<Vec<(ValueId, TCValue)>> for Args {
    fn from(v: Vec<(ValueId, TCValue)>) -> Args {
        Args(v.into_iter().collect())
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum Op {
    Get {
        subject: Subject,
        key: Box<TCValue>,
    },
    Put {
        subject: Subject,
        key: Box<TCValue>,
        value: Box<TCValue>,
    },
    Post {
        subject: Option<TCRef>,
        action: TCPath,
        requires: Vec<(ValueId, TCValue)>,
    },
}

impl Op {
    pub fn get(subject: Subject, key: TCValue) -> Op {
        Op::Get {
            subject,
            key: Box::new(key),
        }
    }

    pub fn put(subject: Subject, key: TCValue, value: TCValue) -> Op {
        Op::Put {
            subject,
            key: Box::new(key),
            value: Box::new(value),
        }
    }

    pub fn post(subject: Option<TCRef>, action: TCPath, requires: Vec<(ValueId, TCValue)>) -> Op {
        Op::Post {
            subject,
            action,
            requires,
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Get { subject, key } => write!(f, "subject: {}, key: {}", subject, key),
            Op::Put {
                subject,
                key,
                value,
            } => write!(f, "subject: {}, key: {}, value: {}", subject, key, value),
            Op::Post {
                subject,
                action,
                requires,
            } => write!(
                f,
                "subject: {}, action: {}, requires: {}",
                subject
                    .as_ref()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| String::from("None")),
                action,
                requires
                    .iter()
                    .map(|(id, val)| format!("{}: {}", id, val))
                    .collect::<Vec<String>>()
                    .join(","),
            ),
        }
    }
}

pub struct OpVisitor;

impl<'de> Visitor<'de> for OpVisitor {
    type Value = Op;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("A Tinychain Op, e.g. {\"/sbin/table/new\": [(\"columns\", [(\"foo\", \"/sbin/value/string\"), ...]]}")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        if let Some(key) = access.next_key::<String>()? {
            if key.contains('/') {
                let key: Vec<&str> = key.split('/').collect();
                let subject: TCRef = key[0][1..].parse().map_err(Error::custom)?;
                let method: TCPath = key[1..]
                    .iter()
                    .map(|s| s.parse())
                    .collect::<TCResult<Vec<PathSegment>>>()
                    .map_err(Error::custom)?
                    .into();
                let requires = access.next_value::<Vec<(ValueId, TCValue)>>()?;

                Ok(Op::post(subject.into(), method, requires))
            } else {
                let subject: TCRef = key[1..].parse().map_err(Error::custom)?;
                let mut value = access.next_value::<Vec<TCValue>>()?;

                if value.len() == 1 {
                    Ok(Op::get(subject.into(), value.remove(0)))
                } else if value.len() == 2 {
                    Ok(Op::put(subject.into(), value.remove(0), value.remove(0)))
                } else {
                    Err(Error::custom(format!(
                        "Expected either 1 (for a Get), or 2 (for a Put) Values for {}",
                        key
                    )))
                }
            }
        } else {
            Err(Error::custom("Op subject must be a Link or Ref, e.g. \"/sbin/value/string\" or \"$result/filter\""))
        }
    }
}

impl<'de> Deserialize<'de> for Op {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(OpVisitor)
    }
}

impl Serialize for Op {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Op::Get { subject, key } => {
                let mut op = s.serialize_map(Some(1))?;
                let key: TCValue = vec![(**key).clone()].into();
                op.serialize_entry(subject, &key)?;
                op.end()
            }
            Op::Put {
                subject,
                key,
                value,
            } => {
                let mut op = s.serialize_map(Some(1))?;
                op.serialize_entry(
                    subject,
                    &TCValue::Vector([*key.clone(), *value.clone()].to_vec()),
                )?;
                op.end()
            }
            Op::Post {
                subject,
                action,
                requires,
            } => {
                let subject = if let Some(subject) = subject {
                    format!("{}{}", subject, action)
                } else {
                    format!("{}", action)
                };

                let mut op = s.serialize_map(Some(1))?;
                op.serialize_entry(&subject, requires)?;
                op.end()
            }
        }
    }
}
