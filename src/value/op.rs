use std::fmt;

use serde::de::{Deserializer, Error, MapAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde::Deserialize;

use crate::value::{Link, TCRef, TCValue};

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize)]
pub enum Subject {
    Link(Link),
    Ref(TCRef),
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Subject::Link(l) => write!(f, "{}", l),
            Subject::Ref(r) => write!(f, "{}", r),
        }
    }
}

impl From<Link> for Subject {
    fn from(l: Link) -> Subject {
        Subject::Link(l)
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
        action: Link,
        requires: Vec<(String, TCValue)>,
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

    pub fn post(subject: Option<TCRef>, action: Link, requires: Vec<(String, TCValue)>) -> Op {
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
                "subject: {}, action: {}, requires: {:?}",
                subject
                    .clone()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| String::from("None")),
                action,
                requires
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
                let subject = TCRef::to(&key[0][1..]).map_err(Error::custom)?;
                let method =
                    Link::to(&format!("/{}", key[1..].join("/"))).map_err(Error::custom)?;
                let requires = access.next_value::<Vec<(String, TCValue)>>()?;

                Ok(Op::post(subject.into(), method, requires))
            } else {
                let subject = TCRef::to(&key[1..].to_string()).map_err(Error::custom)?;
                let value = access.next_value::<Vec<TCValue>>()?;

                if value.len() == 1 {
                    Ok(Op::get(subject.into(), value[0].clone()))
                } else if value.len() == 2 {
                    Ok(Op::put(subject.into(), value[0].clone(), value[1].clone()))
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

impl<'de> Deserialize<'de> for Op where {
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
                op.serialize_entry(subject, &TCValue::Vector([*key.clone()].to_vec()))?;
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
