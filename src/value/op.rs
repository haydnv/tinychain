use std::collections::HashSet;
use std::fmt;

use serde::de::{Deserializer, Error, MapAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde::Deserialize;

use crate::value::link::*;
use crate::value::*;

#[derive(Clone, Hash, Eq, PartialEq, Deserialize)]
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

impl From<TCPath> for Subject {
    fn from(p: TCPath) -> Subject {
        Subject::Link(p.into())
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
pub struct GetOp {
    pub subject: Subject,
    pub key: TCValue,
}

impl<S: Into<Subject>> From<(S, TCValue)> for GetOp {
    fn from(tup: (S, TCValue)) -> GetOp {
        GetOp {
            subject: tup.0.into(),
            key: tup.1,
        }
    }
}

impl fmt::Display for GetOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GET {}/{}", self.subject, self.key)
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct PutOp {
    pub subject: Subject,
    pub key: TCValue,
    pub value: TCValue,
}

impl<S: Into<Subject>> From<(S, TCValue, TCValue)> for PutOp {
    fn from(tup: (S, TCValue, TCValue)) -> PutOp {
        PutOp {
            subject: tup.0.into(),
            key: tup.1,
            value: tup.2,
        }
    }
}

impl fmt::Display for PutOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PUT {}/{}:{}", self.subject, self.key, self.value)
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct PostOp {
    pub subject: TCRef,
    pub action: TCPath,
    pub requires: Vec<(ValueId, TCValue)>,
}

impl From<(TCRef, TCPath, Vec<(ValueId, TCValue)>)> for PostOp {
    fn from(tup: (TCRef, TCPath, Vec<(ValueId, TCValue)>)) -> PostOp {
        PostOp {
            subject: tup.0,
            action: tup.1,
            requires: tup.2,
        }
    }
}

impl fmt::Display for PostOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "POST {}{}: {:?}",
            self.subject, self.action, self.requires
        )
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum Op {
    Get(GetOp),
    Put(PutOp),
    Post(PostOp),
}

impl Op {
    pub fn deps(&self) -> HashSet<TCRef> {
        let mut deps = vec![];
        match self {
            Op::Get(GetOp { subject, key }) => {
                if let Subject::Ref(r) = subject {
                    deps.push(r);
                }
                if let TCValue::Ref(r) = key {
                    deps.push(&r);
                }
            }
            Op::Put(PutOp {
                subject,
                key,
                value,
            }) => {
                if let Subject::Ref(r) = subject {
                    deps.push(r);
                }
                if let TCValue::Ref(r) = key {
                    deps.push(&r);
                }
                if let TCValue::Ref(r) = value {
                    deps.push(&r);
                }
            }
            Op::Post(PostOp {
                subject,
                action: _,
                requires,
            }) => {
                deps.push(subject);
                for (_, v) in requires {
                    if let TCValue::Ref(r) = v {
                        deps.push(&r);
                    }
                }
            }
        }

        deps.into_iter().cloned().collect()
    }
}

impl From<GetOp> for Op {
    fn from(op: GetOp) -> Op {
        Op::Get(op)
    }
}

impl From<PutOp> for Op {
    fn from(op: PutOp) -> Op {
        Op::Put(op)
    }
}

impl From<PostOp> for Op {
    fn from(op: PostOp) -> Op {
        Op::Post(op)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Get(get_op) => write!(f, "{}", get_op),
            Op::Put(put_op) => write!(f, "{}", put_op),
            Op::Post(post_op) => write!(f, "{}", post_op),
        }
    }
}

// TODO: split this into op-specific Visitors
struct OpVisitor;

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
                let action: TCPath = key[1..]
                    .iter()
                    .map(|s| s.parse())
                    .collect::<TCResult<Vec<PathSegment>>>()
                    .map_err(Error::custom)?
                    .into();
                let requires = access.next_value::<Vec<(ValueId, TCValue)>>()?;

                Ok(PostOp {
                    subject,
                    action,
                    requires,
                }
                .into())
            } else {
                let subject: TCRef = key[1..].parse().map_err(Error::custom)?;
                let mut value = access.next_value::<Vec<TCValue>>()?;

                if value.len() == 1 {
                    Ok(GetOp {
                        subject: subject.into(),
                        key: value.remove(0),
                    }
                    .into())
                } else if value.len() == 2 {
                    Ok(PutOp {
                        subject: subject.into(),
                        key: value.remove(0),
                        value: value.remove(0),
                    }
                    .into())
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

// TODO: split this into op-specific definitions
impl Serialize for Op {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Op::Get(GetOp { subject, key }) => {
                let mut op = s.serialize_map(Some(1))?;
                let key: TCValue = vec![key.clone()].into();
                op.serialize_entry(subject, &key)?;
                op.end()
            }
            Op::Put(PutOp {
                subject,
                key,
                value,
            }) => {
                let mut op = s.serialize_map(Some(1))?;
                op.serialize_entry(
                    subject,
                    &TCValue::Vector([key.clone(), value.clone()].to_vec()),
                )?;
                op.end()
            }
            Op::Post(PostOp {
                subject,
                action,
                requires,
            }) => {
                let mut op = s.serialize_map(Some(1))?;
                op.serialize_entry(&format!("{}{}", subject, action), requires)?;
                op.end()
            }
        }
    }
}
