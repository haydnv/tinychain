use std::collections::HashSet;
use std::fmt;

use serde::de::{Error, MapAccess, Visitor};
use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
    pub key: TCValue,
}

impl From<(TCValue,)> for GetOp {
    fn from(tup: (TCValue,)) -> GetOp {
        GetOp { key: tup.0 }
    }
}

impl Serialize for GetOp {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut op = s.serialize_seq(Some(1))?;
        op.serialize_element(&self.key)?;
        op.end()
    }
}

impl fmt::Display for GetOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GET {}", self.key)
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct PutOp {
    pub key: TCValue,
    pub value: TCValue,
}

impl From<(TCValue, TCValue)> for PutOp {
    fn from(tup: (TCValue, TCValue)) -> PutOp {
        PutOp {
            key: tup.0,
            value: tup.1,
        }
    }
}

impl Serialize for PutOp {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut op = s.serialize_seq(Some(1))?;
        op.serialize_element(&self.key)?;
        op.serialize_element(&self.value)?;
        op.end()
    }
}

impl fmt::Display for PutOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PUT {}:{}", self.key, self.value)
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct PostOp {
    pub action: TCPath,
    pub requires: Vec<(ValueId, TCValue)>,
}

impl From<(TCPath, Vec<(ValueId, TCValue)>)> for PostOp {
    fn from(tup: (TCPath, Vec<(ValueId, TCValue)>)) -> PostOp {
        PostOp {
            action: tup.0,
            requires: tup.1,
        }
    }
}

impl fmt::Display for PostOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "POST {}: {:?}", self.action, self.requires)
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum OpArgs {
    Get(GetOp),
    Put(PutOp),
    Post(PostOp),
}

impl fmt::Display for OpArgs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use OpArgs::*;
        match self {
            Get(get_op) => write!(f, "{}", get_op),
            Put(put_op) => write!(f, "{}", put_op),
            Post(post_op) => write!(f, "{}", post_op),
        }
    }
}

impl From<GetOp> for OpArgs {
    fn from(op: GetOp) -> OpArgs {
        OpArgs::Get(op)
    }
}

impl From<PutOp> for OpArgs {
    fn from(op: PutOp) -> OpArgs {
        OpArgs::Put(op)
    }
}

impl From<PostOp> for OpArgs {
    fn from(op: PostOp) -> OpArgs {
        OpArgs::Post(op)
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub struct Op {
    subject: Subject,
    args: OpArgs,
}

impl Op {
    pub fn subject(&'_ self) -> &'_ Subject {
        &self.subject
    }

    pub fn args(&'_ self) -> &'_ OpArgs {
        &self.args
    }

    pub fn deps(&self) -> HashSet<TCRef> {
        let mut deps = vec![];
        if let Subject::Ref(r) = &self.subject {
            deps.push(r);
        }

        match &self.args {
            OpArgs::Get(GetOp { key }) => {
                if let TCValue::Ref(r) = key {
                    deps.push(r);
                }
            }
            OpArgs::Put(PutOp { key, value }) => {
                if let TCValue::Ref(r) = key {
                    deps.push(r);
                }
                if let TCValue::Ref(r) = value {
                    deps.push(r);
                }
            }
            OpArgs::Post(PostOp {
                action: _,
                requires,
            }) => {
                for (_, v) in requires {
                    if let TCValue::Ref(r) = v {
                        deps.push(r);
                    }
                }
            }
        }

        deps.into_iter().cloned().collect()
    }
}

impl<S: Into<Subject>, A: Into<OpArgs>> From<(S, A)> for Op {
    fn from((subject, args): (S, A)) -> Op {
        Op {
            subject: subject.into(),
            args: args.into(),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} - {}", self.subject, self.args)
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
                let args: PostOp = (action, requires).into();
                Ok((subject, args).into())
            } else {
                let subject: TCRef = key[1..].parse().map_err(Error::custom)?;
                let mut value = access.next_value::<Vec<TCValue>>()?;

                if value.len() == 1 {
                    let args: GetOp = (value.remove(0),).into();
                    Ok((subject, args).into())
                } else if value.len() == 2 {
                    let args: PutOp = (value.remove(0), value.remove(0)).into();
                    Ok((subject, args).into())
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
        let mut op = s.serialize_map(Some(1))?;

        use OpArgs::*;
        match &self.args {
            Get(get_op) => op.serialize_entry(&self.subject, &get_op)?,
            Put(put_op) => op.serialize_entry(&self.subject, &put_op)?,
            Post(post_op) => {
                let subject = format!("{}{}", self.subject, post_op.action);
                op.serialize_entry(&subject, &post_op.requires)?
            }
        }

        op.end()
    }
}
