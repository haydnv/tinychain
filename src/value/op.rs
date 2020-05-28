use std::collections::HashSet;
use std::fmt;

use serde::ser::{SerializeMap, SerializeSeq};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

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
pub struct Request {
    subject: Subject,
    op: OpArgs,
}

impl Request {
    pub fn subject(&'_ self) -> &'_ Subject {
        &self.subject
    }

    pub fn op(&'_ self) -> &'_ OpArgs {
        &self.op
    }

    pub fn deps(&self) -> HashSet<TCRef> {
        let mut deps = vec![];
        if let Subject::Ref(r) = &self.subject {
            deps.push(r);
        }

        match &self.op {
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

impl TryFrom<(&str, Vec<TCValue>)> for Request {
    type Error = error::TCError;

    fn try_from(tup: (&str, Vec<TCValue>)) -> TCResult<Request> {
        let key = tup.0;
        let mut values = tup.1;

        if key.starts_with('$') && key.contains('/') {
            let key: Vec<&str> = key.split('/').collect();
            if key.len() != 2 || key[0].is_empty() || key[1].is_empty() {
                return Err(error::bad_request(
                    "Invalid Ref specified for POST Op",
                    key.join("/"),
                ));
            }

            let subject: TCRef = key[0].parse()?;
            let action: TCPath = key[1..]
                .iter()
                .map(|s| s.parse())
                .collect::<TCResult<Vec<PathSegment>>>()?
                .into();

            let requires = values
                .drain(..)
                .map(|v| v.try_into())
                .collect::<TCResult<Vec<(ValueId, TCValue)>>>()?;
            let args: PostOp = (action, requires).into();
            Ok((subject, args).into())
        } else {
            let subject: Subject = if key.starts_with('/') || key.starts_with("http://") {
                let subject: TCPath = key.parse()?;
                subject.into()
            } else if key.starts_with('$') {
                let subject: TCRef = key.parse()?;
                subject.into()
            } else {
                return Err(error::bad_request("Unrecognized Op subject", key));
            };

            if values.len() == 1 {
                let args: GetOp = (values.remove(0),).into();
                Ok((subject, args).into())
            } else if values.len() == 2 {
                let args: PutOp = (values.remove(0), values.remove(0)).into();
                Ok((subject, args).into())
            } else {
                Err(error::bad_request(
                    "Expected either 1 (for a Get), or 2 (for a Put) Values for",
                    key,
                ))
            }
        }
    }
}

impl<S: Into<Subject>, O: Into<OpArgs>> From<(S, O)> for Request {
    fn from((subject, args): (S, O)) -> Request {
        Request {
            subject: subject.into(),
            op: args.into(),
        }
    }
}

impl fmt::Display for Request {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} - {}", self.subject, self.op)
    }
}

// TODO: split this into op-specific Visitors
struct OpVisitor;

impl<'de> de::Visitor<'de> for OpVisitor {
    type Value = Request;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain Op, e.g. {\"/sbin/table/new\": [(\"columns\", [(\"foo\", \"/sbin/value/string\"), ...]]}")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: de::MapAccess<'de>,
    {
        if let Some(key) = access.next_key::<&str>()? {
            let values: Vec<TCValue> = access.next_value()?;
            (key, values).try_into().map_err(de::Error::custom)
        } else {
            Err(de::Error::custom("Op subject must be a Link or Ref, e.g. \"/sbin/value/string\" or \"$result/filter\""))
        }
    }
}

impl<'de> Deserialize<'de> for Request {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(OpVisitor)
    }
}

impl Serialize for Request {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut op = s.serialize_map(Some(1))?;

        use OpArgs::*;
        match &self.op {
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
