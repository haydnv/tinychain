use std::fmt;

use serde::ser::{Serialize, SerializeStructVariant, Serializer};
use serde::Deserialize;

use crate::value::{Link, TCRef, TCValue};

#[derive(Clone, Debug, Hash, Eq, PartialEq, Deserialize, serde::Serialize)]
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

impl Serialize for Op {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Op::Get { subject, key } => {
                let mut op = s.serialize_struct_variant("Op", 0, "Get", 2)?;
                op.serialize_field("subject", subject)?;
                op.serialize_field("key", key)?;
                op.end()
            }
            Op::Put {
                subject,
                key,
                value,
            } => {
                let mut op = s.serialize_struct_variant("Op", 1, "Put", 3)?;
                op.serialize_field("subject", subject)?;
                op.serialize_field("key", key)?;
                op.serialize_field("value", value)?;
                op.end()
            }
            Op::Post {
                subject,
                action,
                requires,
            } => {
                let mut op = s.serialize_struct_variant("Op", 2, "Post", 3)?;
                op.serialize_field("subject", subject)?;
                op.serialize_field("action", action)?;
                op.serialize_field("requires", requires)?;
                op.end()
            }
        }
    }
}
