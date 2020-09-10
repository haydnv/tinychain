use std::fmt;

use crate::class::{Class, Instance, TCResult};
use crate::error;

use super::link::{Link, TCPath};
use super::{label, TCRef, Value, ValueType};

#[derive(Clone, Eq, PartialEq)]
pub enum OpType {
    If,
    Ref,
}

impl Class for OpType {
    type Instance = Op;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let path = path.from_path(&Self::prefix())?;

        if path.is_empty() {
            Err(error::unsupported("You must specify a type of Op"))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "if" => Ok(OpType::If),
                "ref" => Ok(OpType::Ref),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        ValueType::prefix().join(label("op").into())
    }
}

impl From<OpType> for Link {
    fn from(ot: OpType) -> Link {
        let prefix = OpType::prefix();
        match ot {
            OpType::If => prefix.join(label("if").into()).into(),
            OpType::Ref => prefix.join(label("ref").into()).into(),
        }
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::If => write!(f, "type: Conditional Op"),
            Self::Ref => write!(f, "type: Op reference"),
        }
    }
}

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum Subject {
    Ref(TCRef),
    Link(Link),
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Subject::Ref(r) => write!(f, "{}", r),
            Subject::Link(l) => write!(f, "{}", l),
        }
    }
}

impl From<TCRef> for Subject {
    fn from(r: TCRef) -> Subject {
        Subject::Ref(r)
    }
}

impl From<Link> for Subject {
    fn from(link: Link) -> Subject {
        Subject::Link(link)
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum OpRef {
    Get(Subject, Value),
    Put(Subject, Value, Value),
}

impl fmt::Display for OpRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpRef::Get(subject, id) => write!(f, "OpRef::Get {}: {}", subject, id),
            OpRef::Put(subject, id, val) => write!(f, "OpRef::Put {}: {} <- {}", subject, id, val),
        }
    }
}

pub type Cond = (TCRef, Value, Value);

#[derive(Clone, Eq, PartialEq)]
pub enum Op {
    If(Cond),
    Ref(OpRef),
}

impl Instance for Op {
    type Class = OpType;

    fn class(&self) -> OpType {
        match self {
            Self::If(_) => OpType::If,
            Self::Ref(_) => OpType::Ref,
        }
    }
}

impl From<OpRef> for Op {
    fn from(op_ref: OpRef) -> Op {
        Op::Ref(op_ref)
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::If((cond, then, or_else)) => write!(
                f,
                "Op::If({} then {{ {} }} else {{ {} }})",
                cond, then, or_else
            ),
            Op::Ref(op_ref) => write!(f, "{}", op_ref),
        }
    }
}
