use std::fmt;

use super::link::Link;
use super::{TCRef, Value};

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

#[derive(Clone, PartialEq)]
pub enum Op {
    Get(Subject, Value),
    Put(Subject, Value, Value),
}

impl Op {
    pub fn subject(&'_ self) -> &'_ Subject {
        match self {
            Self::Get(subject, _) => subject,
            Self::Put(subject, _, _) => subject,
        }
    }

    pub fn object(&'_ self) -> &'_ Value {
        match self {
            Self::Get(_, object) => object,
            Self::Put(_, object, _) => object,
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Get(subject, id) => write!(f, "Op::Get {}: {}", subject, id),
            Op::Put(subject, id, val) => write!(f, "Op::Put {}: {} <- {}", subject, id, val),
        }
    }
}
