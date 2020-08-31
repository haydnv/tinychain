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
    If(TCRef, Value, Value),
    Get(Subject, Value),
    Put(Subject, Value, Value),
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::If(cond, then, or_else) => write!(
                f,
                "Op::If({} then {{ {} }} else {{ {} }})",
                cond, then, or_else
            ),
            Op::Get(subject, id) => write!(f, "Op::Get {}: {}", subject, id),
            Op::Put(subject, id, val) => write!(f, "Op::Put {}: {} <- {}", subject, id, val),
        }
    }
}
