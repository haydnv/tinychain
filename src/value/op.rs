use std::convert::{TryFrom, TryInto};
use std::fmt;

use crate::class::{Class, Instance, TCResult};
use crate::error;

use super::link::{Link, TCPath};
use super::{label, TCRef, TCString, Value, ValueId, ValueType};

#[derive(Clone, Eq, PartialEq)]
pub enum OpDefType {
    Get,
}

impl Class for OpDefType {
    type Instance = OpDef;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;
        if &suffix == "/get" {
            Ok(OpDefType::Get)
        } else {
            Err(error::not_found(suffix))
        }
    }

    fn prefix() -> TCPath {
        OpType::prefix().join(label("def").into())
    }
}

impl From<OpDefType> for Link {
    fn from(odt: OpDefType) -> Link {
        let prefix = OpDefType::prefix();
        match odt {
            OpDefType::Get => prefix.join(label("get").into()).into(),
        }
    }
}

impl fmt::Display for OpDefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET Op definition"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum OpRefType {
    Get,
    Put,
}

impl Class for OpRefType {
    type Instance = OpRef;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;
        if &suffix == "/get" {
            Ok(OpRefType::Get)
        } else if &suffix == "/put" {
            Ok(OpRefType::Put)
        } else {
            Err(error::not_found(suffix))
        }
    }

    fn prefix() -> TCPath {
        OpType::prefix().join(label("ref").into())
    }
}

impl From<OpRefType> for Link {
    fn from(ort: OpRefType) -> Link {
        let prefix = OpRefType::prefix();
        match ort {
            OpRefType::Get => prefix.join(label("get").into()).into(),
            OpRefType::Put => prefix.join(label("put").into()).into(),
        }
    }
}

impl fmt::Display for OpRefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET Op ref"),
            Self::Put => write!(f, "type: PUT Op ref"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum OpType {
    Def(OpDefType),
    If,
    Ref(OpRefType),
}

impl Class for OpType {
    type Instance = Op;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            Err(error::unsupported("You must specify a type of Op"))
        } else {
            match suffix[0].as_str() {
                "def" => OpDefType::from_path(path).map(OpType::Def),
                "if" if suffix.len() == 1 => Ok(OpType::If),
                "ref" => OpRefType::from_path(path).map(OpType::Ref),
                other => Err(error::not_found(other)),
            }
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
            OpType::Def(odt) => odt.into(),
            OpType::If => prefix.join(label("if").into()).into(),
            OpType::Ref(ort) => ort.into(),
        }
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Def(odt) => write!(f, "{}", odt),
            Self::If => write!(f, "type: Conditional Op"),
            Self::Ref(ort) => write!(f, "{}", ort),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Capture {
    Id(ValueId),
    Ids(Vec<ValueId>),
}

impl Capture {
    pub fn contains(&self, other: &ValueId) -> bool {
        match self {
            Self::Id(this) => this == other,
            Self::Ids(these) => these.contains(other),
        }
    }

    pub fn to_vec(&self) -> Vec<ValueId> {
        match self {
            Capture::Id(id) => vec![id.clone()],
            Capture::Ids(ids) => ids.to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Id(_) => 1,
            Self::Ids(ids) => ids.len(),
        }
    }
}

impl TryFrom<Value> for Capture {
    type Error = error::TCError;

    fn try_from(v: Value) -> TCResult<Capture> {
        if let Value::TCString(TCString::Id(id)) = v {
            Ok(Capture::Id(id))
        } else {
            v.try_into().map(Capture::Ids)
        }
    }
}

impl TryFrom<Option<Value>> for Capture {
    type Error = error::TCError;

    fn try_from(v: Option<Value>) -> TCResult<Capture> {
        match v {
            Some(value) => value.try_into(),
            None => Ok(Capture::Ids(vec![])),
        }
    }
}

impl From<Capture> for Vec<ValueId> {
    fn from(c: Capture) -> Vec<ValueId> {
        match c {
            Capture::Id(id) => vec![id],
            Capture::Ids(ids) => ids,
        }
    }
}

impl fmt::Display for Capture {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id(id) => write!(f, "{}", id),
            Self::Ids(ids) => write!(
                f,
                "({})",
                ids.iter()
                    .map(String::from)
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}

pub type Cond = (TCRef, Value, Value);
pub type GetOp = (TCRef, Vec<(ValueId, Value)>, TCRef);

#[derive(Clone, Eq, PartialEq)]
pub enum OpDef {
    Get(GetOp),
}

impl Instance for OpDef {
    type Class = OpDefType;

    fn class(&self) -> OpDefType {
        match self {
            Self::Get(_) => OpDefType::Get,
        }
    }
}

impl fmt::Display for OpDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get((input, _def, output)) => write!(f, "GET Op {} -> {}", input, output),
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

impl Instance for OpRef {
    type Class = OpRefType;

    fn class(&self) -> OpRefType {
        match self {
            Self::Get(_, _) => OpRefType::Get,
            Self::Put(_, _, _) => OpRefType::Put,
        }
    }
}

impl fmt::Display for OpRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpRef::Get(subject, id) => write!(f, "OpRef::Get {}: {}", subject, id),
            OpRef::Put(subject, id, val) => write!(f, "OpRef::Put {}: {} <- {}", subject, id, val),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Op {
    Def(OpDef),
    If(Cond),
    Ref(OpRef),
}

impl Instance for Op {
    type Class = OpType;

    fn class(&self) -> OpType {
        match self {
            Self::Def(op_def) => OpType::Def(op_def.class()),
            Self::If(_) => OpType::If,
            Self::Ref(op_ref) => OpType::Ref(op_ref.class()),
        }
    }
}

impl From<OpRef> for Op {
    fn from(op_ref: OpRef) -> Op {
        Op::Ref(op_ref)
    }
}

impl TryFrom<Op> for GetOp {
    type Error = error::TCError;

    fn try_from(op: Op) -> TCResult<GetOp> {
        match op {
            Op::Def(OpDef::Get(get_op)) => Ok(get_op),
            other => Err(error::bad_request("Expected GetOp but found", other)),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Def(op_def) => write!(f, "{}", op_def),
            Op::If((cond, then, or_else)) => write!(
                f,
                "Op::If({} then {{ {} }} else {{ {} }})",
                cond, then, or_else
            ),
            Op::Ref(op_ref) => write!(f, "{}", op_ref),
        }
    }
}
