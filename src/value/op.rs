use std::convert::TryFrom;
use std::fmt;

use crate::class::{Class, Instance, TCResult};
use crate::error;

use super::link::{Link, TCPath};
use super::{label, TCRef, Value, ValueId, ValueType};

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
pub enum MethodType {
    Get,
    Put,
}

impl Class for MethodType {
    type Instance = Method;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;
        if &suffix == "/get" {
            Ok(MethodType::Get)
        } else if &suffix == "/put" {
            Ok(MethodType::Put)
        } else {
            Err(error::not_found(suffix))
        }
    }

    fn prefix() -> TCPath {
        OpType::prefix().join(label("method").into())
    }
}

impl From<MethodType> for Link {
    fn from(mt: MethodType) -> Link {
        let prefix = MethodType::prefix();
        match mt {
            MethodType::Get => prefix.join(label("get").into()).into(),
            MethodType::Put => prefix.join(label("put").into()).into(),
        }
    }
}

impl fmt::Display for MethodType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET method"),
            Self::Put => write!(f, "type: PUT method"),
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
    Method(MethodType),
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
            OpType::Method(mt) => mt.into(),
            OpType::Ref(ort) => ort.into(),
        }
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Def(odt) => write!(f, "{}", odt),
            Self::If => write!(f, "type: Conditional Op"),
            Self::Method(mt) => write!(f, "{}", mt),
            Self::Ref(ort) => write!(f, "{}", ort),
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

#[derive(Clone, Eq, PartialEq)]
pub enum Method {
    Get(TCRef, TCPath, Value),
    Put(TCRef, TCPath, Value, Value),
}

impl Instance for Method {
    type Class = MethodType;

    fn class(&self) -> MethodType {
        match self {
            Self::Get(_, _, _) => MethodType::Get,
            Self::Put(_, _, _, _) => MethodType::Put,
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get(subject, path, _) => write!(f, "GET {}{}", subject, path),
            Self::Put(subject, path, _, _) => write!(f, "PUT {}{}", subject, path),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum OpRef {
    Get(Link, Value),
    Put(Link, Value, Value),
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
            OpRef::Get(link, id) => write!(f, "OpRef::Get {}: {}", link, id),
            OpRef::Put(path, id, val) => write!(f, "OpRef::Put {}: {} <- {}", path, id, val),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Op {
    Def(OpDef),
    If(Cond),
    Method(Method),
    Ref(OpRef),
}

impl Instance for Op {
    type Class = OpType;

    fn class(&self) -> OpType {
        match self {
            Self::Def(op_def) => OpType::Def(op_def.class()),
            Self::If(_) => OpType::If,
            Self::Method(method) => OpType::Method(method.class()),
            Self::Ref(op_ref) => OpType::Ref(op_ref.class()),
        }
    }
}

impl From<Method> for Op {
    fn from(method: Method) -> Op {
        Op::Method(method)
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
            Op::Method(method) => write!(f, "{}", method),
            Op::Ref(op_ref) => write!(f, "{}", op_ref),
        }
    }
}
