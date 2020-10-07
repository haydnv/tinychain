use std::fmt;
use std::iter;
use std::sync::Arc;

use futures::stream;
use serde::{Serialize, Serializer};

use crate::auth::Auth;
use crate::class::{Class, Instance, State, TCBoxTryFuture, TCResult};
use crate::error;
use crate::transaction::Txn;

use super::link::{Link, TCPath};
use super::{
    label, Scalar, ScalarClass, ScalarInstance, ScalarType, TCRef, TryCastFrom, TryCastInto, Value,
    ValueId,
};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum OpDefType {
    Get,
    Put,
    Post,
}

impl Class for OpDefType {
    type Instance = OpDef;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;
        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "get" => Ok(OpDefType::Get),
                "put" => Ok(OpDefType::Put),
                "post" => Ok(OpDefType::Post),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::not_found(suffix))
        }
    }

    fn prefix() -> TCPath {
        OpType::prefix().join(label("def").into())
    }
}

impl ScalarClass for OpDefType {
    type Instance = OpDef;

    fn size(self) -> Option<usize> {
        None
    }

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<OpDef> {
        let scalar: Scalar = scalar.into();
        OpDef::try_cast_from(scalar, |v| error::bad_request("Not a valid OpDef", v))
    }
}

impl From<OpDefType> for Link {
    fn from(odt: OpDefType) -> Link {
        let prefix = OpDefType::prefix();
        match odt {
            OpDefType::Get => prefix.join(label("get").into()).into(),
            OpDefType::Put => prefix.join(label("put").into()).into(),
            OpDefType::Post => prefix.join(label("post").into()).into(),
        }
    }
}

impl fmt::Display for OpDefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET Op definition"),
            Self::Put => write!(f, "type: PUT Op definition"),
            Self::Post => write!(f, "type: POST Op definition"),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum MethodType {
    Get,
    Put,
    Post,
}

impl Class for MethodType {
    type Instance = Method;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;
        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "get" => Ok(MethodType::Get),
                "put" => Ok(MethodType::Put),
                "post" => Ok(MethodType::Post),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::not_found(suffix))
        }
    }

    fn prefix() -> TCPath {
        OpType::prefix().join(label("method").into())
    }
}

impl ScalarClass for MethodType {
    type Instance = Method;

    fn size(self) -> Option<usize> {
        None
    }

    fn try_cast<S: Into<Scalar>>(&self, _scalar: S) -> TCResult<Method> {
        Err(error::not_implemented("Cast Scalar into Method"))
    }
}

impl From<MethodType> for Link {
    fn from(mt: MethodType) -> Link {
        let prefix = MethodType::prefix();
        match mt {
            MethodType::Get => prefix.join(label("get").into()).into(),
            MethodType::Put => prefix.join(label("put").into()).into(),
            MethodType::Post => prefix.join(label("post").into()).into(),
        }
    }
}

impl fmt::Display for MethodType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET method"),
            Self::Put => write!(f, "type: PUT method"),
            Self::Post => write!(f, "type: POST method"),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum OpRefType {
    If,
    Get,
    Put,
    Post,
}

impl Class for OpRefType {
    type Instance = OpRef;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;
        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "if" => Ok(OpRefType::If),
                "get" => Ok(OpRefType::Get),
                "put" => Ok(OpRefType::Put),
                "post" => Ok(OpRefType::Post),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::not_found(suffix))
        }
    }

    fn prefix() -> TCPath {
        OpType::prefix().join(label("ref").into())
    }
}

impl ScalarClass for OpRefType {
    type Instance = OpRef;

    fn size(self) -> Option<usize> {
        None
    }

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<OpRef> {
        let scalar: Scalar = scalar.into();
        scalar.try_cast_into(|v| error::bad_request(format!("Cannot cast into {} from", self), v))
    }
}

impl From<OpRefType> for Link {
    fn from(ort: OpRefType) -> Link {
        let prefix = OpRefType::prefix();
        match ort {
            OpRefType::If => prefix.join(label("if").into()).into(),
            OpRefType::Get => prefix.join(label("get").into()).into(),
            OpRefType::Put => prefix.join(label("put").into()).into(),
            OpRefType::Post => prefix.join(label("post").into()).into(),
        }
    }
}

impl fmt::Display for OpRefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::If => write!(f, "type: conditional Op"),
            Self::Get => write!(f, "type: GET Op ref"),
            Self::Put => write!(f, "type: PUT Op ref"),
            Self::Post => write!(f, "type: POST Op ref"),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum OpType {
    Def(OpDefType),
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
                "method" => MethodType::from_path(path).map(OpType::Method),
                "ref" => OpRefType::from_path(path).map(OpType::Ref),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPath {
        ScalarType::prefix().join(label("op").into())
    }
}

impl ScalarClass for OpType {
    type Instance = Op;

    fn size(self) -> Option<usize> {
        None
    }

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<Op> {
        let scalar: Scalar = scalar.into();
        match self {
            Self::Def(odt) => odt.try_cast(scalar).map(Op::Def),
            Self::Method(mt) => mt.try_cast(scalar).map(Op::Method),
            Self::Ref(ort) => ort.try_cast(scalar).map(Op::Ref),
        }
    }
}

impl From<OpType> for Link {
    fn from(ot: OpType) -> Link {
        match ot {
            OpType::Def(odt) => odt.into(),
            OpType::Method(mt) => mt.into(),
            OpType::Ref(ort) => ort.into(),
        }
    }
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Def(odt) => write!(f, "{}", odt),
            Self::Method(mt) => write!(f, "{}", mt),
            Self::Ref(ort) => write!(f, "{}", ort),
        }
    }
}

pub type GetOp = (ValueId, Vec<(ValueId, Scalar)>);
pub type PutOp = (ValueId, ValueId, Vec<(ValueId, Scalar)>);
pub type PostOp = Vec<(ValueId, Scalar)>;

#[derive(Clone, Eq, PartialEq)]
pub enum OpDef {
    Get(GetOp),
    Put(PutOp),
    Post(PostOp),
}

impl OpDef {
    pub fn get<'a>(&'a self, txn: Arc<Txn>, key: Value, auth: Auth) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if let Self::Get((key_id, def)) = self {
                let data = iter::once((key_id.clone(), Scalar::Value(key))).chain(def.to_vec());
                txn.execute(stream::iter(data), auth).await
            } else {
                Err(error::method_not_allowed(self))
            }
        })
    }
}

impl Instance for OpDef {
    type Class = OpDefType;

    fn class(&self) -> OpDefType {
        match self {
            Self::Get(_) => OpDefType::Get,
            Self::Put(_) => OpDefType::Put,
            Self::Post(_) => OpDefType::Post,
        }
    }
}

impl ScalarInstance for OpDef {
    type Class = OpDefType;
}

impl TryCastFrom<Scalar> for OpDef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<PutOp>() || scalar.matches::<GetOp>() || scalar.matches::<PostOp>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<OpDef> {
        if scalar.matches::<PutOp>() {
            scalar.opt_cast_into().map(OpDef::Put)
        } else if scalar.matches::<GetOp>() {
            scalar.opt_cast_into().map(OpDef::Get)
        } else if scalar.matches::<PostOp>() {
            scalar.opt_cast_into().map(OpDef::Post)
        } else {
            None
        }
    }
}

impl fmt::Display for OpDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get(_) => write!(f, "GET Op"),
            Self::Put(_) => write!(f, "PUT Op"),
            Self::Post(_) => write!(f, "POST"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Method {
    Get(TCRef, (TCPath, Value)),
    Put(TCRef, (TCPath, Value, Scalar)),
    Post(TCRef, (TCPath, Vec<(ValueId, Scalar)>)),
}

impl Instance for Method {
    type Class = MethodType;

    fn class(&self) -> MethodType {
        match self {
            Self::Get(_, _) => MethodType::Get,
            Self::Put(_, _) => MethodType::Put,
            Self::Post(_, _) => MethodType::Post,
        }
    }
}

impl ScalarInstance for Method {
    type Class = MethodType;
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get(subject, (path, _)) => write!(f, "GET {}: {}", subject, path),
            Self::Put(subject, (path, _, _)) => write!(f, "PUT {}{}", subject, path),
            Self::Post(subject, (path, _)) => write!(f, "PUT {}{}", subject, path),
        }
    }
}

type GetRef = (Link, Value);
type PutRef = (Link, Value, Scalar);
type PostRef = (Link, Vec<(ValueId, Scalar)>);

#[derive(Clone, Eq, PartialEq)]
pub enum OpRef {
    If(TCRef, Scalar, Scalar),
    Get(GetRef),
    Put(PutRef),
    Post(PostRef),
}

impl Instance for OpRef {
    type Class = OpRefType;

    fn class(&self) -> OpRefType {
        match self {
            Self::If(_, _, _) => OpRefType::If,
            Self::Get((_, _)) => OpRefType::Get,
            Self::Put((_, _, _)) => OpRefType::Put,
            Self::Post((_, _)) => OpRefType::Post,
        }
    }
}

impl ScalarInstance for OpRef {
    type Class = OpRefType;
}

impl TryCastFrom<Scalar> for OpRef {
    fn can_cast_from(s: &Scalar) -> bool {
        s.matches::<(TCRef, Value, Value)>()
            || s.matches::<(Link, Vec<(ValueId, Value)>)>()
            || s.matches::<(Link, Value, Value)>()
            || s.matches::<(Link, Value)>()
    }

    fn opt_cast_from(s: Scalar) -> Option<OpRef> {
        if s.matches::<(TCRef, Value, Value)>() {
            let (cond, then, or_else) = s.opt_cast_into().unwrap();
            Some(OpRef::If(cond, then, or_else))
        } else {
            unimplemented!()
        }
    }
}

impl fmt::Display for OpRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpRef::If(cond, then, or_else) => {
                write!(f, "OpRef::If ({}) then {} else {}", cond, then, or_else)
            }
            OpRef::Get((link, id)) => write!(f, "OpRef::Get {}: {}", link, id),
            OpRef::Put((path, id, val)) => write!(f, "OpRef::Put {}: {} <- {}", path, id, val),
            OpRef::Post((path, _)) => write!(f, "OpRef::Post {}", path),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Op {
    Def(OpDef),
    Method(Method),
    Ref(OpRef),
}

impl Op {
    pub fn is_def(&self) -> bool {
        match self {
            Self::Def(_) => true,
            _ => false,
        }
    }
}

impl Instance for Op {
    type Class = OpType;

    fn class(&self) -> OpType {
        match self {
            Self::Def(op_def) => OpType::Def(op_def.class()),
            Self::Method(method) => OpType::Method(method.class()),
            Self::Ref(op_ref) => OpType::Ref(op_ref.class()),
        }
    }
}

impl ScalarInstance for Op {
    type Class = OpType;
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

impl Serialize for Op {
    fn serialize<S>(&self, _s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        unimplemented!()
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Def(op_def) => write!(f, "{}", op_def),
            Op::Method(method) => write!(f, "{}", method),
            Op::Ref(op_ref) => write!(f, "{}", op_ref),
        }
    }
}
