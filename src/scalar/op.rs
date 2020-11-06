use std::fmt;

use futures::stream;
use log::debug;

use crate::class::{Class, Instance, NativeClass, State, TCBoxTryFuture, TCResult};
use crate::error;
use crate::object::ObjectInstance;
use crate::request::Request;
use crate::transaction::Txn;

use super::link::{Link, TCPathBuf};
use super::object::Object;
use super::{
    label, CastFrom, PathSegment, Scalar, ScalarClass, ScalarInstance, ScalarType, TCRef,
    TryCastFrom, TryCastInto, Value, ValueId,
};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum OpDefType {
    Get,
    Put,
    Post,
}

impl Class for OpDefType {
    type Instance = OpDef;
}

impl NativeClass for OpDefType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "get" => Ok(OpDefType::Get),
                "put" => Ok(OpDefType::Put),
                "post" => Ok(OpDefType::Post),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        OpType::prefix().append(label("def"))
    }
}

impl ScalarClass for OpDefType {
    type Instance = OpDef;

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<OpDef> {
        let scalar: Scalar = scalar.into();

        match self {
            Self::Get => {
                if scalar.matches::<GetOp>() {
                    Ok(OpDef::Get(scalar.opt_cast_into().unwrap()))
                } else if scalar.matches::<Vec<(ValueId, Scalar)>>() {
                    Ok(OpDef::Get((
                        label("key").into(),
                        scalar.opt_cast_into().unwrap(),
                    )))
                } else {
                    Err(error::bad_request("Invalid GET definition", scalar))
                }
            }
            Self::Put => scalar
                .try_cast_into(|v| error::bad_request("Invalid PUT definition", v))
                .map(OpDef::Put),
            Self::Post => scalar
                .try_cast_into(|v| error::bad_request("Invalid POST definition", v))
                .map(OpDef::Post),
        }
    }
}

impl From<OpDefType> for Link {
    fn from(odt: OpDefType) -> Link {
        let suffix = match odt {
            OpDefType::Get => label("get"),
            OpDefType::Put => label("put"),
            OpDefType::Post => label("post"),
        };

        OpDefType::prefix().append(suffix).into()
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
}

impl NativeClass for MethodType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "get" => Ok(MethodType::Get),
                "put" => Ok(MethodType::Put),
                "post" => Ok(MethodType::Post),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        OpType::prefix().append(label("method"))
    }
}

impl ScalarClass for MethodType {
    type Instance = Method;

    fn try_cast<S: Into<Scalar>>(&self, _scalar: S) -> TCResult<Method> {
        Err(error::not_implemented("Cast Scalar into Method"))
    }
}

impl From<MethodType> for Link {
    fn from(mt: MethodType) -> Link {
        let suffix = match mt {
            MethodType::Get => label("get"),
            MethodType::Put => label("put"),
            MethodType::Post => label("post"),
        };

        MethodType::prefix().append(suffix).into()
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
}

impl NativeClass for OpRefType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "if" => Ok(OpRefType::If),
                "get" => Ok(OpRefType::Get),
                "put" => Ok(OpRefType::Put),
                "post" => Ok(OpRefType::Post),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        OpType::prefix().append(label("ref"))
    }
}

impl ScalarClass for OpRefType {
    type Instance = OpRef;

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<OpRef> {
        let scalar: Scalar = scalar.into();
        scalar.try_cast_into(|v| error::bad_request(format!("Cannot cast into {} from", self), v))
    }
}

impl From<OpRefType> for Link {
    fn from(ort: OpRefType) -> Link {
        use OpRefType as ORT;
        let suffix = match ort {
            ORT::If => label("if"),
            ORT::Get => label("get"),
            ORT::Put => label("put"),
            ORT::Post => label("post"),
        };

        OpRefType::prefix().append(suffix).into()
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
}

impl NativeClass for OpType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

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

    fn prefix() -> TCPathBuf {
        ScalarType::prefix().append(label("op"))
    }
}

impl ScalarClass for OpType {
    type Instance = Op;

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
    pub fn get<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        key: Value,
        context: Option<&'a ObjectInstance>,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if let Self::Get((key_id, def)) = self {
                let mut data = if let Some(subject) = context {
                    debug!("OpDef::get {} (context: {})", subject, key);
                    vec![(label("self").into(), State::Object(subject.clone().into()))]
                } else {
                    debug!("OpDef::get {}", key);
                    vec![]
                };

                data.push((key_id.clone(), Scalar::Value(key).into()));
                data.extend(def.to_vec().into_iter().map(|(k, v)| (k, State::Scalar(v))));
                txn.execute(request, stream::iter(data.drain(..))).await
            } else {
                Err(error::method_not_allowed(self))
            }
        })
    }

    pub fn post<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        data: Object,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if let Self::Post(def) = self {
                let mut op: Vec<(ValueId, Scalar)> = data.into_iter().collect();
                op.extend(def.to_vec());
                txn.execute(request, stream::iter(op.drain(..))).await
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

impl CastFrom<OpDef> for Scalar {
    fn cast_from(def: OpDef) -> Scalar {
        match def {
            OpDef::Get((key_name, def)) => Scalar::Tuple(vec![key_name.into(), def.into()]),
            OpDef::Put((key_name, value_name, def)) => {
                Scalar::Tuple(vec![key_name.into(), value_name.into(), def.into()])
            }
            OpDef::Post(def) => def.into(),
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
    Get((TCRef, TCPathBuf), Value),
    Put((TCRef, TCPathBuf), (Value, Scalar)),
    Post((TCRef, TCPathBuf), Object),
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
            Self::Get((subject, path), _) => write!(f, "GET {}: {}", subject, path),
            Self::Put((subject, path), (_, _)) => write!(f, "PUT {}{}", subject, path),
            Self::Post((subject, path), _) => write!(f, "PUT {}{}", subject, path),
        }
    }
}

type GetRef = (Link, Value);
type PutRef = (Link, Value, Scalar);
type PostRef = (Link, Object);

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
        s.matches::<(TCRef, Scalar, Scalar)>()
            || s.matches::<(Link, Vec<(ValueId, Value)>)>()
            || s.matches::<(Link, Value, Value)>()
            || s.matches::<(Link, Value)>()
    }

    fn opt_cast_from(s: Scalar) -> Option<OpRef> {
        if s.matches::<(TCRef, Scalar, Scalar)>() {
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

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Op::Def(op_def) => write!(f, "{}", op_def),
            Op::Method(method) => write!(f, "{}", method),
            Op::Ref(op_ref) => write!(f, "{}", op_ref),
        }
    }
}
