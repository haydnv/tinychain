use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::auth::{Scope, SCOPE_EXECUTE};
use crate::class::{Class, Instance, NativeClass, State, TCResult, TCType};
use crate::error;
use crate::handler::Handler;
use crate::request::Request;
use crate::transaction::Txn;

use super::link::{Link, TCPathBuf};
use super::map::Map;
use super::{
    label, Id, PathSegment, Scalar, ScalarClass, ScalarInstance, ScalarType, TryCastFrom,
    TryCastInto, Value,
};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum OpDefType {
    Get,
    Put,
    Post,
    Delete,
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
                "delete" => Ok(OpDefType::Delete),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        ScalarType::prefix().append(label("op"))
    }
}

impl ScalarClass for OpDefType {
    type Instance = OpDef;

    fn try_cast<S>(&self, scalar: S) -> TCResult<OpDef>
    where
        Scalar: From<S>,
    {
        let scalar = Scalar::from(scalar);

        match self {
            Self::Get => {
                if scalar.matches::<GetOp>() {
                    Ok(OpDef::Get(scalar.opt_cast_into().unwrap()))
                } else if scalar.matches::<Vec<(Id, Scalar)>>() {
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
            Self::Delete => scalar
                .try_cast_into(|v| error::bad_request("Invalid DELETE definition", v))
                .map(OpDef::Delete),
        }
    }
}

impl From<OpDefType> for Link {
    fn from(odt: OpDefType) -> Link {
        let suffix = match odt {
            OpDefType::Get => label("get"),
            OpDefType::Put => label("put"),
            OpDefType::Post => label("post"),
            OpDefType::Delete => label("delete"),
        };

        OpDefType::prefix().append(suffix).into()
    }
}

impl From<OpDefType> for TCType {
    fn from(odt: OpDefType) -> TCType {
        ScalarType::Op(odt).into()
    }
}

impl fmt::Display for OpDefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET Op definition"),
            Self::Put => write!(f, "type: PUT Op definition"),
            Self::Post => write!(f, "type: POST Op definition"),
            Self::Delete => write!(f, "type: DELETE Op definition"),
        }
    }
}

pub type GetOp = (Id, Vec<(Id, Scalar)>);
pub type PutOp = (Id, Id, Vec<(Id, Scalar)>);
pub type PostOp = Vec<(Id, Scalar)>;
pub type DeleteOp = (Id, Vec<(Id, Scalar)>);

#[derive(Clone, Eq, PartialEq)]
pub enum OpDef {
    Get(GetOp),
    Put(PutOp),
    Post(PostOp),
    Delete(DeleteOp),
}

impl OpDef {
    pub fn handler<'a>(&'a self, context: Option<State>) -> Box<dyn Handler + 'a> {
        match self {
            Self::Get(op) => Box::new(GetHandler { op, context }),
            Self::Put(op) => Box::new(PutHandler { op, context }),
            Self::Post(op) => Box::new(PostHandler { op, context }),
            Self::Delete(op) => Box::new(DeleteHandler { op, context }),
        }
    }
}

impl Instance for OpDef {
    type Class = OpDefType;

    fn class(&self) -> OpDefType {
        match self {
            Self::Get(_) => OpDefType::Get,
            Self::Put(_) => OpDefType::Put,
            Self::Post(_) => OpDefType::Post,
            Self::Delete(_) => OpDefType::Delete,
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

impl Serialize for OpDef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let class = Link::from(self.class()).to_string();
        let mut map = s.serialize_map(Some(1))?;

        match self {
            Self::Get(def) => map.serialize_entry(&class, def),
            Self::Put(def) => map.serialize_entry(&class, def),
            Self::Post(def) => map.serialize_entry(&class, def),
            Self::Delete(def) => map.serialize_entry(&class, def),
        }?;

        map.end()
    }
}

impl fmt::Display for OpDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get(_) => write!(f, "GET Op"),
            Self::Put(_) => write!(f, "PUT Op"),
            Self::Post(_) => write!(f, "POST Op"),
            Self::Delete(_) => write!(f, "DELETE Op"),
        }
    }
}

struct GetHandler<'a> {
    op: &'a GetOp,
    context: Option<State>,
}

#[async_trait]
impl<'a> Handler for GetHandler<'a> {
    fn subject(&self) -> TCType {
        if let Some(context) = &self.context {
            context.class()
        } else {
            OpDefType::Get.into()
        }
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_EXECUTE.into())
    }

    async fn get(&self, request: &Request, txn: &Txn, key: Value) -> TCResult<State> {
        self.authorize(request)?;

        let (key_id, op) = self.op.clone();
        let mut graph = HashMap::new();
        graph.insert(key_id, State::from(key));
        if let Some(context) = &self.context {
            graph.insert(label("self").into(), context.clone());
        }

        txn.execute(request, graph, op.to_vec()).await
    }
}

struct PutHandler<'a> {
    op: &'a PutOp,
    context: Option<State>,
}

#[async_trait]
impl<'a> Handler for PutHandler<'a> {
    fn subject(&self) -> TCType {
        if let Some(context) = &self.context {
            context.class()
        } else {
            OpDefType::Get.into()
        }
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_EXECUTE.into())
    }

    async fn handle_put(
        &self,
        _request: &Request,
        _txn: &Txn,
        _key: Value,
        _value: State,
    ) -> TCResult<()> {
        Err(error::not_implemented("OpDef::put"))
    }
}

struct PostHandler<'a> {
    op: &'a PostOp,
    context: Option<State>,
}

#[async_trait]
impl<'a> Handler for PostHandler<'a> {
    fn subject(&self) -> TCType {
        if let Some(context) = &self.context {
            context.class()
        } else {
            OpDefType::Get.into()
        }
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_EXECUTE.into())
    }

    async fn handle_post(&self, request: &Request, txn: &Txn, params: Map) -> TCResult<State> {
        let graph = params
            .into_inner()
            .into_iter()
            .map(|(id, scalar)| (id, State::from(scalar)))
            .collect();

        txn.execute(request, graph, self.op.to_vec()).await
    }
}

struct DeleteHandler<'a> {
    op: &'a DeleteOp,
    context: Option<State>,
}

#[async_trait]
impl<'a> Handler for DeleteHandler<'a> {
    fn subject(&self) -> TCType {
        if let Some(context) = &self.context {
            context.class()
        } else {
            OpDefType::Get.into()
        }
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_EXECUTE.into())
    }

    async fn handle_delete(&self, _txn: &Txn, _key: Value) -> TCResult<()> {
        Err(error::not_implemented("OpDef::delete"))
    }
}
