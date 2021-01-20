use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use destream::en::{EncodeMap, Encoder, ToStream};
use safecast::{Match, TryCastFrom, TryCastInto};

use error::*;
use generic::*;
use value::{Link, Value};

use crate::transaction::Txn;

use super::{Scalar, ScalarType};

const PREFIX: PathLabel = path_label(&["sbin", "op"]);

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
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && path[..2] == &PREFIX[..] {
            match suffix[0].as_str() {
                "get" => Some(OpDefType::Get),
                "put" => Some(OpDefType::Put),
                "post" => Some(OpDefType::Post),
                "delete" => Some(OpDefType::Delete),
                other => None,
            }
        } else {
            None
        }
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

impl<'en> ToStream<'en> for OpDef {
    fn to_stream<E: Encoder<'en>>(&self, e: E) -> Result<E::Ok, E::Error> {
        let class = Link::from(self.class().path()).to_string();
        let mut map = e.encode_map(Some(1))?;

        match self {
            Self::Get(def) => map.encode_entry(&class, def),
            Self::Put(def) => map.encode_entry(&class, def),
            Self::Post(def) => map.encode_entry(&class, def),
            Self::Delete(def) => map.encode_entry(&class, def),
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

    async fn get(self: Box<Self>, request: &Request, txn: &Txn, key: Value) -> TCResult<State> {
        self.authorize(request)?;

        let (key_id, op) = self.op.clone();
        let mut graph = HashMap::new();
        graph.insert(key_id, State::from(key));
        if let Some(context) = self.context {
            graph.insert(label("self").into(), context);
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
            OpDefType::Put.into()
        }
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_EXECUTE.into())
    }

    async fn handle_put(
        self: Box<Self>,
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

    async fn handle_post(
        self: Box<Self>,
        request: &Request,
        txn: &Txn,
        params: Map<Scalar>,
    ) -> TCResult<State> {
        let mut graph: HashMap<Id, State> = params
            .into_inner()
            .into_iter()
            .map(|(id, scalar)| (id, State::from(scalar)))
            .collect();

        if let Some(context) = self.context {
            graph.insert(label("self").into(), context);
        }

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

    async fn handle_delete(self: Box<Self>, _txn: &Txn, _key: Value) -> TCResult<()> {
        Err(error::not_implemented("OpDef::delete"))
    }
}
