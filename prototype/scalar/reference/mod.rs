use std::collections::{HashMap, HashSet};
use std::fmt;

use async_trait::async_trait;
use serde::ser::{Serialize, Serializer};

use crate::class::{Class, Instance, NativeClass, State, TCType};
use crate::error;
use crate::request::Request;
use crate::scalar::{
    label, Id, Link, PathSegment, Scalar, ScalarClass, ScalarInstance, ScalarType, TCPath,
    TCPathBuf, TryCastFrom, TryCastInto,
};
use crate::transaction::Txn;
use crate::TCResult;

pub mod flow;
pub mod id;
pub mod op;

pub use flow::*;
pub use id::*;
pub use op::*;

#[async_trait]
pub trait Refer {
    fn requires(&self, deps: &mut HashSet<Id>);

    async fn resolve(
        self,
        request: &Request,
        txn: &Txn,
        context: &HashMap<Id, State>,
    ) -> TCResult<State>;
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RefType {
    Flow(FlowControlType),
    Id,
    Method(MethodType),
    Op(OpRefType),
}

impl Class for RefType {
    type Instance = Scalar;
}

impl NativeClass for RefType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::method_not_allowed(TCPath::from(path)))
        } else if suffix.len() == 1 && &suffix[0] == "id" {
            Ok(RefType::Id)
        } else {
            match suffix[0].as_str() {
                "flow" => FlowControlType::from_path(path).map(RefType::Flow),
                "method" => MethodType::from_path(path).map(RefType::Method),
                "op" => OpRefType::from_path(path).map(RefType::Op),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPathBuf {
        ScalarType::prefix().append(label("ref"))
    }
}

impl ScalarClass for RefType {
    type Instance = TCRef;

    fn try_cast<S>(&self, scalar: S) -> TCResult<TCRef>
    where
        Scalar: From<S>,
    {
        match self {
            Self::Flow(ft) => ft.try_cast(scalar).map(Box::new).map(TCRef::Flow),
            Self::Id => Scalar::from(scalar)
                .try_cast_into(|v| error::bad_request("Cannot cast into Ref from", v))
                .map(TCRef::Id),
            Self::Method(mt) => mt.try_cast(scalar).map(TCRef::Method),
            Self::Op(ort) => ort.try_cast(scalar).map(TCRef::Op),
        }
    }
}

impl From<RefType> for Link {
    fn from(rt: RefType) -> Link {
        use RefType as RT;

        match rt {
            RT::Flow(ft) => ft.into(),
            RT::Id => RefType::prefix().append(label("id")).into(),
            RT::Method(mt) => mt.into(),
            RT::Op(ort) => ort.into(),
        }
    }
}

impl From<RefType> for TCType {
    fn from(rt: RefType) -> TCType {
        TCType::Scalar(ScalarType::Ref(rt))
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Flow(ft) => write!(f, "{}", ft),
            Self::Id => write!(f, "type Ref"),
            Self::Method(mt) => write!(f, "{}", mt),
            Self::Op(ort) => write!(f, "{}", ort),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCRef {
    Flow(Box<FlowControl>),
    Id(IdRef),
    Method(Method),
    Op(OpRef),
}

impl Instance for TCRef {
    type Class = RefType;

    fn class(&self) -> RefType {
        match self {
            TCRef::Flow(control) => RefType::Flow(control.class()),
            TCRef::Id(_) => RefType::Id,
            TCRef::Method(method) => RefType::Method(method.class()),
            TCRef::Op(op_ref) => RefType::Op(op_ref.class()),
        }
    }
}

impl ScalarInstance for TCRef {
    type Class = RefType;
}

#[async_trait]
impl Refer for TCRef {
    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            TCRef::Flow(control) => {
                control.requires(deps);
            }
            TCRef::Id(id_ref) => {
                deps.insert(id_ref.id().clone());
            }
            TCRef::Method(method) => {
                method.requires(deps);
            }
            TCRef::Op(op_ref) => {
                op_ref.requires(deps);
            }
        }
    }

    async fn resolve(
        self,
        request: &Request,
        txn: &Txn,
        context: &HashMap<Id, State>,
    ) -> TCResult<State> {
        match self {
            TCRef::Flow(control) => control.resolve(request, txn, context).await,
            TCRef::Id(id_ref) => id_ref.resolve(request, txn, context).await,
            TCRef::Method(method) => method.resolve(request, txn, context).await,
            TCRef::Op(op_ref) => op_ref.resolve(request, txn, context).await,
        }
    }
}

impl From<IdRef> for TCRef {
    fn from(id_ref: IdRef) -> TCRef {
        TCRef::Id(id_ref)
    }
}

impl From<OpRef> for TCRef {
    fn from(op_ref: OpRef) -> TCRef {
        TCRef::Op(op_ref)
    }
}

impl TryCastFrom<Scalar> for TCRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        if let Scalar::Ref(_) = scalar {
            true
        } else {
            false
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<TCRef> {
        if let Scalar::Ref(tc_ref) = scalar {
            Some(*tc_ref)
        } else {
            None
        }
    }
}

impl Serialize for TCRef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Flow(control) => control.serialize(s),
            Self::Id(id_ref) => id_ref.serialize(s),
            Self::Method(method) => method.serialize(s),
            Self::Op(op_ref) => op_ref.serialize(s),
        }
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Flow(control) => write!(f, "{}", control),
            Self::Id(id_ref) => write!(f, "{}", id_ref),
            Self::Method(method) => write!(f, "{}", method),
            Self::Op(op_ref) => write!(f, "{}", op_ref),
        }
    }
}
