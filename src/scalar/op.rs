use std::fmt;

use futures::stream;
use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::class::{Class, Instance, NativeClass, State, TCBoxTryFuture, TCResult, TCType};
use crate::error;
use crate::object::InstanceExt;
use crate::request::Request;
use crate::transaction::Txn;

use super::link::{Link, TCPathBuf};
use super::object::Object;
use super::{
    label, CastFrom, Id, PathSegment, Scalar, ScalarClass, ScalarInstance, ScalarType, TryCastFrom,
    TryCastInto, Value,
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
        }
    }
}

pub type GetOp = (Id, Vec<(Id, Scalar)>);
pub type PutOp = (Id, Id, Vec<(Id, Scalar)>);
pub type PostOp = Vec<(Id, Scalar)>;

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
        context: Option<InstanceExt<State>>,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if let Self::Get((key_id, def)) = self {
                let mut data = Vec::with_capacity(def.len() + 2);

                if let Some(subject) = context {
                    data.push((label("self").into(), State::Object(subject.clone().into())));
                }

                data.push((key_id.clone(), Scalar::Value(key).into()));
                data.extend(def.to_vec().into_iter().map(|(k, v)| (k, State::Scalar(v))));

                txn.execute(request, stream::iter(data.into_iter())).await
            } else {
                Err(error::method_not_allowed(self))
            }
        })
    }

    pub fn put<'a>(
        &'a self,
        _request: &'a Request,
        _txn: &'a Txn,
        _key: Value,
        _value: State,
        _context: Option<InstanceExt<State>>,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { Err(error::not_implemented("OpDef::put")) })
    }

    pub fn post<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        params: Object,
        context: Option<InstanceExt<State>>,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if let Self::Post(def) = self {
                let mut data = Vec::with_capacity(def.len() + params.len() + 1);

                if let Some(subject) = context {
                    data.push((label("self").into(), subject.into()));
                }

                data.extend(
                    params
                        .into_iter()
                        .chain(def.into_iter().cloned())
                        .map(|(id, scalar)| (id, State::from(scalar))),
                );

                txn.execute(request, stream::iter(data.into_iter())).await
            } else {
                Err(error::method_not_allowed(self))
            }
        })
    }

    pub fn delete<'a>(
        &'a self,
        _request: &'a Request,
        _txn: &'a Txn,
        _key: Value,
        _context: Option<InstanceExt<State>>,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { Err(error::not_implemented("OpDef::delete")) })
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

impl Serialize for OpDef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let class = Link::from(self.class()).to_string();
        let mut map = s.serialize_map(Some(1))?;

        match self {
            Self::Get(def) => map.serialize_entry(&class, def),
            Self::Put(def) => map.serialize_entry(&class, def),
            Self::Post(def) => map.serialize_entry(&class, def),
        }?;

        map.end()
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
