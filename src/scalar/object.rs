use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

use crate::auth::Auth;
use crate::class::{Class, Instance, State, TCBoxTryFuture, TCType};
use crate::error::{self, TCResult};
use crate::scalar::{label, Link, Op, Scalar, TCPath, Value, ValueId, ValueInstance};
use crate::transaction::Txn;

use super::{ScalarClass, ScalarInstance};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct ObjectType;

impl Class for ObjectType {
    type Instance = Object;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        if path == &Self::prefix() {
            Ok(ObjectType)
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        TCType::prefix().join(label("object").into())
    }
}

impl ScalarClass for ObjectType {
    type Instance = Object;

    fn try_cast<S: Into<Scalar>>(&self, _scalar: S) -> TCResult<Object> {
        Err(error::not_implemented("ObjectType::try_cast"))
    }
}

impl From<ObjectType> for Link {
    fn from(_: ObjectType) -> Link {
        ObjectType::prefix().into()
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "user-defined Class")
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Object {
    data: HashMap<ValueId, Scalar>,
}

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> Self::Class {
        ObjectType
    }
}

impl Object {
    pub fn get<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        key: Value,
        auth: Auth,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if path.is_empty() {
                return Ok(State::Scalar(Scalar::Object(self.clone())));
            }

            match self.data.get(&path[0]) {
                Some(scalar) => match scalar {
                    Scalar::Object(object) => object.get(txn, path.slice_from(1), key, auth).await,
                    Scalar::Op(op) => match &**op {
                        Op::Def(op_def) => op_def.get(txn, key, auth).await,
                        other => Err(error::not_implemented(other)),
                    },
                    Scalar::Value(value) => value
                        .get(path.slice_from(1), key)
                        .map(Scalar::Value)
                        .map(State::Scalar),
                    other if path.len() == 1 => Ok(State::Scalar(other.clone())),
                    _ => Err(error::not_found(path)),
                },
                _ => Err(error::not_found(path)),
            }
        })
    }

    pub fn put<'a>(
        &'a self,
        _txn: Arc<Txn>,
        _path: TCPath,
        _key: Value,
        _value: State,
        _auth: Auth,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move { Err(error::not_implemented("Object::put")) })
    }
}

impl ScalarInstance for Object {
    type Class = ObjectType;
}

impl<'de> Deserialize<'de> for Object {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Deserialize::deserialize(d).map(|data| Object { data })
    }
}

impl Serialize for Object {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.data.serialize(s)
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
