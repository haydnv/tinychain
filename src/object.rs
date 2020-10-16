use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::auth::Auth;
use crate::class::{Class, Instance, NativeClass, State, TCBoxTryFuture, TCType};
use crate::error::{self, TCResult};
use crate::scalar::{label, Link, Op, Scalar, TCPath, TryCastInto, Value, ValueId, ValueInstance};
use crate::transaction::Txn;

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct ObjectClassType;

impl Class for ObjectClassType {
    type Instance = ObjectType;
}

impl From<ObjectClassType> for Link {
    fn from(_: ObjectClassType) -> Link {
        TCType::prefix().join(label("class").into()).into()
    }
}

impl fmt::Display for ObjectClassType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "user-defined Class")
    }
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct ObjectType {
    extends: Option<Link>,
    proto: HashMap<ValueId, Scalar>,
}

impl ObjectType {
    pub fn prefix() -> TCPath {
        TCType::prefix().join(label("object").into())
    }

    pub fn post(
        path: TCPath,
        mut params: HashMap<ValueId, Scalar>,
        _auth: Auth,
    ) -> TCResult<State> {
        if path.is_empty() {
            let extends = match params.remove(&label("extends").into()) {
                Some(extends) => {
                    let extends = Value::try_from(extends)?;
                    Some(extends.try_cast_into(|v| {
                        error::bad_request("Expected a Link to a Class, found", v)
                    })?)
                }
                None => None,
            };

            let data = params
                .remove(&label("data").into())
                .unwrap_or_else(|| Scalar::Map(HashMap::new()));
            let data = data.try_cast_into(|v| {
                error::bad_request("Expected a Map to define the requested Object, found", v)
            })?;

            if params.is_empty() {
                let class = ObjectType {
                    extends,
                    proto: HashMap::new(),
                };
                Ok(State::Object(Object { class, data }))
            } else {
                Err(error::bad_request(
                    "Found unrecognized parameter",
                    params.keys().next().unwrap(),
                ))
            }
        } else {
            Err(error::not_found(path))
        }
    }
}

impl Class for ObjectType {
    type Instance = Object;
}

impl Instance for ObjectType {
    type Class = ObjectClassType;

    fn class(&self) -> ObjectClassType {
        ObjectClassType
    }
}

impl From<ObjectType> for Link {
    fn from(ot: ObjectType) -> Link {
        if let Some(link) = ot.extends {
            link
        } else {
            ObjectType::prefix().into()
        }
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(link) = &self.extends {
            write!(f, "class {}", link)
        } else {
            write!(f, "user-defined Class")
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Object {
    class: ObjectType,
    data: HashMap<ValueId, Scalar>,
}

impl Object {
    pub fn data(&'_ self) -> &'_ HashMap<ValueId, Scalar> {
        &self.data
    }
}

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> Self::Class {
        self.class.clone()
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
                return Ok(State::Object(self.clone()));
            }

            match self.data.get(&path[0]) {
                Some(scalar) => match scalar {
                    Scalar::Op(op) => match &**op {
                        Op::Def(op_def) => op_def.get(txn, key, auth, Some(self.clone())).await,
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

impl Serialize for Object {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = s.serialize_map(Some(1))?;
        map.serialize_entry(&Link::from(self.class()).to_string(), &self.data)?;
        map.end()
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
