use std::fmt;

use async_trait::async_trait;
use futures::TryFutureExt;
use log::debug;

use crate::class::{Class, Instance, NativeClass, Public, State, TCType};
use crate::error::{self, TCResult};
use crate::request::Request;
use crate::scalar::{self, label, Link, PathSegment, TCPath, TCPathBuf, Value};
use crate::transaction::Txn;

mod class;
mod instance;

pub use class::{InstanceClass, InstanceClassType};
pub use instance::ObjectInstance;

#[derive(Clone, Eq, PartialEq)]
pub enum ObjectType {
    Class(InstanceClassType),
    Instance(InstanceClass),
}

impl ObjectType {
    pub fn post(path: &[PathSegment], data: scalar::Object) -> TCResult<Object> {
        debug!("ObjectType::post {} <- {}", TCPath::from(path), data);

        if path == &Self::prefix()[..] {
            InstanceClass::post(path, data).map(Object::Instance)
        } else if path.starts_with(&InstanceClassType::prefix()) {
            InstanceClassType::post(path, data).map(Object::Class)
        } else {
            Err(error::path_not_found(path))
        }
    }
}

impl Class for ObjectType {
    type Instance = Object;
}

impl NativeClass for ObjectType {
    fn from_path(path: &[PathSegment]) -> TCResult<ObjectType> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Ok(ObjectType::Instance(InstanceClass::default()))
        } else if suffix.len() == 1 && &suffix[0] == "class" {
            Ok(ObjectType::Class(InstanceClassType))
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        TCType::prefix().append(label("object"))
    }
}

impl From<ObjectType> for Link {
    fn from(ot: ObjectType) -> Link {
        match ot {
            ObjectType::Class(ict) => ict.into(),
            ObjectType::Instance(ic) => ic.into(),
        }
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => write!(f, "{}", ict),
            Self::Instance(ic) => write!(f, "{}", ic),
        }
    }
}

#[derive(Clone)]
pub enum Object {
    Class(InstanceClass),
    Instance(ObjectInstance),
}

#[async_trait]
impl Public for Object {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State> {
        match self {
            Self::Class(ic) => {
                ic.clone()
                    .get(request, txn, path, key)
                    .map_ok(Object::Instance)
                    .map_ok(State::Object)
                    .await
            }
            Self::Instance(instance) => instance.get(request, txn, path, key).await,
        }
    }

    async fn put(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _key: Value,
        _value: State,
    ) -> TCResult<()> {
        Err(error::not_implemented("Object::put"))
    }

    async  fn post(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _params: scalar::Object,
    ) -> TCResult<State> {
        Err(error::not_implemented("Object::post"))
    }
}

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        match self {
            Self::Class(ic) => ObjectType::Class(ic.class()),
            Self::Instance(i) => ObjectType::Instance(i.class()),
        }
    }
}

impl From<InstanceClass> for Object {
    fn from(ic: InstanceClass) -> Object {
        Object::Class(ic)
    }
}

impl From<ObjectInstance> for Object {
    fn from(oi: ObjectInstance) -> Object {
        Object::Instance(oi)
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => write!(f, "{}", ict),
            Self::Instance(ic) => write!(f, "{}", ic),
        }
    }
}
