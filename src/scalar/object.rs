use std::collections::HashMap;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use serde::ser::{Serialize, Serializer};

use crate::class::{Class, Instance, NativeClass, State, TCBoxTryFuture, TCResult};
use crate::error;
use crate::handler::Public;
use crate::request::Request;
use crate::transaction::Txn;

use super::{
    label, Id, Link, PathSegment, Scalar, ScalarInstance, ScalarType, TCPathBuf, TryCastFrom,
    TryCastInto, Value,
};

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct ObjectType;

impl Class for ObjectType {
    type Instance = Object;
}

impl NativeClass for ObjectType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        if path == Self::prefix().as_slice() {
            Ok(Self)
        } else {
            Err(error::path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        ScalarType::prefix().append(label("object"))
    }
}

impl From<ObjectType> for Link {
    fn from(_ot: ObjectType) -> Link {
        ObjectType::prefix().into()
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "generic Object type")
    }
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct Object(HashMap<Id, Scalar>);

impl Object {
    pub fn get<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        path: &'a [PathSegment],
        key: Value,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if path.is_empty() {
                return Ok(State::Scalar(Scalar::Object(self.clone())));
            }

            let scalar = self
                .0
                .get(&path[0])
                .ok_or_else(|| error::not_found(&path[0]))?;

            match scalar {
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .get(request, txn, key, Some(self.clone().into()))
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(path)),
                other => other.get(request, txn, &path[1..], key).await,
            }
        })
    }

    pub fn put<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        path: &'a [PathSegment],
        key: Value,
        value: State,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            if path.is_empty() {
                return Err(error::method_not_allowed(self));
            }

            let scalar = self
                .0
                .get(&path[0])
                .ok_or_else(|| error::not_found(&path[0]))?;

            match scalar {
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .put(request, txn, key, value, Some(self.clone().into()))
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(path)),
                other => other.put(request, txn, &path[1..], key, value).await,
            }
        })
    }

    pub fn post<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        path: &'a [PathSegment],
        params: Object,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            if path.is_empty() {
                return Err(error::method_not_allowed(self));
            }

            let scalar = self
                .0
                .get(&path[0])
                .ok_or_else(|| error::not_found(&path[0]))?;

            match scalar {
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .post(request, txn, params, Some(self.clone().into()))
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(path)),
                other => other.post(request, txn, &path[1..], params).await,
            }
        })
    }

    pub fn delete<'a>(
        &'a self,
        _request: &'a Request,
        _txn: &'a Txn,
        _path: &'a [PathSegment],
        _key: Value,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { Err(error::not_implemented("Object::delete")) })
    }
}

impl Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> Self::Class {
        ObjectType
    }
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
        Object::get(self, request, txn, path, key).await
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        Object::put(self, request, txn, path, key, value).await
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        Object::post(self, request, txn, path, params).await
    }

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<()> {
        Object::delete(self, request, txn, path, key).await
    }
}

impl Deref for Object {
    type Target = HashMap<Id, Scalar>;

    fn deref(&'_ self) -> &'_ HashMap<Id, Scalar> {
        &self.0
    }
}

impl DerefMut for Object {
    fn deref_mut(&'_ mut self) -> &'_ mut HashMap<Id, Scalar> {
        &mut self.0
    }
}

impl<T: Into<Scalar>> FromIterator<(Id, T)> for Object {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> Self {
        let mut object = HashMap::new();

        for (id, attr) in iter {
            let scalar = attr.into();
            object.insert(id, scalar);
        }

        Object(object)
    }
}

impl From<HashMap<Id, Scalar>> for Object {
    fn from(map: HashMap<Id, Scalar>) -> Object {
        Object(map)
    }
}

impl TryCastFrom<Scalar> for Object {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Object(_) => true,
            other => other.matches::<Vec<(Id, Scalar)>>(),
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Object> {
        match scalar {
            Scalar::Object(object) => Some(object),
            other if other.matches::<Vec<(Id, Scalar)>>() => {
                let data: Vec<(Id, Scalar)> = other.opt_cast_into().unwrap();
                Some(Object::from_iter(data))
            }
            _ => None,
        }
    }
}

impl TryCastFrom<State> for Object {
    fn can_cast_from(state: &State) -> bool {
        if let State::Scalar(scalar) = state {
            Object::can_cast_from(scalar)
        } else {
            false
        }
    }

    fn opt_cast_from(state: State) -> Option<Object> {
        if let State::Scalar(scalar) = state {
            Object::opt_cast_from(scalar)
        } else {
            None
        }
    }
}

impl From<Object> for HashMap<Id, Scalar> {
    fn from(object: Object) -> HashMap<Id, Scalar> {
        object.0
    }
}

impl IntoIterator for Object {
    type Item = (Id, Scalar);
    type IntoIter = std::collections::hash_map::IntoIter<Id, Scalar>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl Serialize for Object {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.0
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
