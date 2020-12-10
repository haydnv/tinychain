use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use log::debug;

use crate::class::{Instance, Public, State};
use crate::error::{self, TCResult};
use crate::request::Request;
use crate::scalar::{self, PathSegment, Scalar, TCPath, TryCastInto, Value, ValueInstance};
use crate::transaction::Txn;

use super::InstanceClass;

#[derive(Clone)]
pub struct InstanceExt<T: Clone + Public + Send + Sync> {
    parent: Box<T>,
    class: InstanceClass,
}

impl<T: Clone + Public + Send + Sync> InstanceExt<T> {
    pub fn new(parent: T, class: InstanceClass) -> InstanceExt<T> {
        InstanceExt {
            parent: Box::new(parent),
            class,
        }
    }

    pub fn into_state(self) -> InstanceExt<State>
    where
        T: Into<State>,
    {
        let parent = Box::new((*self.parent).into());
        let class = self.class;
        InstanceExt { parent, class }
    }

    pub fn try_as<E, O: Clone + Public + TryFrom<T, Error = E> + Send + Sync>(
        self,
    ) -> Result<InstanceExt<O>, E> {
        let class = self.class;
        let parent = (*self.parent).try_into()?;
        Ok(InstanceExt {
            parent: Box::new(parent),
            class,
        })
    }
}

#[async_trait]
impl<T: Clone + Public + Into<State> + Send + Sync> Public for InstanceExt<T> {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State> {
        debug!("ObjectInstance::get {}: {}", TCPath::from(path), key);

        let proto = self.class.proto().deref();
        match proto.get(&path[0]) {
            Some(scalar) => match scalar {
                Scalar::Object(object) => object.get(request, txn, &path[1..], key).await,
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .get(request, txn, key, Some(self.clone().into_state().into()))
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(&path[1..])),
                Scalar::Tuple(tuple) => {
                    let i: usize =
                        key.try_cast_into(|v| error::bad_request("Invalid index for tuple", v))?;
                    tuple
                        .get(i)
                        .cloned()
                        .map(State::Scalar)
                        .ok_or_else(|| error::not_found(i))
                }
                Scalar::Value(value) => value
                    .get(&path[1..], key)
                    .map(Scalar::Value)
                    .map(State::Scalar),
                other => Err(error::method_not_allowed(other)),
            },
            None => self.parent.get(request, txn, path, key).await,
        }
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        debug!("ObjectInstance::put {}: {}", TCPath::from(path), key);

        let proto = self.class.proto().deref();
        match proto.get(&path[0]) {
            Some(scalar) => match scalar {
                Scalar::Object(object) => object.put(request, txn, &path[1..], key, value).await,
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .put(
                            request,
                            txn,
                            key,
                            value,
                            Some(self.clone().into_state().into()),
                        )
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(&path[1..])),
                other => Err(error::method_not_allowed(other)),
            },
            None => self.parent.put(request, txn, path, key, value).await,
        }
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: scalar::Object,
    ) -> TCResult<State> {
        debug!("ObjectInstance::post {}", TCPath::from(path));

        let proto = self.class.proto().deref();
        match proto.get(&path[0]) {
            Some(scalar) => match scalar {
                Scalar::Object(object) => object.post(request, txn, &path[1..], params).await,
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .post(request, txn, params, Some(self.clone().into_state().into()))
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(&path[1..])),
                other => Err(error::method_not_allowed(other)),
            },
            None => self.parent.post(request, txn, path, params).await,
        }
    }
}

impl<T: Clone + Public + Send + Sync> Instance for InstanceExt<T> {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl<T: Clone + Instance + Public + Send + Sync> From<T> for InstanceExt<T> {
    fn from(instance: T) -> InstanceExt<T> {
        let class = InstanceClass::from_class(instance.class());

        InstanceExt {
            parent: Box::new(instance),
            class,
        }
    }
}

impl From<scalar::Object> for InstanceExt<State> {
    fn from(scalar: scalar::Object) -> InstanceExt<State> {
        let class = InstanceClass::from_class(scalar.class());

        InstanceExt {
            parent: Box::new(State::Scalar(scalar.into())),
            class,
        }
    }
}

impl<T: Clone + Public + Send + Sync> fmt::Display for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
