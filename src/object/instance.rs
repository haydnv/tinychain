use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use log::debug;

use crate::class::{Instance, State};
use crate::error::{self, TCResult};
use crate::handler::Public;
use crate::request::Request;
use crate::scalar::{self, PathSegment, Scalar, TCPath, Value};
use crate::transaction::Txn;

use super::InstanceClass;

#[derive(Clone)]
pub struct InstanceExt<T: Instance> {
    parent: Box<T>,
    class: InstanceClass,
}

impl<T: Instance> InstanceExt<T> {
    pub fn new(parent: T, class: InstanceClass) -> InstanceExt<T> {
        InstanceExt {
            parent: Box::new(parent),
            class,
        }
    }

    pub fn into_state(self) -> InstanceExt<State>
    where
        State: From<T>,
    {
        let parent = Box::new((*self.parent).into());
        let class = self.class;
        InstanceExt { parent, class }
    }

    pub fn try_as<E, O: Instance + TryFrom<T, Error = E>>(self) -> Result<InstanceExt<O>, E> {
        let class = self.class;
        let parent = (*self.parent).try_into()?;

        Ok(InstanceExt {
            parent: Box::new(parent),
            class,
        })
    }
}

#[async_trait]
impl<T: Instance + Public> Public for InstanceExt<T>
where
    State: From<T>,
{
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
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .route(request, Some(self.clone().into_state().into()))
                        .get(request, txn, key)
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(&path[1..])),
                scalar => scalar.get(request, txn, path, key).await,
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
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .route(request, Some(self.clone().into_state().into()))
                        .put(request, txn, key, value)
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(&path[1..])),
                scalar => scalar.put(request, txn, path, key, value).await,
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
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .route(request, Some(self.clone().into_state().into()))
                        .post(request, txn, params)
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(&path[1..])),
                scalar => scalar.post(request, txn, path, params).await,
            },
            None => self.parent.post(request, txn, path, params).await,
        }
    }

    async fn delete(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _key: Value,
    ) -> TCResult<()> {
        Err(error::not_implemented("InstanceExt::delete"))
    }
}

impl<T: Instance> Instance for InstanceExt<T> {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl<T: Instance> From<T> for InstanceExt<T> {
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

impl<T: Instance> fmt::Display for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
