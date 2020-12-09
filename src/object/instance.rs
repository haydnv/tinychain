use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use log::debug;

use crate::class::{Instance, Public, State};
use crate::error::{self, TCResult};
use crate::request::Request;
use crate::scalar::{self, PathSegment, Scalar, TCPath, Value, ValueInstance};
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
                Scalar::Op(op_def) if path.len() == 1 => {
                    op_def
                        .get(request, txn, key, Some(self.clone().into_state().into()))
                        .await
                }
                Scalar::Value(value) => value
                    .get(&path[1..], key)
                    .map(Scalar::Value)
                    .map(State::Scalar),
                other => Err(error::not_implemented(format!(
                    "ObjectInstance::get {}",
                    other
                ))),
            },
            None => self.parent.get(request, txn, path, key).await,
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
        Err(error::not_implemented("InstanceExt::put"))
    }

    async fn post(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _params: scalar::Object,
    ) -> TCResult<State> {
        Err(error::not_implemented("InstanceExt::post"))
    }
}

impl<T: Clone + Public + Send + Sync> Instance for InstanceExt<T> {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl From<scalar::Object> for InstanceExt<State> {
    fn from(scalar: scalar::Object) -> InstanceExt<State> {
        InstanceExt {
            parent: Box::new(State::Scalar(scalar.into())),
            class: InstanceClass::default(),
        }
    }
}

impl<T: Clone + Public + Send + Sync> fmt::Display for InstanceExt<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
