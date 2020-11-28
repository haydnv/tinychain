use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;

use futures::TryFutureExt;
use log::debug;

use crate::class::{Instance, State, TCBoxTryFuture};
use crate::error::{self, TCResult};
use crate::request::Request;
use crate::scalar::{self, Key, OpRef, PathSegment, Scalar, TCPath, Value, ValueInstance};
use crate::transaction::Txn;

use super::InstanceClass;

#[derive(Clone)]
pub struct ObjectInstance {
    parent: Box<State>,
    class: InstanceClass,
}

impl ObjectInstance {
    pub async fn new(
        request: &Request,
        txn: &Txn,
        class: InstanceClass,
        schema: Value,
    ) -> TCResult<ObjectInstance> {
        let ctr = OpRef::Get((class.extends(), Key::Value(schema)));
        let parent = txn
            .resolve(request, &HashMap::new(), ctr.into())
            .map_ok(Box::new)
            .await?;

        Ok(ObjectInstance { parent, class })
    }

    pub fn get<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        path: &'a [PathSegment],
        key: Value,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            debug!("ObjectInstance::get {}: {}", TCPath::from(path), key);

            let proto = self.class.proto().deref();
            match proto.get(&path[0]) {
                Some(scalar) => match scalar {
                    Scalar::Op(op_def) if path.len() == 1 => {
                        op_def.get(request, txn, key, Some(self)).await
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
                None => match &*self.parent {
                    State::Object(parent) => parent.get(request, txn, path, key).await,
                    State::Scalar(scalar) => match scalar {
                        Scalar::Object(object) => object.get(request, txn, path, key).await,
                        Scalar::Value(value) => {
                            value.get(path, key).map(Scalar::Value).map(State::Scalar)
                        }
                        _ => Err(error::not_implemented(format!(
                            "Class inheritance for Scalar (parent is {})",
                            scalar
                        ))),
                    },
                    _ => Err(error::not_implemented("Class inheritance for State")),
                },
            }
        })
    }

    pub async fn post(
        &self,
        _request: &Request,
        _txn: &Txn,
        path: &[PathSegment],
        _data: scalar::Object,
    ) -> TCResult<State> {
        if path.is_empty() {
            Err(error::not_implemented("ObjectInstance::post"))
        } else {
            Err(error::path_not_found(path))
        }
    }
}

impl Instance for ObjectInstance {
    type Class = InstanceClass;

    fn class(&self) -> Self::Class {
        self.class.clone()
    }
}

impl From<scalar::object::Object> for ObjectInstance {
    fn from(generic: scalar::object::Object) -> ObjectInstance {
        ObjectInstance {
            parent: Box::new(State::Scalar(Scalar::Object(generic))),
            class: InstanceClass::default(),
        }
    }
}

impl fmt::Display for ObjectInstance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
