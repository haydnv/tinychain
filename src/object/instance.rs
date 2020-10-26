use std::fmt;
use std::sync::Arc;

use crate::auth::Auth;
use crate::class::{Instance, State, TCBoxTryFuture};
use crate::error::{self, TCResult};
use crate::scalar::{self, TCPath, Value};
use crate::transaction::Txn;

use super::InstanceClass;

#[derive(Clone, Eq, PartialEq)]
pub struct ObjectInstance {
    class: InstanceClass,
    data: scalar::object::Object,
}

impl ObjectInstance {
    pub fn get<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        key: Value,
        auth: Auth,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            match self.data.get(txn, path, key, auth).await {
                Ok(state) => Ok(state),
                Err(not_found) if not_found.reason() == &error::Code::NotFound => {
                    Err(error::not_implemented("Class method resolution"))
                }
                Err(cause) => Err(cause),
            }
        })
    }

    pub async fn post(
        &self,
        _txn: Arc<Txn>,
        path: TCPath,
        _data: scalar::Object,
        _auth: Auth,
    ) -> TCResult<State> {
        if path.is_empty() {
            Err(error::not_implemented("ObjectInstance::post"))
        } else {
            Err(error::not_found(path))
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
            class: InstanceClass::default(),
            data: generic,
        }
    }
}

impl fmt::Display for ObjectInstance {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Object of type {}", self.class())
    }
}
