use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::auth::Auth;
use crate::class::{Class, Instance, NativeClass, State, TCBoxTryFuture, TCType};
use crate::error;
use crate::scalar::{self, label, Link, Scalar, TCPath, Value, ValueId};
use crate::transaction::Txn;

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct InstanceClassType;

impl Class for InstanceClassType {
    type Instance = InstanceClass;
}

impl From<InstanceClassType> for Link {
    fn from(_: InstanceClassType) -> Link {
        TCType::prefix().join(label("class").into()).into()
    }
}

impl fmt::Display for InstanceClassType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "user-defined Class")
    }
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct InstanceClass {
    extends: Option<Link>,
    proto: HashMap<ValueId, Scalar>,
}

impl InstanceClass {
    pub fn prefix() -> TCPath {
        TCType::prefix().join(label("object").into())
    }
}

impl Class for InstanceClass {
    type Instance = ObjectInstance;
}

impl Instance for InstanceClass {
    type Class = InstanceClassType;

    fn class(&self) -> InstanceClassType {
        InstanceClassType
    }
}

impl From<InstanceClass> for Link {
    fn from(ic: InstanceClass) -> Link {
        if let Some(link) = ic.extends {
            link
        } else {
            InstanceClass::prefix().into()
        }
    }
}

impl fmt::Display for InstanceClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(link) = &self.extends {
            write!(f, "class {}", link)
        } else {
            write!(f, "generic Object type")
        }
    }
}

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
