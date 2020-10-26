use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::auth::Auth;
use crate::class::{Class, Instance, NativeClass, TCType};
use crate::error::{self, TCResult};
use crate::scalar::{self, label, Link, Scalar, TCPath, ValueId};
use crate::transaction::Txn;

use super::{ObjectInstance, ObjectType};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct InstanceClassType;

impl InstanceClassType {
    pub fn post(
        _txn: Arc<Txn>,
        path: TCPath,
        _data: scalar::Object,
        _auth: Auth,
    ) -> TCResult<InstanceClass> {
        println!("InstanceClassType::post {}", path);

        if path == Self::prefix() {
            Err(error::not_implemented("InstanceClassType::post"))
        } else {
            Err(error::not_found(path))
        }
    }
}

impl Class for InstanceClassType {
    type Instance = InstanceClass;
}

impl NativeClass for InstanceClassType {
    fn from_path(path: &TCPath) -> TCResult<Self> {
        if path == &Self::prefix() {
            Ok(Self)
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        ObjectType::prefix().join(label("class").into())
    }
}

impl From<InstanceClassType> for Link {
    fn from(_: InstanceClassType) -> Link {
        InstanceClassType::prefix().into()
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
    pub fn post(
        _txn: Arc<Txn>,
        path: TCPath,
        _data: scalar::Object,
        _auth: Auth,
    ) -> TCResult<ObjectInstance> {
        println!("InstanceClass::post {}", path);

        if path.is_empty() {
            Err(error::not_implemented("InstanceClass::post"))
        } else {
            Err(error::not_found(path))
        }
    }

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
