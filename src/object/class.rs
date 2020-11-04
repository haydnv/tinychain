use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::sync::Arc;

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::error::{self, TCResult};
use crate::request::Request;
use crate::scalar::{self, label, Link, Scalar, TCPath, TryCastInto, Value, ValueId};
use crate::transaction::Txn;

use super::{ObjectInstance, ObjectType};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct InstanceClassType;

impl InstanceClassType {
    pub fn post(path: TCPath, data: scalar::Object) -> TCResult<InstanceClass> {
        println!("InstanceClassType::post {}", path);

        if path == Self::prefix() {
            let mut data: HashMap<ValueId, Scalar> = data.into();

            let extends = if let Some(extends) = data.remove(&label("extends").into()) {
                let link = extends.try_cast_into(|v| {
                    error::bad_request("'extends' must be a Link to a Class, not", v)
                })?;

                Some(link)
            } else {
                None
            };

            let proto: scalar::Object = data
                .remove(&label("proto").into())
                .unwrap_or_else(|| scalar::Object::default().into())
                .try_into()?;

            if data.is_empty() {
                Ok(InstanceClass { extends, proto })
            } else {
                Err(error::bad_request(
                    format!("{} got unrecognized parameters", Self::prefix()),
                    Value::from(data.keys().cloned().collect::<Vec<ValueId>>()),
                ))
            }
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
    proto: scalar::Object,
}

impl InstanceClass {
    pub fn extends(&self) -> Link {
        if let Some(link) = &self.extends {
            link.clone()
        } else {
            Self::prefix().into()
        }
    }

    pub fn proto(&'_ self) -> &'_ scalar::Object {
        &self.proto
    }

    pub async fn get(
        self,
        request: &Request,
        txn: Arc<Txn>,
        path: TCPath,
        key: Value,
    ) -> TCResult<ObjectInstance> {
        if path.is_empty() {
            ObjectInstance::new(request, txn, self, key).await
        } else {
            Err(error::not_found(path))
        }
    }

    pub fn post(path: TCPath, _data: scalar::Object) -> TCResult<ObjectInstance> {
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
