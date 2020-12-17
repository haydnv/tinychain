use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;

use async_trait::async_trait;
use log::debug;

use crate::class::{Class, Instance, NativeClass, State, TCType};
use crate::error;
use crate::general::{Map, TCResult};
use crate::handler::*;
use crate::request::Request;
use crate::scalar::{
    label, Id, Key, Link, MethodType, OpRef, PathSegment, Scalar, TCPath, TCPathBuf, TryCastInto,
    Value,
};
use crate::transaction::Txn;

use super::{InstanceExt, ObjectType};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct InstanceClassType;

impl InstanceClassType {
    pub fn post(path: &[PathSegment], mut data: Map<Scalar>) -> TCResult<InstanceClass> {
        debug!("InstanceClassType::post {}", TCPath::from(path));

        if path == &Self::prefix()[..] {
            let extends = if let Some(extends) = data.remove(&label("extends").into()) {
                let link = extends.try_cast_into(|v| {
                    error::bad_request("'extends' must be a Link to a Class, not", v)
                })?;

                Some(link)
            } else {
                None
            };

            let proto = data
                .remove(&label("proto").into())
                .unwrap_or_else(|| Scalar::Map(Map::<Scalar>::default()))
                .try_into()?;

            if data.is_empty() {
                Ok(InstanceClass { extends, proto })
            } else {
                Err(error::bad_request(
                    format!("{} got unrecognized parameters", Self::prefix()),
                    Value::from(data.keys().cloned().collect::<Vec<Id>>()),
                ))
            }
        } else {
            Err(error::path_not_found(path))
        }
    }
}

impl Class for InstanceClassType {
    type Instance = InstanceClass;
}

impl NativeClass for InstanceClassType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        if path == &Self::prefix()[..] {
            Ok(Self)
        } else {
            Err(error::path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        ObjectType::prefix().append(label("class"))
    }
}

impl From<InstanceClassType> for Link {
    fn from(_: InstanceClassType) -> Link {
        InstanceClassType::prefix().into()
    }
}

impl From<InstanceClassType> for TCType {
    fn from(ict: InstanceClassType) -> TCType {
        TCType::Object(ObjectType::Class(ict))
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
    proto: Map<Scalar>,
}

impl InstanceClass {
    pub fn from_class<C: Class>(class: C) -> InstanceClass {
        let extends = Some(class.into());
        let proto = Map::<Scalar>::default();
        Self { extends, proto }
    }

    pub fn extends(&self) -> Link {
        if let Some(link) = &self.extends {
            link.clone()
        } else {
            Self::prefix().into()
        }
    }

    pub fn proto(&'_ self) -> &'_ Map<Scalar> {
        &self.proto
    }

    pub fn prefix() -> TCPathBuf {
        TCType::prefix().append(label("object"))
    }
}

impl Class for InstanceClass {
    type Instance = InstanceExt<State>;
}

impl Instance for InstanceClass {
    type Class = InstanceClassType;

    fn class(&self) -> InstanceClassType {
        InstanceClassType
    }
}

impl Route for InstanceClass {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        if path.is_empty() && method == MethodType::Get {
            Some(Box::new(Initializer { class: self }))
        } else {
            None
        }
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

pub struct Initializer<'a> {
    class: &'a InstanceClass,
}

#[async_trait]
impl<'a> Handler for Initializer<'a> {
    fn subject(&self) -> TCType {
        self.class.class().into()
    }

    async fn get(&self, request: &Request, txn: &Txn, schema: Value) -> TCResult<State> {
        let ctr = OpRef::Get((self.class.extends(), Key::Value(schema)));
        let parent = txn.resolve_op(request, &HashMap::new(), ctr.into()).await?;
        Ok(InstanceExt::new(parent, self.class.clone()).into())
    }
}
