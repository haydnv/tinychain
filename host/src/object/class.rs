use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use destream::de;

use generic::{path_label, Id, Map, PathLabel, PathSegment, TCPathBuf};

use crate::scalar::*;
use crate::state::State;

use super::InstanceExt;

const PATH: PathLabel = path_label(&["state", "class"]);

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct InstanceClassType;

impl generic::Class for InstanceClassType {
    type Instance = InstanceClass;
}

impl generic::NativeClass for InstanceClassType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path == &PATH[..] {
            Some(Self)
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        PATH.into()
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
    pub fn extends(&self) -> Link {
        if let Some(link) = &self.extends {
            link.clone()
        } else {
            TCPathBuf::from(PATH).into()
        }
    }

    pub fn proto(&'_ self) -> &'_ Map<Scalar> {
        &self.proto
    }
}

impl generic::Class for InstanceClass {
    type Instance = InstanceExt<State>;
}

impl generic::Instance for InstanceClass {
    type Class = InstanceClassType;

    fn class(&self) -> InstanceClassType {
        InstanceClassType
    }
}

#[async_trait]
impl de::FromStream for InstanceClass {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(InstanceClassVisitor).await
    }
}

impl From<InstanceClass> for Link {
    fn from(ic: InstanceClass) -> Link {
        if let Some(link) = ic.extends {
            link
        } else {
            TCPathBuf::from(PATH).into()
        }
    }
}

impl fmt::Display for InstanceClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(link) = &self.extends {
            write!(f, "class {}", link)
        } else {
            f.write_str("generic Object type")
        }
    }
}

struct InstanceClassVisitor;

#[async_trait]
impl de::Visitor for InstanceClassVisitor {
    type Value = InstanceClass;

    fn expecting() -> &'static str {
        "a user-defined Class"
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<InstanceClass, A::Error> {
        if let Some(key) = access.next_key::<String>(()).await? {
            if let Ok(extends) = key.parse() {
                let proto = access.next_value(()).await?;
                return Ok(InstanceClass {
                    extends: Some(extends),
                    proto,
                });
            }

            let mut proto = if let Some(len) = access.size_hint() {
                HashMap::with_capacity(len)
            } else {
                HashMap::new()
            };

            let id: Id = key.parse().map_err(de::Error::custom)?;
            proto.insert(id, access.next_value(()).await?);

            while let Some(id) = access.next_key(()).await? {
                let value = access.next_value(()).await?;
                proto.insert(id, value);
            }

            Ok(InstanceClass {
                extends: None,
                proto: proto.into(),
            })
        } else {
            Ok(InstanceClass {
                extends: None,
                proto: Map::default(),
            })
        }
    }
}
