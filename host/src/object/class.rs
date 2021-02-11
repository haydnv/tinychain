//! User-defined class implementation.

use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};

use generic::{path_label, Id, Map, PathLabel, TCPathBuf};

use crate::scalar::*;
use crate::state::State;

use super::{InstanceExt, ObjectType};

const PATH: PathLabel = path_label(&["state", "class"]);

/// A user-defined class.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct InstanceClass {
    extends: Option<Link>,
    proto: Map<Scalar>,
}

impl InstanceClass {
    /// Construct a new subclass of the class at `extends` with the given instance data.
    pub fn new(extends: Option<Link>, proto: Map<Scalar>) -> Self {
        Self { extends, proto }
    }

    /// Return the parent class of this class.
    pub fn extends(&self) -> Link {
        if let Some(link) = &self.extends {
            link.clone()
        } else {
            TCPathBuf::from(PATH).into()
        }
    }

    /// Consume this class and return its data.
    pub fn into_inner(self) -> (Option<Link>, Map<Scalar>) {
        (self.extends, self.proto)
    }

    /// Return the instance data of this class.
    pub fn proto(&'_ self) -> &'_ Map<Scalar> {
        &self.proto
    }
}

impl generic::Class for InstanceClass {
    type Instance = InstanceExt<State>;
}

impl generic::Instance for InstanceClass {
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        ObjectType::Class
    }
}

#[async_trait]
impl de::FromStream for InstanceClass {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(InstanceClassVisitor).await
    }
}

impl<'en> en::IntoStream<'en> for InstanceClass {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        if let Some(class) = self.extends {
            use en::EncodeMap;

            let mut map = encoder.encode_map(Some(1))?;
            map.encode_entry(class.to_string(), self.proto.into_inner())?;
            map.end()
        } else {
            self.proto.into_inner().into_stream(encoder)
        }
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
                log::debug!("Class extends {}", extends);
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
