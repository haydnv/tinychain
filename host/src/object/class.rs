//! User-defined class implementation.

use async_hash::Hash;
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use destream::{de, en};
use safecast::{CastFrom, TryCastFrom};
use sha2::digest::Output;
use sha2::Digest;

use tc_value::{Link, Value};
use tcgeneric::{path_label, Id, Map, NativeClass, PathLabel, TCPathBuf};

use crate::scalar::*;
use crate::state::StateType;

use super::ObjectType;

const PATH: PathLabel = path_label(&["state", "class"]);

/// A user-defined class.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct InstanceClass {
    extends: Option<Link>,
    link: Option<Link>,
    proto: Map<Scalar>,
}

impl InstanceClass {
    /// Construct a new class.
    pub fn new(link: Option<Link>, proto: Map<Scalar>) -> Self {
        Self {
            extends: None,
            link,
            proto,
        }
    }

    /// Extend an existing class.
    pub fn extend(extends: Link, link: Option<Link>, proto: Map<Scalar>) -> Self {
        Self {
            extends: Some(extends),
            link,
            proto,
        }
    }

    /// Construct a new anonymous class.
    pub fn anonymous(extends: Option<Link>, proto: Map<Scalar>) -> Self {
        Self {
            extends,
            link: None,
            proto,
        }
    }

    /// Return the link to this class, if any.
    pub fn link(&self) -> Link {
        if let Some(link) = &self.link {
            link.clone()
        } else {
            TCPathBuf::from(PATH).into()
        }
    }

    /// Consume this class and return its data.
    pub fn into_inner(self) -> (Option<Link>, Map<Scalar>) {
        (self.extends, self.proto)
    }

    /// Return `false` if this class has a self-`Link`
    pub fn is_anonymous(&self) -> bool {
        self.link.is_none()
    }

    /// Return the instance data of this class.
    pub fn proto(&'_ self) -> &'_ Map<Scalar> {
        &self.proto
    }
}

impl<D: Digest> Hash<D> for InstanceClass {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self)
    }
}

impl<'a, D: Digest> Hash<D> for &'a InstanceClass {
    fn hash(self) -> Output<D> {
        if let Some(link) = &self.extends {
            Hash::<D>::hash((link, self.proto.deref()))
        } else {
            Hash::<D>::hash(self.proto.deref())
        }
    }
}

impl tcgeneric::Class for InstanceClass {}

impl tcgeneric::Instance for InstanceClass {
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        ObjectType::Class
    }
}

impl From<StateType> for InstanceClass {
    fn from(st: StateType) -> Self {
        Self::extend(st.path().into(), None, Map::default())
    }
}

impl TryCastFrom<InstanceClass> for StateType {
    fn can_cast_from(class: &InstanceClass) -> bool {
        if class.proto.is_empty() {
            if let Some(classpath) = &class.extends {
                if classpath.host().is_none() {
                    return StateType::from_path(classpath.path()).is_some();
                }
            }
        }

        false
    }

    fn opt_cast_from(class: InstanceClass) -> Option<Self> {
        if class.proto.is_empty() {
            if let Some(classpath) = &class.extends {
                if classpath.host().is_none() {
                    return StateType::from_path(classpath.path());
                }
            }
        }

        None
    }
}

impl CastFrom<InstanceClass> for Scalar {
    fn cast_from(class: InstanceClass) -> Self {
        Self::Map(class.proto)
    }
}

impl CastFrom<InstanceClass> for Value {
    fn cast_from(class: InstanceClass) -> Self {
        class.extends.into()
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

impl fmt::Debug for InstanceClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for InstanceClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(link) = &self.link {
            write!(f, "class {}", link)
        } else if let Some(link) = &self.extends {
            write!(f, "anonymous subclass of {}", link)
        } else {
            f.write_str("Object")
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
                log::debug!("prototype is {}", proto);
                return Ok(InstanceClass::extend(extends, None, proto));
            }

            let mut proto = Map::new();
            let id: Id = key.parse().map_err(de::Error::custom)?;
            proto.insert(id, access.next_value(()).await?);

            while let Some(id) = access.next_key(()).await? {
                let value = access.next_value(()).await?;
                proto.insert(id, value);
            }

            Ok(InstanceClass::anonymous(None, proto.into()))
        } else {
            Ok(InstanceClass::anonymous(None, Map::default()))
        }
    }
}
