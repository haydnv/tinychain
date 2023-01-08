//! User-defined object-orientation features.

use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use safecast::{TryCastFrom, TryCastInto};
use sha2::digest::Output;
use sha2::Sha256;

use tc_error::*;
use tc_transact::{AsyncHash, IntoView};
use tc_value::Value;
use tcgeneric::{label, path_label, NativeClass, PathLabel, PathSegment, TCPathBuf};

use crate::fs::Dir;
use crate::scalar::Scalar;
use crate::state::State;
use crate::txn::Txn;

pub use class::*;
pub use instance::*;

mod class;
mod instance;

const PREFIX: PathLabel = path_label(&["state", "object"]);

/// The type of a user-defined [`Object`].
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum ObjectType {
    Class,
    Instance,
}

impl tcgeneric::Class for ObjectType {}

impl NativeClass for ObjectType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[..2] == &PREFIX[..] {
            match path[2].as_str() {
                "class" => Some(Self::Class),
                "instance" => Some(Self::Instance),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::Class => "class",
            Self::Instance => "instance",
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
    }
}

impl fmt::Display for ObjectType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class => f.write_str("user-defined Class"),
            Self::Instance => f.write_str("user-defined Instance"),
        }
    }
}

/// A user-defined [`InstanceClass`] or [`InstanceExt`].
#[derive(Clone)]
pub enum Object {
    Class(InstanceClass),
    Instance(InstanceExt<State>),
}

impl tcgeneric::Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        match self {
            Self::Class(_) => ObjectType::Class,
            Self::Instance(_) => ObjectType::Instance,
        }
    }
}

#[async_trait]
impl AsyncHash<Dir> for Object {
    type Txn = Txn;

    async fn hash(self, _txn: &Self::Txn) -> TCResult<Output<Sha256>> {
        match self {
            Self::Class(class) => Ok(async_hash::Hash::<Sha256>::hash(class)),
            Self::Instance(instance) => Err(TCError::bad_request(
                "cannot hash a user-defined instance",
                instance,
            )),
        }
    }
}

impl From<InstanceClass> for Object {
    fn from(class: InstanceClass) -> Object {
        Object::Class(class)
    }
}

impl From<InstanceExt<State>> for Object {
    fn from(instance: InstanceExt<State>) -> Object {
        Object::Instance(instance)
    }
}

impl TryCastFrom<Object> for Scalar {
    fn can_cast_from(object: &Object) -> bool {
        match object {
            Object::Class(class) => Self::can_cast_from(class),
            Object::Instance(instance) => Self::can_cast_from(instance),
        }
    }

    fn opt_cast_from(object: Object) -> Option<Self> {
        match object {
            Object::Class(class) => Self::opt_cast_from(class),
            Object::Instance(instance) => Self::opt_cast_from(instance),
        }
    }
}

impl TryCastFrom<Object> for Value {
    fn can_cast_from(object: &Object) -> bool {
        match object {
            Object::Class(class) => Self::can_cast_from(class),
            Object::Instance(instance) => Self::can_cast_from(instance),
        }
    }

    fn opt_cast_from(object: Object) -> Option<Self> {
        match object {
            Object::Class(class) => Self::opt_cast_from(class),
            Object::Instance(instance) => Self::opt_cast_from(instance),
        }
    }
}

#[async_trait]
impl<'en> IntoView<'en, Dir> for Object {
    type Txn = Txn;
    type View = ObjectView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<ObjectView<'en>> {
        match self {
            Self::Class(class) => Ok(ObjectView::Class(class)),
            Self::Instance(instance) => instance.into_view(txn).map_ok(ObjectView::Instance).await,
        }
    }
}

#[async_trait]
impl fmt::Debug for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => fmt::Debug::fmt(ict, f),
            Self::Instance(ic) => fmt::Debug::fmt(ic, f),
        }
    }
}

#[async_trait]
impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => fmt::Display::fmt(ict, f),
            Self::Instance(ic) => fmt::Display::fmt(ic, f),
        }
    }
}

/// A view of an [`Object`] at a specific [`Txn`], used for serialization.
pub enum ObjectView<'en> {
    Class(InstanceClass),
    Instance(InstanceView<'en>),
}

impl<'en> en::IntoStream<'en> for ObjectView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;

        match self {
            Self::Class(class) => map.encode_entry(ObjectType::Class.path().to_string(), class),
            Self::Instance(instance) => {
                map.encode_entry(ObjectType::Instance.path().to_string(), instance)
            }
        }?;

        map.end()
    }
}

/// A helper struct for Object deserialization
pub struct ObjectVisitor;

impl ObjectVisitor {
    pub fn new() -> Self {
        Self
    }

    pub async fn visit_map_value<Err: de::Error>(
        self,
        class: ObjectType,
        state: State,
    ) -> Result<Object, Err> {
        match class {
            ObjectType::Class => {
                let class =
                    state.try_cast_into(|s| de::Error::invalid_value(s, "a Class definition"))?;

                Ok(Object::Class(class))
            }
            ObjectType::Instance => Ok(Object::Instance(InstanceExt::new(
                state,
                InstanceClass::default(),
            ))),
        }
    }
}
