//! User-defined object-orientation features.

use std::fmt;

use async_trait::async_trait;
use destream::{de, en, EncodeMap};
use futures::TryFutureExt;

use tc_error::{TCError, TCResult};
use tc_transact::IntoView;
use tcgeneric::{label, path_label, Instance, Map, NativeClass, PathLabel, PathSegment, TCPathBuf};

use crate::fs::Dir;
use crate::scalar::Value;
use crate::state::{State, StateClass};
use crate::txn::Txn;

pub use class::*;
pub use instance::*;
use safecast::TryCastInto;

mod class;
mod instance;

const ERR_DECODE_INSTANCE: &str = "Instance does not support direct decoding; use an OpRef instead";
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

impl StateClass for ObjectType {
    type Get = Object;

    fn try_cast_from_value(self, value: Value) -> TCResult<Self::Get> {
        let object = match self {
            Self::Class => {
                let classpath =
                    value.try_cast_into(|v| TCError::bad_request("invalid parent class", v))?;

                let class = InstanceClass::new(Some(classpath), Map::default());
                Object::Class(class)
            }
            Self::Instance => {
                let class = InstanceClass::new(Some(value.class().path().into()), Map::default());
                Object::Instance(InstanceExt::new(value.into(), class))
            }
        };

        Ok(object)
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

#[async_trait]
impl de::FromStream for Object {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(ObjectVisitor).await
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

struct ObjectVisitor;

#[async_trait]
impl de::Visitor for ObjectVisitor {
    type Value = Object;

    fn expecting() -> &'static str {
        "a user-defined Class or Instance"
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Object, A::Error> {
        let key = access
            .next_key::<TCPathBuf>(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, Self::expecting()))?;

        if let Some(class) = ObjectType::from_path(&key) {
            match class {
                ObjectType::Class => access.next_value(()).map_ok(Object::Class).await,
                ObjectType::Instance => Err(de::Error::custom(ERR_DECODE_INSTANCE)),
            }
        } else {
            Err(de::Error::invalid_value(key, Self::expecting()))
        }
    }
}
