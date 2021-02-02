use std::fmt;

use async_trait::async_trait;
use destream::de;
use futures::TryFutureExt;

use generic::{PathSegment, TCPathBuf};

use crate::state::State;

mod class;
mod instance;

pub use class::*;
pub use instance::*;

const PREFIX: generic::PathLabel = generic::path_label(&["state", "object"]);

#[derive(Clone, Eq, PartialEq)]
pub enum ObjectType {
    Class,
    Instance,
}

impl generic::Class for ObjectType {
    type Instance = Object;
}

impl generic::NativeClass for ObjectType {
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

        TCPathBuf::from(PREFIX).append(generic::label(suffix))
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

#[derive(Clone)]
pub enum Object {
    Class(InstanceClass),
    Instance(InstanceExt<State>),
}

impl generic::Instance for Object {
    type Class = ObjectType;

    fn class(&self) -> ObjectType {
        match self {
            Self::Class(ic) => ObjectType::Class,
            Self::Instance(i) => ObjectType::Instance,
        }
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
impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Class(ict) => fmt::Display::fmt(ict, f),
            Self::Instance(ic) => fmt::Display::fmt(ic, f),
        }
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

        if let Some(class) = <ObjectType as generic::NativeClass>::from_path(&key) {
            match class {
                ObjectType::Class => access.next_value(()).map_ok(Object::Class).await,
                ObjectType::Instance => unimplemented!(),
            }
        } else {
            Err(de::Error::invalid_value(key, Self::expecting()))
        }
    }
}
