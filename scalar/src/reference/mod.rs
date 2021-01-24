use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::{de, Decoder, Encoder, FromStream, IntoStream, MapAccess, ToStream};

use generic::*;

use super::{Link, Scalar};

pub mod id;

pub use id::*;

const PREFIX: PathLabel = path_label(&["state", "scalar", "ref"]);

pub trait RefInstance {
    fn requires(&self, deps: &mut HashSet<Id>);
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RefType {
    Id,
}

impl Class for RefType {
    type Instance = TCRef;
}

impl NativeClass for RefType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() > 3 && &path[0..3] == &PREFIX[..] {
            match path[3].as_str() {
                "id" if path.len() == 4 => Some(Self::Id),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::Id => "id",
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id => f.write_str("Id"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCRef {
    Id(IdRef),
}

impl Instance for TCRef {
    type Class = RefType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Id(_) => RefType::Id,
        }
    }
}

impl RefInstance for TCRef {
    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::Id(id_ref) => id_ref.requires(deps),
        }
    }
}

struct RefVisitor;

#[async_trait]
impl de::Visitor for RefVisitor {
    type Value = TCRef;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Ref, like {\"$subject\": []} or {\"/path/to/op\": [\"key\"]")
    }

    async fn visit_map<A: MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let key = access
            .next_key::<String>()
            .await?
            .ok_or_else(|| de::Error::custom("expected a Ref or Link, found empty map"))?;
        let value = access.next_value::<Scalar>().await?;

        if key.starts_with('$') {
            let subject = IdRef::from_str(&key).map_err(de::Error::custom)?;
            if value.is_none() {
                Ok(TCRef::Id(subject))
            } else {
                unimplemented!()
            }
        } else if let Ok(_link) = Link::from_str(&key) {
            unimplemented!()
        } else {
            Err(de::Error::invalid_type(key, &"Ref or Link"))
        }
    }
}

#[async_trait]
impl FromStream for TCRef {
    async fn from_stream<D: Decoder>(d: &mut D) -> Result<Self, <D as Decoder>::Error> {
        d.decode_map(RefVisitor).await
    }
}

impl<'en> ToStream<'en> for TCRef {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Id(id_ref) => id_ref.to_stream(e),
        }
    }
}

impl<'en> IntoStream<'en> for TCRef {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Id(id_ref) => id_ref.into_stream(e),
        }
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id(id_ref) => fmt::Display::fmt(id_ref, f),
        }
    }
}
