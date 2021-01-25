use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, Decoder, Encoder, FromStream, IntoStream, ToStream};
use futures::TryFutureExt;

use generic::*;

use super::Scalar;

pub mod id;
pub mod op;

pub use id::*;
pub use op::*;

const PREFIX: PathLabel = path_label(&["state", "scalar", "ref"]);

pub trait RefInstance {
    fn requires(&self, deps: &mut HashSet<Id>);
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum RefType {
    Id,
    Op(OpRefType),
}

impl Class for RefType {
    type Instance = TCRef;
}

impl NativeClass for RefType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() > 3 && &path[0..3] == &PREFIX[..] {
            match path[3].as_str() {
                "id" if path.len() == 4 => Some(Self::Id),
                "op" => OpRefType::from_path(path).map(RefType::Op),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::Id => "id",
            Self::Op(ort) => return ort.path(),
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id => f.write_str("Id"),
            Self::Op(ort) => fmt::Display::fmt(ort, f),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCRef {
    Id(IdRef),
    Op(OpRef),
}

impl Instance for TCRef {
    type Class = RefType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Id(_) => RefType::Id,
            Self::Op(op_ref) => RefType::Op(op_ref.class()),
        }
    }
}

impl RefInstance for TCRef {
    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::Id(id_ref) => id_ref.requires(deps),
            Self::Op(op_ref) => op_ref.requires(deps),
        }
    }
}

pub struct RefVisitor;

impl RefVisitor {
    pub async fn visit_map_value<A: de::MapAccess>(
        class: RefType,
        access: &mut A,
    ) -> Result<TCRef, A::Error> {
        match class {
            RefType::Id => access.next_value().map_ok(TCRef::Id).await,
            RefType::Op(ort) => {
                OpRefVisitor::visit_map_value(ort, access)
                    .map_ok(TCRef::Op)
                    .await
            }
        }
    }

    pub fn visit_ref_value<E: de::Error>(subject: Subject, params: Scalar) -> Result<TCRef, E> {
        if params.is_none() {
            match subject {
                Subject::Link(link) => Err(de::Error::invalid_type(link, &"a Ref")),
                Subject::Ref(id_ref) => Ok(TCRef::Id(id_ref)),
            }
        } else {
            OpRefVisitor::visit_ref_value(subject, params).map(TCRef::Op)
        }
    }
}

#[async_trait]
impl de::Visitor for RefVisitor {
    type Value = TCRef;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Ref, like {\"$subject\": []} or {\"/path/to/op\": [\"key\"]")
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let subject = access
            .next_key::<Subject>()
            .await?
            .ok_or_else(|| de::Error::custom("expected a Ref or Link, found empty map"))?;

        if let Subject::Link(link) = &subject {
            if link.host().is_none() {
                if let Some(class) = RefType::from_path(link.path()) {
                    return Self::visit_map_value(class, &mut access).await;
                }
            }
        }

        let params = access.next_value().await?;
        Self::visit_ref_value(subject, params)
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
            Self::Op(op_ref) => op_ref.to_stream(e),
        }
    }
}

impl<'en> IntoStream<'en> for TCRef {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Id(id_ref) => id_ref.into_stream(e),
            Self::Op(op_ref) => op_ref.into_stream(e),
        }
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Id(id_ref) => fmt::Display::fmt(id_ref, f),
            Self::Op(op_ref) => fmt::Display::fmt(op_ref, f),
        }
    }
}
