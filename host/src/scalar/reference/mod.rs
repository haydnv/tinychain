//! Utilities to reference to a [`State`] within a [`Txn`], and resolve that [`TCRef`].

use std::collections::HashSet;
use std::convert::TryFrom;
use std::fmt;

use async_trait::async_trait;
use destream::{de, Decoder, EncodeMap, Encoder, FromStream, IntoStream, ToStream};
use futures::TryFutureExt;
use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tcgeneric::*;

use crate::route::Public;
use crate::state::State;
use crate::txn::Txn;

use super::{Scalar, Scope, Value};

pub use after::After;
pub use case::Case;
pub use id::*;
pub use op::*;
pub use r#if::IfRef;

mod after;
mod case;
mod r#if;

pub mod id;
pub mod op;

const PREFIX: PathLabel = path_label(&["state", "scalar", "ref"]);

/// Trait defining dependencies and a resolution method for a [`TCRef`].
#[async_trait]
pub trait Refer {
    /// Return `true` if resolving this reference may mutate some `State`.
    fn is_write(&self) -> bool;

    /// Add the dependency [`Id`]s of this reference to the given set.
    fn requires(&self, deps: &mut HashSet<Id>);

    /// Resolve this reference with respect to the given context.
    async fn resolve<'a, T: Public + Instance>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State>;
}

/// The [`Class`] of a [`TCRef`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefType {
    After,
    Case,
    Id,
    If,
    Op(OpRefType),
}

impl Class for RefType {}

impl NativeClass for RefType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() > 3 && &path[0..3] == &PREFIX[..] {
            match path[3].as_str() {
                "after" if path.len() == 4 => Some(Self::After),
                "case" if path.len() == 4 => Some(Self::Case),
                "id" if path.len() == 4 => Some(Self::Id),
                "if" if path.len() == 4 => Some(Self::If),
                "op" => OpRefType::from_path(path).map(RefType::Op),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::After => "after",
            Self::Case => "case",
            Self::Id => "id",
            Self::If => "if",
            Self::Op(ort) => return ort.path(),
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::After => f.write_str("After"),
            Self::Case => f.write_str("Case"),
            Self::Id => f.write_str("Id"),
            Self::If => f.write_str("If"),
            Self::Op(ort) => fmt::Display::fmt(ort, f),
        }
    }
}

/// A reference to a `State`.
#[derive(Clone, Eq, PartialEq)]
pub enum TCRef {
    After(Box<After>),
    Case(Box<Case>),
    Id(IdRef),
    If(Box<IfRef>),
    Op(OpRef),
}

impl Instance for TCRef {
    type Class = RefType;

    fn class(&self) -> Self::Class {
        match self {
            Self::After(_) => RefType::After,
            Self::Case(_) => RefType::Case,
            Self::Id(_) => RefType::Id,
            Self::If(_) => RefType::If,
            Self::Op(op_ref) => RefType::Op(op_ref.class()),
        }
    }
}

#[async_trait]
impl Refer for TCRef {
    fn is_write(&self) -> bool {
        match self {
            Self::After(after) => after.is_write(),
            Self::Case(case) => case.is_write(),
            Self::Id(id_ref) => id_ref.is_write(),
            Self::If(if_ref) => if_ref.is_write(),
            Self::Op(op_ref) => op_ref.is_write(),
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::After(after) => after.requires(deps),
            Self::Case(case) => case.requires(deps),
            Self::Id(id_ref) => id_ref.requires(deps),
            Self::If(if_ref) => if_ref.requires(deps),
            Self::Op(op_ref) => op_ref.requires(deps),
        }
    }

    async fn resolve<'a, T: Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        debug!("TCRef::resolve {}", self);

        let mut state = State::from(self);
        while let State::Scalar(Scalar::Ref(tc_ref)) = state {
            state = match *tc_ref {
                Self::After(after) => after.resolve(context, txn).await,
                Self::Case(case) => case.resolve(context, txn).await,
                Self::Id(id_ref) => id_ref.resolve(context, txn).await,
                Self::If(if_ref) => if_ref.resolve(context, txn).await,
                Self::Op(op_ref) => op_ref.resolve(context, txn).await,
            }?;
        }

        Ok(state)
    }
}

impl TryFrom<TCRef> for OpRef {
    type Error = TCError;

    fn try_from(tc_ref: TCRef) -> TCResult<Self> {
        match tc_ref {
            TCRef::Op(op_ref) => Ok(op_ref),
            other => Err(TCError::bad_request("expected an OpRef but found", other)),
        }
    }
}

impl TryCastFrom<TCRef> for OpRef {
    fn can_cast_from(tc_ref: &TCRef) -> bool {
        match tc_ref {
            TCRef::Op(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(tc_ref: TCRef) -> Option<Self> {
        match tc_ref {
            TCRef::Op(op) => Some(op),
            _ => None,
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
            RefType::After => access.next_value(()).map_ok(TCRef::Id).await,
            RefType::Case => {
                access
                    .next_value(())
                    .map_ok(Box::new)
                    .map_ok(TCRef::Case)
                    .await
            }
            RefType::Id => access.next_value(()).map_ok(TCRef::Id).await,
            RefType::If => {
                access
                    .next_value(())
                    .map_ok(Box::new)
                    .map_ok(TCRef::If)
                    .await
            }
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
                Subject::Ref(id_ref, path) if path.is_empty() => Ok(TCRef::Id(id_ref)),
                Subject::Ref(id_ref, path) => Ok(TCRef::Op(OpRef::Get((
                    Subject::Ref(id_ref, path),
                    Value::default().into(),
                )))),
            }
        } else {
            OpRefVisitor::visit_ref_value(subject, params).map(TCRef::Op)
        }
    }
}

#[async_trait]
impl de::Visitor for RefVisitor {
    type Value = TCRef;

    fn expecting() -> &'static str {
        "a Ref, like {\"$subject\": []} or {\"/path/to/op\": [\"key\"]"
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let subject = access
            .next_key::<Subject>(())
            .await?
            .ok_or_else(|| de::Error::custom("expected a Ref or Link, found empty map"))?;

        if let Subject::Link(link) = &subject {
            if link.host().is_none() {
                if let Some(class) = RefType::from_path(link.path()) {
                    return Self::visit_map_value(class, &mut access).await;
                }
            }
        }

        let params = access.next_value(()).await?;
        Self::visit_ref_value(subject, params)
    }
}

#[async_trait]
impl FromStream for TCRef {
    type Context = ();

    async fn from_stream<D: Decoder>(_: (), d: &mut D) -> Result<Self, <D as Decoder>::Error> {
        d.decode_map(RefVisitor).await
    }
}

impl<'en> ToStream<'en> for TCRef {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        if let Self::Id(id_ref) = self {
            return id_ref.to_stream(e);
        } else if let Self::Op(op_ref) = self {
            return op_ref.to_stream(e);
        };

        let mut map = e.encode_map(Some(1))?;

        map.encode_key(self.class().path().to_string())?;
        match self {
            Self::Id(_) => unimplemented!("this should not be possible"),
            Self::Op(_) => unimplemented!("this should not be possible"),

            Self::After(after) => map.encode_value(after),
            Self::Case(case) => map.encode_value(case),
            Self::If(if_ref) => map.encode_value(if_ref),
        }?;

        map.end()
    }
}

impl<'en> IntoStream<'en> for TCRef {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        if let Self::Id(id_ref) = self {
            return id_ref.into_stream(e);
        } else if let Self::Op(op_ref) = self {
            return op_ref.into_stream(e);
        };

        let mut map = e.encode_map(Some(1))?;

        map.encode_key(self.class().path().to_string())?;
        match self {
            Self::Id(_) => unimplemented!("this should not be possible"),
            Self::Op(_) => unimplemented!("this should not be possible"),

            Self::After(after) => map.encode_value(after),
            Self::Case(case) => map.encode_value(case),
            Self::If(if_ref) => map.encode_value(if_ref),
        }?;

        map.end()
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::After(after) => fmt::Display::fmt(after, f),
            Self::Case(case) => fmt::Display::fmt(case, f),
            Self::Id(id_ref) => fmt::Display::fmt(id_ref, f),
            Self::If(if_ref) => fmt::Display::fmt(if_ref, f),
            Self::Op(op_ref) => fmt::Display::fmt(op_ref, f),
        }
    }
}
