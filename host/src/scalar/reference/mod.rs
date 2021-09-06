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
use crate::state::{State, ToState};
use crate::txn::Txn;

use super::{Scalar, Scope, Value};

pub use after::After;
pub use before::Before;
pub use case::Case;
pub use id::*;
pub use op::*;
pub use r#if::IfRef;
pub use r#while::While;
pub use with::With;

mod after;
mod before;
mod case;
mod r#if;
mod r#while;
mod with;

pub mod id;
pub mod op;

const PREFIX: PathLabel = path_label(&["state", "scalar", "ref"]);

/// Trait defining dependencies and a resolution method for a [`TCRef`].
#[async_trait]
pub trait Refer {
    /// Replace references to "$self" with the given relative path.
    ///
    /// This is used to control whether or not an OpDef will be replicated.
    fn dereference_self(self, path: &TCPathBuf) -> Self;

    /// Return `true` if this is a conditional reference (e.g. `If` or `Case`).
    fn is_conditional(&self) -> bool;

    /// Return `true` if this references a write operation to a cluster other than the path given.
    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool;

    /// Replace the given relative path with "$self".
    ///
    /// This is used to control whether or not an OpDef will be replicated.
    fn reference_self(self, path: &TCPathBuf) -> Self;

    /// Add the dependency [`Id`]s of this reference to the given set.
    fn requires(&self, deps: &mut HashSet<Id>);

    /// Resolve this reference with respect to the given context.
    async fn resolve<'a, T: ToState + Public + Instance>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State>;
}

/// The [`Class`] of a [`TCRef`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefType {
    After,
    Before,
    Case,
    Id,
    If,
    Op(OpRefType),
    While,
    With,
}

impl Class for RefType {}

impl NativeClass for RefType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 4 && &path[0..3] == &PREFIX[..] {
            match path[3].as_str() {
                "after" => Some(Self::After),
                "before" => Some(Self::Before),
                "case" => Some(Self::Case),
                "id" => Some(Self::Id),
                "if" => Some(Self::If),
                "while" => Some(Self::While),
                "with" => Some(Self::With),
                _ => None,
            }
        } else if let Some(ort) = OpRefType::from_path(path) {
            Some(RefType::Op(ort))
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::After => "after",
            Self::Before => "before",
            Self::Case => "case",
            Self::Id => "id",
            Self::If => "if",
            Self::Op(ort) => return ort.path(),
            Self::While => "while",
            Self::With => "with",
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
    }
}

impl fmt::Display for RefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::After => f.write_str("After"),
            Self::Before => f.write_str("Before"),
            Self::Case => f.write_str("Case"),
            Self::Id => f.write_str("Id"),
            Self::If => f.write_str("If"),
            Self::Op(ort) => fmt::Display::fmt(ort, f),
            Self::While => f.write_str("While"),
            Self::With => f.write_str("With"),
        }
    }
}

/// A reference to a `State`.
#[derive(Clone, Eq, PartialEq)]
pub enum TCRef {
    After(Box<After>),
    Before(Box<Before>),
    Case(Box<Case>),
    Id(IdRef),
    If(Box<IfRef>),
    Op(OpRef),
    While(Box<While>),
    With(Box<With>),
}

impl Instance for TCRef {
    type Class = RefType;

    fn class(&self) -> Self::Class {
        match self {
            Self::After(_) => RefType::After,
            Self::Before(_) => RefType::Before,
            Self::Case(_) => RefType::Case,
            Self::Id(_) => RefType::Id,
            Self::If(_) => RefType::If,
            Self::Op(op_ref) => RefType::Op(op_ref.class()),
            Self::While(_) => RefType::While,
            Self::With(_) => RefType::With,
        }
    }
}

#[async_trait]
impl Refer for TCRef {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::After(after) => {
                let after = after.dereference_self(path);
                Self::After(Box::new(after))
            }
            Self::Before(before) => {
                let before = before.dereference_self(path);
                Self::Before(Box::new(before))
            }
            Self::Case(case) => {
                let case = case.dereference_self(path);
                Self::Case(Box::new(case))
            }
            Self::Id(id_ref) => Self::Id(id_ref.dereference_self(path)),
            Self::If(if_ref) => {
                let if_ref = if_ref.dereference_self(path);
                Self::If(Box::new(if_ref))
            }
            Self::Op(op_ref) => Self::Op(op_ref.dereference_self(path)),
            Self::While(while_ref) => {
                let while_ref = while_ref.dereference_self(path);
                Self::While(Box::new(while_ref))
            }
            Self::With(with) => {
                let with = with.dereference_self(path);
                Self::With(Box::new(with))
            }
        }
    }

    fn is_conditional(&self) -> bool {
        match self {
            Self::After(after) => after.is_conditional(),
            Self::Before(before) => before.is_conditional(),
            Self::Case(case) => case.is_conditional(),
            Self::Id(id_ref) => id_ref.is_conditional(),
            Self::If(if_ref) => if_ref.is_conditional(),
            Self::Op(op_ref) => op_ref.is_conditional(),
            Self::While(while_ref) => while_ref.is_conditional(),
            Self::With(with) => with.is_conditional(),
        }
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        match self {
            Self::After(after) => after.is_inter_service_write(cluster_path),
            Self::Before(before) => before.is_inter_service_write(cluster_path),
            Self::Case(case) => case.is_inter_service_write(cluster_path),
            Self::Id(id_ref) => id_ref.is_inter_service_write(cluster_path),
            Self::If(if_ref) => if_ref.is_inter_service_write(cluster_path),
            Self::Op(op_ref) => op_ref.is_inter_service_write(cluster_path),
            Self::While(while_ref) => while_ref.is_inter_service_write(cluster_path),
            Self::With(with) => with.is_inter_service_write(cluster_path),
        }
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::After(after) => {
                let after = after.reference_self(path);
                Self::After(Box::new(after))
            }
            Self::Before(before) => {
                let before = before.reference_self(path);
                Self::Before(Box::new(before))
            }
            Self::Case(case) => {
                let case = case.reference_self(path);
                Self::Case(Box::new(case))
            }
            Self::Id(id_ref) => Self::Id(id_ref.reference_self(path)),
            Self::If(if_ref) => {
                let if_ref = if_ref.reference_self(path);
                Self::If(Box::new(if_ref))
            }
            Self::Op(op_ref) => Self::Op(op_ref.reference_self(path)),
            Self::While(while_ref) => {
                let while_ref = while_ref.reference_self(path);
                Self::While(Box::new(while_ref))
            }
            Self::With(with) => {
                let with = with.reference_self(path);
                Self::With(Box::new(with))
            }
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::After(after) => after.requires(deps),
            Self::Before(before) => before.requires(deps),
            Self::Case(case) => case.requires(deps),
            Self::Id(id_ref) => id_ref.requires(deps),
            Self::If(if_ref) => if_ref.requires(deps),
            Self::Op(op_ref) => op_ref.requires(deps),
            Self::While(while_ref) => while_ref.requires(deps),
            Self::With(with) => with.requires(deps),
        }
    }

    async fn resolve<'a, T: ToState + Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        debug!("TCRef::resolve {}", self);

        match self {
            Self::If(if_ref) => if_ref.resolve(context, txn).await,
            Self::Case(case) => case.resolve(context, txn).await,
            Self::After(after) => after.resolve(context, txn).await,
            Self::Before(before) => before.resolve(context, txn).await,
            Self::Id(id_ref) => id_ref.resolve(context, txn).await,
            Self::Op(op_ref) => op_ref.resolve(context, txn).await,
            Self::While(while_ref) => while_ref.resolve(context, txn).await,
            Self::With(with) => with.resolve(context, txn).await,
        }
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

impl TryCastFrom<TCRef> for Id {
    fn can_cast_from(tc_ref: &TCRef) -> bool {
        match tc_ref {
            TCRef::Id(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(tc_ref: TCRef) -> Option<Self> {
        match tc_ref {
            TCRef::Id(id_ref) => Some(id_ref.into_id()),
            _ => None,
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

/// A helper struct used to deserialize a [`TCRef`]
pub struct RefVisitor;

impl RefVisitor {
    /// Deserialize a map value, assuming it's an instance of the given [`RefType`].
    pub async fn visit_map_value<A: de::MapAccess>(
        class: RefType,
        access: &mut A,
    ) -> Result<TCRef, A::Error> {
        match class {
            RefType::After => {
                access
                    .next_value(())
                    .map_ok(Box::new)
                    .map_ok(TCRef::After)
                    .await
            }
            RefType::Before => {
                access
                    .next_value(())
                    .map_ok(Box::new)
                    .map_ok(TCRef::Before)
                    .await
            }
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
            RefType::While => {
                access
                    .next_value(())
                    .map_ok(Box::new)
                    .map_ok(TCRef::While)
                    .await
            }
            RefType::With => {
                access
                    .next_value(())
                    .map_ok(Box::new)
                    .map_ok(TCRef::With)
                    .await
            }
        }
    }

    /// Deserialize a [`TCRef`] with the given `subject`.
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
        let subject = access.next_key::<Subject>(()).await?;

        let subject =
            subject.ok_or_else(|| de::Error::custom("expected a Ref or Link, found empty map"))?;

        if let Subject::Link(link) = &subject {
            if link.host().is_none() {
                if let Some(class) = RefType::from_path(link.path()) {
                    debug!("RefVisitor visiting instance of {}...", class);
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
            Self::Before(before) => map.encode_value(before),
            Self::Case(case) => map.encode_value(case),
            Self::If(if_ref) => map.encode_value(if_ref),
            Self::While(while_ref) => map.encode_value(while_ref),
            Self::With(with) => map.encode_value(with),
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
            Self::Before(before) => map.encode_value(before),
            Self::Case(case) => map.encode_value(case),
            Self::If(if_ref) => map.encode_value(if_ref),
            Self::While(while_ref) => map.encode_value(while_ref),
            Self::With(with) => map.encode_value(with),
        }?;

        map.end()
    }
}

impl fmt::Debug for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::After(after) => fmt::Debug::fmt(after, f),
            Self::Before(before) => fmt::Debug::fmt(before, f),
            Self::Case(case) => fmt::Debug::fmt(case, f),
            Self::Id(id_ref) => fmt::Debug::fmt(id_ref, f),
            Self::If(if_ref) => fmt::Debug::fmt(if_ref, f),
            Self::Op(op_ref) => fmt::Debug::fmt(op_ref, f),
            Self::While(while_ref) => fmt::Debug::fmt(while_ref, f),
            Self::With(with) => fmt::Debug::fmt(with, f),
        }
    }
}

impl fmt::Display for TCRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::After(after) => fmt::Display::fmt(after, f),
            Self::Before(before) => fmt::Display::fmt(before, f),
            Self::Case(case) => fmt::Display::fmt(case, f),
            Self::Id(id_ref) => fmt::Display::fmt(id_ref, f),
            Self::If(if_ref) => fmt::Display::fmt(if_ref, f),
            Self::Op(op_ref) => fmt::Display::fmt(op_ref, f),
            Self::While(while_ref) => fmt::Display::fmt(while_ref, f),
            Self::With(with) => fmt::Display::fmt(with, f),
        }
    }
}
