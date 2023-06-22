//! Resolve a reference conditionally.

use std::collections::HashSet;
use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use get_size::GetSize;
use get_size_derive::*;
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::public::{Public, StateInstance, ToState};
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment, TCPathBuf};

use crate::{OpDef, Scalar, Scope};

use super::{Refer, TCRef};

/// A conditional reference.
#[derive(Clone, Eq, PartialEq, GetSize)]
pub struct IfRef {
    cond: TCRef,
    then: Scalar,
    or_else: Scalar,
}

#[async_trait]
impl<State> Refer<State> for IfRef
where
    State: StateInstance + Refer<State> + From<Scalar>,
    State::Closure: From<(Map<State>, OpDef)> + TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError>,
    Value: TryFrom<State, Error = TCError> + TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            cond: self.cond.dereference_self(path),
            then: self.then.dereference_self(path),
            or_else: self.or_else.dereference_self(path),
        }
    }

    fn is_conditional(&self) -> bool {
        true
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.cond.is_inter_service_write(cluster_path)
            || self.then.is_inter_service_write(cluster_path)
            || self.or_else.is_inter_service_write(cluster_path)
    }

    fn is_ref(&self) -> bool {
        true
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        Self {
            cond: self.cond.reference_self(path),
            then: self.then.reference_self(path),
            or_else: self.or_else.reference_self(path),
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        self.cond.requires(deps);
    }

    async fn resolve<'a, T: ToState<State> + Public<State> + Instance>(
        self,
        context: &'a Scope<'a, State, T>,
        txn: &'a State::Txn,
    ) -> TCResult<State> {
        debug!("If::resolve {:?}", self);

        if self.cond.is_conditional() {
            return Err(bad_request!(
                "If does not allow a nested conditional {:?}",
                self.cond,
            ));
        }

        let cond = self.cond.resolve(context, txn).await?;
        debug!("If condition is {:?}", cond);

        if cond.matches::<bool>() {
            if cond.opt_cast_into().expect("if condition") {
                Ok(self.then.into())
            } else {
                Ok(self.or_else.into())
            }
        } else {
            Err(TCError::unexpected(cond, "a boolean condition"))
        }
    }
}

impl<'a, D: Digest> Hash<D> for &'a IfRef {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((&self.cond, &self.then, &self.or_else))
    }
}

impl TryCastFrom<Scalar> for IfRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<(TCRef, Scalar, Scalar)>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        scalar.opt_cast_into().map(|(cond, then, or_else)| Self {
            cond,
            then,
            or_else,
        })
    }
}

#[async_trait]
impl de::FromStream for IfRef {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (cond, then, or_else) =
            <(TCRef, Scalar, Scalar) as de::FromStream>::from_stream(context, decoder).await?;

        Ok(Self {
            cond,
            then,
            or_else,
        })
    }
}

impl<'en> en::IntoStream<'en> for IfRef {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.cond, self.then, self.or_else).into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for IfRef {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.cond, &self.then, &self.or_else), encoder)
    }
}

impl fmt::Debug for IfRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "if {:?} then {:?} else {:?}",
            self.cond, self.then, self.or_else
        )
    }
}
