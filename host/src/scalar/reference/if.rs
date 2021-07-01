//! Resolve a reference conditionally.

use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tcgeneric::{Id, Instance, TCPathBuf};

use crate::route::Public;
use crate::scalar::{Number, Scalar, Scope, Value};
use crate::state::State;
use crate::txn::Txn;

use super::{Refer, TCRef};
use crate::generic::PathSegment;

/// A conditional reference.
#[derive(Clone, Eq, PartialEq)]
pub struct IfRef {
    cond: TCRef,
    then: Scalar,
    or_else: Scalar,
}

#[async_trait]
impl Refer for IfRef {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            cond: self.cond.dereference_self(path),
            then: self.then.dereference_self(path),
            or_else: self.or_else.dereference_self(path),
        }
    }

    fn is_derived_write(&self) -> bool {
        self.cond.is_derived_write()
            || self.then.is_derived_write()
            || self.or_else.is_derived_write()
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.cond.is_inter_service_write(cluster_path)
            || self.then.is_inter_service_write(cluster_path)
            || self.or_else.is_inter_service_write(cluster_path)
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

    async fn resolve<'a, T: Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        debug!("If::resolve {}", self);

        let cond = self.cond.resolve(context, txn).await?;
        debug!("If condition is {}", cond);

        if let State::Scalar(Scalar::Value(Value::Number(Number::Bool(b)))) = cond {
            if b.into() {
                Ok(self.then.into())
            } else {
                Ok(self.or_else.into())
            }
        } else {
            Err(TCError::bad_request(
                "expected boolean condition but found",
                cond,
            ))
        }
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

impl fmt::Display for IfRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "if {} then {} else {}",
            self.cond, self.then, self.or_else
        )
    }
}
