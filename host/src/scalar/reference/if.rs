//! Resolve a reference conditionally.

use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tcgeneric::{Id, Instance};

use crate::route::Public;
use crate::scalar::{Number, Scalar, Scope, Value};
use crate::state::State;
use crate::txn::Txn;

use super::{Refer, TCRef};

/// A conditional reference.
#[derive(Clone, Eq, PartialEq)]
pub struct IfRef {
    cond: TCRef,
    then: Scalar,
    or_else: Scalar,
}

#[async_trait]
impl Refer for IfRef {
    fn requires(&self, deps: &mut HashSet<Id>) {
        self.cond.requires(deps);
    }

    async fn resolve<T: Instance + Public>(self, context: &Scope<T>, txn: &Txn) -> TCResult<State> {
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

impl fmt::Display for IfRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "if {} then {} else {}",
            self.cond, self.then, self.or_else
        )
    }
}
