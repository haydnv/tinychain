//! Resolve a `Closure` repeatedly while a condition is met.

use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::try_join;
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tcgeneric::{Id, Instance, PathSegment, TCPathBuf};

use crate::closure::Closure;
use crate::route::Public;
use crate::scalar::{Number, Scalar, Scope, Value};
use crate::state::{State, ToState};
use crate::txn::Txn;

use super::Refer;

/// A while loop.
#[derive(Clone, Eq, PartialEq)]
pub struct While {
    cond: Scalar,
    closure: Scalar,
    state: Scalar,
}

#[async_trait]
impl Refer for While {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            cond: self.cond.dereference_self(path),
            closure: self.closure.dereference_self(path),
            state: self.state.dereference_self(path),
        }
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.cond.is_inter_service_write(cluster_path)
            || self.closure.is_inter_service_write(cluster_path)
            || self.state.is_inter_service_write(cluster_path)
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        Self {
            cond: self.cond.reference_self(path),
            closure: self.closure.reference_self(path),
            state: self.state.reference_self(path),
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        self.cond.requires(deps);
        self.closure.requires(deps);
        self.state.requires(deps);
    }

    async fn resolve<'a, T: ToState + Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        debug!("While::resolve {}", self);

        let (cond, closure, mut state) = try_join!(
            self.cond.resolve(context, txn),
            self.closure.resolve(context, txn),
            self.state.resolve(context, txn)
        )?;

        let cond = Closure::try_cast_from(cond, |s| {
            TCError::bad_request("while loop condition should be an Op or Closure, found", s)
        })?;

        let closure = Closure::try_cast_from(closure, |s| {
            TCError::bad_request("while loop requires an Op or Closure, found", s)
        })?;

        loop {
            let still_going = cond.clone().call(txn, state.clone()).await?;

            debug!("While condition is {}", cond);
            if let State::Scalar(Scalar::Value(Value::Number(Number::Bool(b)))) = still_going {
                if b.into() {
                    state = closure.clone().call(txn, state).await?;
                } else {
                    break Ok(state);
                }
            } else {
                break Err(TCError::bad_request(
                    "expected boolean while condition but found",
                    cond,
                ));
            }
        }
    }
}

impl TryCastFrom<Scalar> for While {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<(Scalar, Scalar, Scalar)>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        if scalar.matches::<(Scalar, Scalar, Scalar)>() {
            scalar.opt_cast_into().map(|(cond, closure, state)| Self {
                cond,
                closure,
                state,
            })
        } else {
            None
        }
    }
}

#[async_trait]
impl de::FromStream for While {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        let while_loop = Scalar::from_stream(context, decoder).await?;
        Self::try_cast_from(while_loop, |s| de::Error::invalid_value(s, "a While loop"))
    }
}

impl<'en> en::IntoStream<'en> for While {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.cond, self.closure, self.state).into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for While {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.cond, &self.closure, &self.state), encoder)
    }
}

impl fmt::Debug for While {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "while {:?} call {:?} with state {:?}",
            self.cond, self.closure, self.state
        )
    }
}

impl fmt::Display for While {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "while {} call {} with state {}",
            self.cond, self.closure, self.state
        )
    }
}
