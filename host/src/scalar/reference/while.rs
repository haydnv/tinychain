//! Resolve a `Closure` repeatedly while a condition is met.

use std::collections::HashSet;
use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use futures::try_join;
use get_size::GetSize;
use get_size_derive::*;
use log::{debug, warn};
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
#[derive(Clone, Eq, PartialEq, GetSize)]
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

    fn is_conditional(&self) -> bool {
        self.closure.is_conditional()
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
        debug!("While::resolve {:?}", self);

        if self.cond.is_conditional() {
            return Err(bad_request!(
                "While does not allow a nested conditional {:?}",
                self.cond,
            ));
        } else if self.state.is_conditional() {
            return Err(bad_request!(
                "While does not allow a nested conditional {:?}",
                self.state,
            ));
        }

        let (cond, closure, mut state) = try_join!(
            self.cond.resolve(context, txn),
            self.closure.resolve(context, txn),
            self.state.resolve(context, txn)
        )?;

        let cond = Closure::try_cast_from(cond, |s| {
            TCError::unexpected(s, "an Op or Closure for a While")
        })?;

        let closure = Closure::try_cast_from(closure, |s| {
            TCError::unexpected(s, "an Op or Closure for a While")
        })?;

        debug!("While condition definition is {:?}", cond);

        loop {
            let mut cond = cond.clone();
            let still_going = loop {
                match cond.clone().call(txn, state.clone()).await? {
                    State::Scalar(Scalar::Value(Value::Number(Number::Bool(still_going)))) => {
                        break still_going.into()
                    }
                    State::Closure(closure) => {
                        warn!("While condition returned a nested {:?}", closure);
                        cond = closure;
                    }
                    State::Scalar(Scalar::Op(op_def)) => {
                        warn!("While condition returned a nested {:?}", op_def);
                        cond = op_def.into()
                    }
                    other => return Err(TCError::unexpected(other, "a condition for a While")),
                }
            };

            if still_going {
                state = closure.clone().call(txn, state).await?;

                if state.is_conditional() {
                    return Err(bad_request!(
                        "conditional State {:?} is not allowed in a While loop",
                        state,
                    ));
                }

                debug!("While loop state is {:?}", state);
            } else {
                break Ok(state);
            }
        }
    }
}

impl<'a, D: Digest> Hash<D> for &'a While {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((&self.cond, &self.closure, &self.state))
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
        Self::try_cast_from(while_loop, |s| {
            de::Error::invalid_value(format!("{s:?}"), "a While loop")
        })
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
