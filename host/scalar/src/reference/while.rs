//! Resolve a `Closure` repeatedly while a condition is met.

use std::collections::HashSet;
use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use futures::try_join;
use get_size::GetSize;
use get_size_derive::*;
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::public::{ClosureInstance, Public, StateInstance, ToState};
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment, TCPathBuf};

use crate::{OpDef, Scalar, Scope};

use super::Refer;

/// A while loop.
#[derive(Clone, Eq, PartialEq, GetSize)]
pub struct While {
    cond: Scalar,
    closure: Scalar,
    state: Scalar,
}

#[async_trait]
impl<State> Refer<State> for While
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

    fn is_ref(&self) -> bool {
        true
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

    async fn resolve<'a, T: ToState<State> + Public<State> + Instance>(
        self,
        context: &'a Scope<'a, State, T>,
        txn: &'a State::Txn,
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

        debug!("While condition definition is {:?}", cond);

        loop {
            let mut cond = cond.clone();
            let still_going = loop {
                let cond_op = State::Closure::try_cast_from(cond.clone(), |s| {
                    bad_request!("expected an Op or Closure for a While loop but found {s:?}")
                })?;

                let intermediate = Box::new(cond_op).call(txn.clone(), state.clone()).await?;
                if intermediate.is_ref() {
                    cond = intermediate;
                } else {
                    break bool::try_cast_from(intermediate, |s| {
                        bad_request!("expected a boolean condition but found {s:?}")
                    })?;
                }
            };

            if still_going {
                let while_op: State::Closure = closure.clone().try_cast_into(|s| {
                    bad_request!("expected an Op or Closure for a While loop but found {s:?}")
                })?;

                state = Box::new(while_op).call(txn.clone(), state).await?;

                if state.is_conditional() {
                    return Err(bad_request!(
                        "conditional state {state:?} is not allowed in a While loop",
                    ));
                }

                debug!("While loop state is {state:?}");
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
