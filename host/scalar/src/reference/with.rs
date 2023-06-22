//! Limits the execution scope of an inline `Op`.

use std::collections::HashSet;
use std::fmt;
use std::ops::Deref;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use get_size::GetSize;
use get_size_derive::*;
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::public::{Public, StateInstance, ToState};
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment, TCPathBuf, Tuple};

use crate::{OpDef, Scalar, Scope, SELF};

use super::Refer;

/// A flow control operator which closes over the context of an [`OpDef`] to produce a [`Closure`].
#[derive(Clone, Eq, PartialEq, GetSize)]
pub struct With {
    capture: Tuple<Id>,
    op: OpDef,
}

impl With {
    pub fn new(capture: Tuple<Id>, op: OpDef) -> Self {
        With { capture, op }
    }
}

#[async_trait]
impl<State> Refer<State> for With
where
    State: StateInstance + Refer<State> + From<Scalar>,
    State::Closure: From<(Map<State>, OpDef)> + TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError>,
    Value: TryFrom<State, Error = TCError> + TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            capture: self.capture.into_iter().filter(|id| id != &SELF).collect(),
            op: self.op.dereference_self::<State>(path),
        }
    }

    fn is_conditional(&self) -> bool {
        false
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.op.is_inter_service_write::<State>(cluster_path)
    }

    fn is_ref(&self) -> bool {
        true
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        let before = self.op.clone();
        let op = self.op.reference_self::<State>(path);
        let capture = if op == before {
            self.capture
        } else {
            let mut capture = self.capture;
            capture.push(SELF.into());
            capture
        };

        Self { capture, op }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        deps.extend(self.capture.iter().filter(|id| *id != &SELF).cloned())
    }

    async fn resolve<'a, T: ToState<State> + Public<State> + Instance>(
        self,
        context: &'a Scope<'a, State, T>,
        _txn: &'a State::Txn,
    ) -> TCResult<State> {
        let closed_over = self
            .capture
            .into_iter()
            .map(|id| {
                context.resolve_id(&id).map(|state| {
                    debug!("closure captured {}: {:?}", id, state);
                    (id, state)
                })
            })
            .collect::<TCResult<Map<State>>>()?;

        Ok(State::Closure::from((closed_over, self.op)).into())
    }
}

impl<'a, D: Digest> Hash<D> for &'a With {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((self.capture.deref(), &self.op))
    }
}

impl TryCastFrom<Scalar> for With {
    fn can_cast_from(scalar: &Scalar) -> bool {
        if let Scalar::Tuple(tuple) = scalar {
            if tuple.len() == 2 {
                if !OpDef::can_cast_from(&tuple[1]) {
                    return false;
                }

                return match &tuple[0] {
                    Scalar::Tuple(capture) => capture.iter().all(Id::can_cast_from),
                    Scalar::Value(Value::Tuple(capture)) => capture.iter().all(Id::can_cast_from),
                    _ => false,
                };
            }
        }

        false
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        let (capture, op): (Scalar, OpDef) = scalar.opt_cast_into()?;
        let capture = match capture {
            Scalar::Tuple(capture) => capture
                .into_iter()
                .map(Id::opt_cast_from)
                .collect::<Option<Tuple<Id>>>(),

            Scalar::Value(Value::Tuple(capture)) => capture
                .into_iter()
                .map(Id::opt_cast_from)
                .collect::<Option<Tuple<Id>>>(),

            _ => None,
        }?;

        Some(Self { capture, op })
    }
}

#[async_trait]
impl de::FromStream for With {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(context, decoder)
            .map_ok(|(capture, op)| Self { capture, op })
            .await
    }
}

impl<'en> en::IntoStream<'en> for With {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.capture, self.op).into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for With {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.capture, &self.op), encoder)
    }
}

impl fmt::Debug for With {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "with {:?}: {:?}", self.capture, self.op)
    }
}
