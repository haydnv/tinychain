//! Limits the execution scope of an inline [`Op`].

use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tcgeneric::{Id, Instance, Map, PathSegment, TCPathBuf};

use crate::route::Public;
use crate::scalar::{Executor, OpDef, Scalar, Scope};
use crate::state::State;
use crate::txn::Txn;

use super::Refer;

/// A conditional reference.
#[derive(Clone, Eq, PartialEq)]
pub struct With {
    op_context: Scalar,
    op: OpDef,
}

#[async_trait]
impl Refer for With {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            op_context: self.op_context.dereference_self(path),
            op: self.op.dereference_self(path),
        }
    }

    fn is_derived_write(&self) -> bool {
        self.op_context.is_derived_write() || self.op.is_derived_write()
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.op_context.is_inter_service_write(cluster_path)
            || self.op.is_inter_service_write(cluster_path)
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        Self {
            op_context: self.op_context.reference_self(path),
            op: self.op.reference_self(path),
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        self.op_context.requires(deps);
    }

    async fn resolve<'a, T: Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        let capture = if let Some(capture) = self.op.last() {
            capture.clone()
        } else {
            return Ok(State::default());
        };

        let op_context = self.op_context.resolve(context, txn).await?;

        let (params, op_def) = match self.op {
            OpDef::Get((key_name, op_def)) | OpDef::Delete((key_name, op_def)) => {
                let mut params = Map::new();
                params.insert(key_name, op_context);
                (params, op_def)
            }
            OpDef::Put((key_name, value_name, op_def)) => {
                let (key, value) = op_context
                    .try_cast_into(|s| TCError::bad_request("invalid params for PUT Op", s))?;

                let mut params = Map::new();
                params.insert(key_name, key);
                params.insert(value_name, value);

                (params, op_def)
            }
            OpDef::Post(op_def) => {
                let params = op_context
                    .try_cast_into(|s| TCError::bad_request("invalid params for POST Op", s))?;

                (params, op_def)
            }
        };

        Executor::with_context(txn, &State::default(), params, op_def)
            .capture(capture)
            .await
    }
}

impl TryCastFrom<Scalar> for With {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<(Scalar, Scalar)>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        scalar.opt_cast_into().map(|(subject, op)| Self {
            op_context: subject,
            op,
        })
    }
}

#[async_trait]
impl de::FromStream for With {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(context, decoder)
            .map_ok(|(subject, op)| Self {
                op_context: subject,
                op,
            })
            .await
    }
}

impl<'en> en::IntoStream<'en> for With {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.op_context, self.op).into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for With {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.op_context, &self.op), encoder)
    }
}

impl fmt::Debug for With {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "with {:?}: {:?}", self.op_context, self.op)
    }
}

impl fmt::Display for With {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "with {}: {}", self.op_context, self.op)
    }
}
