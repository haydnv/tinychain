//! Delay resolving a `TCRef` until a given dependency is resolved.

use async_hash::Hash;
use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};
use sha2::digest::{Digest, Output};

use tc_error::*;
use tcgeneric::{Id, Instance, PathSegment, TCPathBuf};

use crate::route::Public;
use crate::scalar::{Scalar, Scope};
use crate::state::{State, ToState};
use crate::txn::Txn;

use super::Refer;

/// Struct to delay resolving another reference(s) until some preliminary reference is resolved.
#[derive(Clone, Eq, PartialEq)]
pub struct After {
    when: Scalar,
    then: Scalar,
}

#[async_trait]
impl Refer for After {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            when: self.when.dereference_self(path),
            then: self.then.dereference_self(path),
        }
    }

    fn is_conditional(&self) -> bool {
        self.then.is_conditional()
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.when.is_inter_service_write(cluster_path)
            || self.then.is_inter_service_write(cluster_path)
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        Self {
            when: self.when.reference_self(path),
            then: self.then.reference_self(path),
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        self.when.requires(deps);
        self.then.requires(deps);
    }

    async fn resolve<'a, T: ToState + Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        debug!("After::resolve {} from context ()", self);
        if self.when.is_conditional() {
            return Err(TCError::bad_request(
                "After does not allow a conditional clause",
                self.when,
            ));
        }

        self.when.resolve(context, txn).await?;
        self.then.resolve(context, txn).await
    }
}

impl<'a, D: Digest> Hash<D> for &'a After {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((&self.when, &self.then))
    }
}

impl TryCastFrom<Scalar> for After {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<(Scalar, Scalar)>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        scalar
            .opt_cast_into()
            .map(|(when, then)| Self { when, then })
    }
}

#[async_trait]
impl de::FromStream for After {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (when, then) =
            <(Scalar, Scalar) as de::FromStream>::from_stream(context, decoder).await?;

        Ok(Self { when, then })
    }
}

impl<'en> en::IntoStream<'en> for After {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.when, self.then).into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for After {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.when, &self.then), encoder)
    }
}

impl fmt::Debug for After {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "after {:?} then {:?}", self.when, self.then)
    }
}

impl fmt::Display for After {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "after {} then {}", self.when, self.then)
    }
}
