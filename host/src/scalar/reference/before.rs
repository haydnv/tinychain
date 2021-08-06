//! Make certain to resolve a `TCRef` before a given other reference is resolved.

use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::TCResult;
use tcgeneric::{Id, Instance, PathSegment, TCPathBuf};

use crate::route::Public;
use crate::scalar::{Scalar, Scope};
use crate::state::{State, ToState};
use crate::txn::Txn;

use super::Refer;

/// Struct to resolve some preliminary reference before a given other is resolved.
#[derive(Clone, Eq, PartialEq)]
pub struct Before {
    when: Scalar,
    then: Scalar,
}

#[async_trait]
impl Refer for Before {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            when: self.when.dereference_self(path),
            then: self.then.dereference_self(path),
        }
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
        debug!("Before::resolve {} from context ()", self);
        let resolved = self.then.resolve(context, txn).await?;
        self.when.resolve(context, txn).await?;
        Ok(resolved)
    }
}

impl TryCastFrom<Scalar> for Before {
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
impl de::FromStream for Before {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (when, then) =
            <(Scalar, Scalar) as de::FromStream>::from_stream(context, decoder).await?;

        Ok(Self { when, then })
    }
}

impl<'en> en::IntoStream<'en> for Before {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.when, self.then).into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for Before {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.when, &self.then), encoder)
    }
}

impl fmt::Debug for Before {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ensure {:?} before {:?}", self.when, self.then)
    }
}

impl fmt::Display for Before {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ensure {} before {}", self.when, self.then)
    }
}
