//! Delay resolving a [`TCRef`] until a given dependency is resolved.

use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use error::TCResult;
use generic::{Id, Instance};

use crate::route::Public;
use crate::scalar::{Scalar, Scope};
use crate::state::State;
use crate::txn::Txn;

use super::{Refer, TCRef};

/// Struct to delay resolving another reference(s) until some preliminary reference is resolved.
#[derive(Clone, Eq, PartialEq)]
pub struct After {
    when: TCRef,
    then: Scalar,
}

#[async_trait]
impl Refer for After {
    fn requires(&self, deps: &mut HashSet<Id>) {
        self.when.requires(deps);
        self.then.requires(deps);
    }

    async fn resolve<T: Instance + Public>(self, context: &Scope<T>, txn: &Txn) -> TCResult<State> {
        debug!("After::resolve {}", self);

        self.when.resolve(context, txn).await?;
        Ok(self.then.into())
    }
}

impl TryCastFrom<Scalar> for After {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<(TCRef, Scalar)>()
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
            <(TCRef, Scalar) as de::FromStream>::from_stream(context, decoder).await?;

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

impl fmt::Display for After {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "after {} then {}", self.when, self.then)
    }
}
