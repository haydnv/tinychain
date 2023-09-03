//! Delay resolving a `TCRef` until a given dependency is resolved.

use std::collections::HashSet;
use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use get_size::GetSize;
use get_size_derive::*;
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::public::{Public, StateInstance, ToState};
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment, TCPathBuf};

use crate::{OpDef, Scalar, Scope};

use super::Refer;

/// Struct to delay resolving another reference(s) until some preliminary reference is resolved.
#[derive(Clone, Eq, PartialEq, GetSize)]
pub struct After {
    when: Scalar,
    then: Scalar,
}

#[async_trait]
impl<State> Refer<State> for After
where
    State: StateInstance + Refer<State> + From<Scalar>,
    State::Closure: From<(Map<State>, OpDef)> + TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError>,
    Value: TryFrom<State, Error = TCError> + TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
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

    fn is_ref(&self) -> bool {
        true
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

    async fn resolve<'a, T: ToState<State> + Public<State> + Instance>(
        self,
        context: &'a Scope<'a, State, T>,
        txn: &'a State::Txn,
    ) -> TCResult<State> {
        debug!("After::resolve {:?} from context ()", self);
        if self.when.is_conditional() {
            return Err(bad_request!(
                "After does not allow a conditional clause {:?}",
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
