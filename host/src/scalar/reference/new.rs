use std::collections::HashSet;
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::{try_join, TryFutureExt};
use log::debug;
use safecast::*;

use tc_error::*;
use tcgeneric::{Id, Instance, PathSegment, TCPathBuf};

use crate::route::Public;
use crate::scalar::{Scalar, Scope};
use crate::state::{State, ToState};
use crate::txn::Txn;

use super::Refer;
use crate::object::InstanceExt;

#[derive(Clone, Eq, PartialEq)]
pub struct New {
    class: Scalar,
    data: Scalar,
}

#[async_trait]
impl Refer for New {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            class: self.class.dereference_self(path),
            data: self.data.dereference_self(path),
        }
    }

    fn is_conditional(&self) -> bool {
        self.class.is_conditional() || self.data.is_conditional()
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.class.is_inter_service_write(cluster_path)
            || self.data.is_inter_service_write(cluster_path)
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        Self {
            class: self.class.reference_self(path),
            data: self.data.reference_self(path),
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        self.class.requires(deps);
        self.data.requires(deps);
    }

    async fn resolve<'a, T: ToState + Public + Instance>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        let (class, data) = try_join!(
            self.class.resolve(context, txn),
            self.data.resolve(context, txn)
        )?;

        debug!("is {} actually a class?", class);
        let class =
            class.try_cast_into(|s| TCError::bad_request("expected a Class but found", s))?;
        Ok(InstanceExt::new(data, class).to_state())
    }
}

impl TryCastFrom<Scalar> for New {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<(Scalar, Scalar)>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        scalar
            .opt_cast_into()
            .map(|(class, data)| Self { class, data })
    }
}

#[async_trait]
impl de::FromStream for New {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(cxt, decoder)
            .map_ok(|(class, data)| Self { class, data })
            .await
    }
}

impl<'en> en::IntoStream<'en> for New {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((self.class, self.data), encoder)
    }
}

impl<'en> en::ToStream<'en> for New {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.class, &self.data), encoder)
    }
}

impl fmt::Debug for New {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "new instance of {:?} with data {:?}",
            self.class, self.data
        )
    }
}

impl fmt::Display for New {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "new instance of {} with data {}", self.class, self.data)
    }
}
