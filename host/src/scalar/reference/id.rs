//! Reference another [`State`] in the same [`Txn`] context.

use async_hash::Hash;
use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de;
use destream::en::{EncodeMap, Encoder, IntoStream, ToStream};
use log::debug;
use safecast::TryCastFrom;
use sha2::digest::{Digest, Output};

use tc_error::*;
use tcgeneric::{Id, Instance, Label, PathSegment, TCPathBuf};

use crate::route::Public;
use crate::scalar::{Scope, Value, SELF};
use crate::state::{State, ToState};
use crate::txn::Txn;

use super::Refer;

const EMPTY_SLICE: &[usize] = &[];

/// A reference to the [`State`] at a given [`Id`] within the same transaction context.
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct IdRef {
    to: Id,
}

impl IdRef {
    pub fn into_id(self) -> Id {
        self.to
    }

    pub fn id(&'_ self) -> &'_ Id {
        &self.to
    }
}

#[async_trait]
impl Refer for IdRef {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        if self.to == SELF {
            panic!("cannot dereference {} to {}", self, path);
        } else {
            self
        }
    }

    fn is_conditional(&self) -> bool {
        false
    }

    fn is_inter_service_write(&self, _cluster_path: &[PathSegment]) -> bool {
        false
    }

    fn reference_self(self, _path: &TCPathBuf) -> Self {
        self
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        if self.to != SELF {
            deps.insert(self.to.clone());
        }
    }

    async fn resolve<'a, T: ToState + Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        _txn: &'a Txn,
    ) -> TCResult<State> {
        debug!("IdRef::resolve {}", self);
        context.resolve_id(self.id())
    }
}

impl PartialEq<Id> for IdRef {
    fn eq(&self, other: &Id) -> bool {
        self.id() == other
    }
}

impl<'a, D: Digest> Hash<D> for &'a IdRef {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self.to)
    }
}

impl From<Label> for IdRef {
    fn from(to: Label) -> Self {
        Self { to: to.into() }
    }
}

impl From<Id> for IdRef {
    fn from(to: Id) -> Self {
        Self { to }
    }
}

impl FromStr for IdRef {
    type Err = TCError;

    #[inline]
    fn from_str(to: &str) -> TCResult<Self> {
        if !to.starts_with('$') || to.len() < 2 {
            Err(TCError::bad_request("invalid IdRef", to))
        } else {
            to[1..]
                .parse()
                .map(|to| IdRef { to })
                .map_err(TCError::unsupported)
        }
    }
}

impl TryCastFrom<Value> for IdRef {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::String(s) => Self::from_str(s).is_ok(),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::String(s) => Self::from_str(&s).ok(),
            _ => None,
        }
    }
}

impl From<IdRef> for Id {
    fn from(r: IdRef) -> Id {
        r.to
    }
}

impl fmt::Debug for IdRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "reference to {:?}", self.to)
    }
}

impl fmt::Display for IdRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "${}", self.to)
    }
}

#[async_trait]
impl de::FromStream for IdRef {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), d: &mut D) -> Result<Self, D::Error> {
        let id_ref = String::from_stream(context, d).await?;
        id_ref.parse().map_err(de::Error::custom)
    }
}

impl<'en> ToStream<'en> for IdRef {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        let mut map = e.encode_map(Some(1))?;
        map.encode_entry(self.to_string(), EMPTY_SLICE)?;
        map.end()
    }
}

impl<'en> IntoStream<'en> for IdRef {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        let mut map = e.encode_map(Some(1))?;
        map.encode_entry(self.to_string(), EMPTY_SLICE)?;
        map.end()
    }
}
