use std::collections::HashSet;
use std::convert::TryFrom;
use std::fmt;
use std::ops::Deref;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use get_size::GetSize;
use get_size_derive::*;
use safecast::{Match, TryCastFrom};

use tc_error::*;
use tcgeneric::{Id, Instance, PathSegment, TCPathBuf, Tuple};

use crate::route::Public;
use crate::scalar::{Scalar, Scope, Value};
use crate::state::{State, ToState};
use crate::txn::Txn;

use super::{Refer, TCRef};

/// A switch-case flow control
#[derive(Clone, Eq, PartialEq, GetSize)]
pub struct Case {
    cond: TCRef,
    switch: Tuple<Scalar>,
    case: Tuple<Scalar>,
}

#[async_trait]
impl Refer for Case {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        Self {
            cond: self.cond.dereference_self(path),

            switch: self
                .switch
                .into_iter()
                .map(|scalar| scalar.dereference_self(path))
                .collect(),

            case: self
                .case
                .into_iter()
                .map(|scalar| scalar.dereference_self(path))
                .collect(),
        }
    }

    fn is_conditional(&self) -> bool {
        true
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.cond.is_inter_service_write(cluster_path)
            || self
                .switch
                .iter()
                .any(|scalar| scalar.is_inter_service_write(cluster_path))
            || self
                .case
                .iter()
                .any(|scalar| scalar.is_inter_service_write(cluster_path))
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        Self {
            cond: self.cond.reference_self(path),

            switch: self
                .switch
                .into_iter()
                .map(|scalar| scalar.reference_self(path))
                .collect(),

            case: self
                .case
                .into_iter()
                .map(|scalar| scalar.reference_self(path))
                .collect(),
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        self.cond.requires(deps);

        for switch in self.switch.iter() {
            switch.requires(deps);
        }
    }

    async fn resolve<'a, T: ToState + Public + Instance>(
        mut self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        assert_eq!(self.switch.len() + 1, self.case.len());

        if self.cond.is_conditional() {
            return Err(bad_request!(
                "Case does not allow a nested conditional {:?}",
                self.cond,
            ));
        }

        for switch in self.switch.iter() {
            if switch.is_conditional() {
                return Err(bad_request!(
                    "Case does not allow a nested conditional {:?}",
                    switch,
                ));
            }
        }

        let cond = self.cond.resolve(context, txn).await?;
        let cond = Value::try_from(cond)?;
        for (i, switch) in self.switch.into_iter().enumerate() {
            let switch = switch.resolve(context, txn).await?;
            let switch = Value::try_from(switch)?;
            if cond == switch {
                return Ok(self.case.remove(i).into());
            }
        }

        Ok(self.case.pop().unwrap().into())
    }
}

impl<'a, D: Digest> Hash<D> for &'a Case {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((&self.cond, self.switch.deref(), self.case.deref()))
    }
}

impl TryCastFrom<Scalar> for Case {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<(TCRef, Tuple<Scalar>, Tuple<Scalar>)>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        if let Some((cond, switch, case)) =
            <(TCRef, Tuple<Scalar>, Tuple<Scalar>)>::opt_cast_from(scalar)
        {
            if case.len() == switch.len() + 1 {
                Some(Case { cond, switch, case })
            } else {
                None
            }
        } else {
            None
        }
    }
}

#[async_trait]
impl de::FromStream for Case {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (cond, switch, case) =
            <(TCRef, Tuple<Scalar>, Tuple<Scalar>) as de::FromStream>::from_stream(
                context, decoder,
            )
            .await?;

        if case.len() == switch.len() + 1 {
            Ok(Self { cond, switch, case })
        } else {
            Err(de::Error::custom(
                "case length must equal switch length plus one",
            ))
        }
    }
}

impl<'en> en::IntoStream<'en> for Case {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.cond, self.switch.into_inner(), self.case.into_inner()).into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for Case {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            (&self.cond, self.switch.deref(), self.case.deref()),
            encoder,
        )
    }
}

impl fmt::Debug for Case {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "switch ({:?})...", self.cond)
    }
}
