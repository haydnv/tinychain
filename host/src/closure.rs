//! An [`OpDef`] which closes over zero or more [`State`]s

use async_hash::Hash;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::stream::{FuturesUnordered, TryStreamExt};
use log::debug;
use safecast::{CastInto, TryCastInto};
use sha2::digest::{Digest, Output};
use sha2::Sha256;

use tc_error::*;
use tc_transact::IntoView;
use tcgeneric::{Id, Instance, Map, PathSegment, TCPathBuf, Tuple};

use crate::fs;
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler};
use crate::scalar::{Executor, OpDef, OpDefType, OpRef, Scalar, SELF};
use crate::state::{State, StateView};
use crate::txn::Txn;

/// An [`OpDef`] which closes over zero or more [`State`]s
#[derive(Clone)]
pub struct Closure {
    context: Map<State>,
    op: OpDef,
}

impl Closure {
    /// Construct a new `Closure`.
    pub fn new(context: Map<State>, op: OpDef) -> Self {
        Self { context, op }
    }

    /// Compute the SHA256 hash of this `Closure`.
    pub async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        let context = State::Map(self.context).hash(txn).await?;

        let mut hasher = Sha256::default();
        hasher.update(&context);
        hasher.update(&Hash::<Sha256>::hash(self.op));
        Ok(hasher.finalize())
    }

    /// Return the context and [`OpDef`] which define this `Closure`.
    pub fn into_inner(self) -> (Map<State>, OpDef) {
        (self.context, self.op)
    }

    /// Replace references to `$self` with the given `path`.
    pub fn dereference_self(self, path: &TCPathBuf) -> Self {
        let mut context = self.context;
        context.remove(&SELF.into());

        let op = self.op.dereference_self(path);

        Self { context, op }
    }

    /// Return `true` if this `Closure` may write to service other than where it's defined
    pub fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.op.is_inter_service_write(cluster_path)
    }

    /// Replace references to the given `path` with `$self`
    pub fn reference_self(self, path: &TCPathBuf) -> Self {
        let before = self.op.clone();
        let op = self.op.reference_self(path);

        let context = if op == before {
            self.context
        } else {
            let op_ref = OpRef::Get((path.clone().into(), Scalar::default()));
            let mut context = self.context;
            context.insert(SELF.into(), op_ref.into());
            context
        };

        Self { context, op }
    }

    /// Execute this `Closure` with the given `args`
    pub async fn call(self, txn: &Txn, args: State) -> TCResult<State> {
        let capture = if let Some(capture) = self.op.last().cloned() {
            capture
        } else {
            return Ok(State::default());
        };

        let mut context = self.context;
        let subject = context.remove(&SELF.into());

        debug!("call Closure with state {} and args {}", context, args);

        match self.op {
            OpDef::Get((key_name, op_def)) => {
                let key = args.try_cast_into(|s| TCError::bad_request("invalid key", s))?;
                context.insert(key_name, key);

                Executor::with_context(txn, subject.as_ref(), context, op_def)
                    .capture(capture)
                    .await
            }
            OpDef::Put((key_name, value_name, op_def)) => {
                let (key, value) = args
                    .try_cast_into(|s| TCError::bad_request("invalid arguments for PUT Op", s))?;

                context.insert(key_name, key);
                context.insert(value_name, value);

                Executor::with_context(txn, subject.as_ref(), context, op_def)
                    .capture(capture)
                    .await
            }
            OpDef::Post(op_def) => {
                let params: Map<State> = args.try_into()?;
                context.extend(params);

                Executor::with_context(txn, subject.as_ref(), context, op_def)
                    .capture(capture)
                    .await
            }
            OpDef::Delete((key_name, op_def)) => {
                let key = args.try_cast_into(|s| TCError::bad_request("invalid key", s))?;
                context.insert(key_name, key);

                Executor::with_context(txn, subject.as_ref(), context, op_def)
                    .capture(capture)
                    .await
            }
        }
    }

    /// Execute this `Closure` with an owned [`Txn`] and the given `args`.
    pub async fn call_owned(self, txn: Txn, args: State) -> TCResult<State> {
        self.call(&txn, args).await
    }
}

impl<'a> Handler<'a> for Closure {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if self.op.class() == OpDefType::Get {
            Some(Box::new(|txn, key| Box::pin(self.call(txn, key.into()))))
        } else {
            None
        }
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if self.op.class() == OpDefType::Put {
            Some(Box::new(|txn, key, value| {
                Box::pin(self.call(txn, (key, value).cast_into()).map_ok(|_| ()))
            }))
        } else {
            None
        }
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if self.op.class() == OpDefType::Post {
            Some(Box::new(|txn, params| {
                Box::pin(self.call(txn, params.into()))
            }))
        } else {
            None
        }
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if self.op.class() == OpDefType::Delete {
            Some(Box::new(|txn, key| {
                Box::pin(self.call(txn, key.into()).map_ok(|_| ()))
            }))
        } else {
            None
        }
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Closure {
    type Txn = Txn;
    type View = (HashMap<Id, StateView<'en>>, OpDef);

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let mut context = HashMap::with_capacity(self.context.len());
        let mut resolvers: FuturesUnordered<_> = self
            .context
            .into_iter()
            .map(|(id, state)| state.into_view(txn.clone()).map_ok(|view| (id, view)))
            .collect();

        while let Some((id, state)) = resolvers.try_next().await? {
            context.insert(id, state);
        }

        Ok((context, self.op))
    }
}

#[async_trait]
impl de::FromStream for Closure {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(ClosureVisitor { txn }).await
    }
}

impl From<OpDef> for Closure {
    fn from(op: OpDef) -> Self {
        Self {
            context: Map::default(),
            op,
        }
    }
}

impl fmt::Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "closure over {:?}: {:?}", self.context, self.op)
    }
}

impl fmt::Display for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let context: Tuple<&Id> = self.context.keys().collect();
        write!(f, "closure over {}: {}", context, self.op)
    }
}

struct ClosureVisitor {
    txn: Txn,
}

#[async_trait]
impl de::Visitor for ClosureVisitor {
    type Value = Closure;

    fn expecting() -> &'static str {
        "a Closure"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let context = match seq.next_element(self.txn).await? {
            Some(State::Map(context)) => Ok(context),
            Some(other) => Err(de::Error::invalid_type(other, "a Closure context")),
            None => Err(de::Error::invalid_length(0, "a Closure context and Op")),
        }?;

        let op = seq.next_element(()).await?;
        let op = op.ok_or_else(|| de::Error::invalid_length(1, "a Closure Op"))?;
        Ok(Closure { context, op })
    }
}
