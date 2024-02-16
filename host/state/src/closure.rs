//! An [`OpDef`] which closes over zero or more [`State`]s

use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::marker::PhantomData;

use async_hash::{Digest, Hash, Output, Sha256};
use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::stream::{FuturesUnordered, TryStreamExt};
use log::debug;
use safecast::{CastInto, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::{Executor, OpDef, OpDefType, OpRef, Scalar, SELF};
use tc_transact::public::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler};
use tc_transact::{AsyncHash, Gateway, IntoView, Transaction, TxnId};
use tcgeneric::{Id, Instance, Map, PathSegment, TCPathBuf};

use super::view::StateView;
use super::{CacheBlock, State};

/// An [`OpDef`] which closes over zero or more [`State`]s
pub struct Closure<Txn> {
    context: Map<State<Txn>>,
    op: OpDef,
}

impl<Txn> Clone for Closure<Txn> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            op: self.op.clone(),
        }
    }
}

impl<Txn> Closure<Txn> {
    /// Return the context and [`OpDef`] which define this `Closure`.
    pub fn into_inner(self) -> (Map<State<Txn>>, OpDef) {
        (self.context, self.op)
    }
}

impl<Txn> Closure<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    /// Replace references to `$self` with the given `path`.
    pub fn dereference_self(self, path: &TCPathBuf) -> Self {
        let mut context = self.context;
        context.remove::<Id>(&SELF.into());

        let op = self.op.dereference_self::<State<Txn>>(path);

        Self { context, op }
    }

    /// Return `true` if this `Closure` may write to service other than where it's defined
    pub fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.op.is_inter_service_write::<State<Txn>>(cluster_path)
    }

    /// Replace references to the given `path` with `$self`
    pub fn reference_self(self, path: &TCPathBuf) -> Self {
        let before = self.op.clone();
        let op = self.op.reference_self::<State<Txn>>(path);

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
    pub async fn call(self, txn: &Txn, args: State<Txn>) -> TCResult<State<Txn>> {
        let capture = if let Some(capture) = self.op.last().cloned() {
            capture
        } else {
            return Ok(State::default());
        };

        let mut context = self.context;
        let subject = context.remove::<Id>(&SELF.into());

        debug!("call Closure with state {:?} and args {:?}", context, args);

        match self.op {
            OpDef::Get((key_name, op_def)) => {
                let key = args.try_cast_into(|s| TCError::unexpected(s, "a Value"))?;

                context.insert(key_name, key);

                Executor::with_context(txn, subject.as_ref(), context, op_def)
                    .capture(capture)
                    .await
            }
            OpDef::Put((key_name, value_name, op_def)) => {
                let (key, value) =
                    args.try_cast_into(|s| TCError::unexpected(s, "arguments for PUT Op"))?;

                context.insert(key_name, key);
                context.insert(value_name, value);

                Executor::with_context(txn, subject.as_ref(), context, op_def)
                    .capture(capture)
                    .await
            }
            OpDef::Post(op_def) => {
                let params: Map<State<Txn>> = args.try_into()?;
                context.extend(params);

                Executor::with_context(txn, subject.as_ref(), context, op_def)
                    .capture(capture)
                    .await
            }
            OpDef::Delete((key_name, op_def)) => {
                let key = args.try_cast_into(|s| TCError::unexpected(s, "a Value"))?;
                context.insert(key_name, key);

                Executor::with_context(txn, subject.as_ref(), context, op_def)
                    .capture(capture)
                    .await
            }
        }
    }

    /// Execute this `Closure` with an owned [`Txn`] and the given `args`.
    pub async fn call_owned(self, txn: Txn, args: State<Txn>) -> TCResult<State<Txn>> {
        self.call(&txn, args).await
    }
}

#[async_trait]
impl<Txn> tc_transact::public::ClosureInstance<State<Txn>> for Closure<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    async fn call(self: Box<Self>, txn: Txn, args: State<Txn>) -> TCResult<State<Txn>> {
        self.call_owned(txn, args).await
    }
}

impl<Txn> From<(Map<State<Txn>>, OpDef)> for Closure<Txn> {
    fn from(tuple: (Map<State<Txn>>, OpDef)) -> Self {
        let (context, op) = tuple;

        Self { context, op }
    }
}

impl<'a, Txn> Handler<'a, State<Txn>> for Closure<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State<Txn>>>
    where
        'b: 'a,
    {
        if self.op.class() == OpDefType::Get {
            Some(Box::new(|txn, key| Box::pin(self.call(txn, key.into()))))
        } else {
            None
        }
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State<Txn>>>
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State<Txn>>>
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

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
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
impl<Txn> AsyncHash for Closure<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let context = State::Map(self.context).hash(txn_id).await?;

        let mut hasher = Sha256::default();
        hasher.update(&context);
        hasher.update(&Hash::<Sha256>::hash(self.op));
        Ok(hasher.finalize())
    }
}

#[async_trait]
impl<'en, Txn> IntoView<'en, CacheBlock> for Closure<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
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
impl<Txn> de::FromStream for Closure<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder
            .decode_seq(ClosureVisitor {
                txn,
                phantom: PhantomData,
            })
            .await
    }
}

impl<Txn> From<OpDef> for Closure<Txn> {
    fn from(op: OpDef) -> Self {
        Self {
            context: Map::default(),
            op,
        }
    }
}

impl<Txn> TryCastFrom<Scalar> for Closure<Txn> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Op(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Op(op) => Some(Self {
                context: Map::default(),
                op,
            }),
            _ => None,
        }
    }
}

impl<Txn> fmt::Debug for Closure<Txn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "closure over {:?}: {:?}", self.context, self.op)
    }
}

struct ClosureVisitor<Txn> {
    txn: Txn,
    phantom: PhantomData<CacheBlock>,
}

#[async_trait]
impl<Txn> de::Visitor for ClosureVisitor<Txn>
where
    Txn: Transaction<CacheBlock> + Gateway<State<Txn>>,
{
    type Value = Closure<Txn>;

    fn expecting() -> &'static str {
        "a Closure"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let context = match seq.next_element(self.txn).await? {
            Some(State::Map(context)) => Ok(context),
            Some(other) => Err(de::Error::invalid_type(
                format!("{other:?}"),
                "a Closure context",
            )),
            None => Err(de::Error::invalid_length(0, "a Closure context and Op")),
        }?;

        let op = seq.next_element(()).await?;
        let op = op.ok_or_else(|| de::Error::invalid_length(1, "a Closure Op"))?;
        Ok(Closure { context, op })
    }
}
