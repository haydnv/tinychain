use std::collections::HashMap;
use std::fmt;

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::stream::{FuturesUnordered, TryStreamExt};

use tc_error::*;
use tc_transact::IntoView;
use tcgeneric::{Id, Map, Tuple};

use crate::fs;
use crate::scalar::{OpDef, Scalar};
use crate::state::{State, StateView};
use crate::txn::Txn;

#[derive(Clone)]
pub struct Closure {
    context: Map<State>,
    op: OpDef,
}

impl Closure {
    pub fn new(context: Map<State>, op: OpDef) -> Self {
        Self { context, op }
    }

    pub fn into_callable(self, state: State) -> TCResult<(Map<State>, Vec<(Id, Scalar)>)> {
        let mut context = self.context;
        let (params, op_def) = self.op.into_callable(state)?;
        context.extend(params);
        Ok((context, op_def))
    }

    pub fn into_inner(self) -> (Map<State>, OpDef) {
        (self.context, self.op)
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
