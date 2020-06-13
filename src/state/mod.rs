use std::convert::TryFrom;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::Stream;
use serde::de::DeserializeOwned;
use serde::ser::Serialize;

use crate::error;
use crate::internal::archive::Archive;
use crate::transaction::{Transact, Txn};
use crate::value::{TCResult, Value};

pub mod graph;
pub mod index;

// TODO: can this `Box<dyn...>` be replaced by `impl Stream<...>`?
pub type GetResult = TCResult<Pin<Box<dyn Stream<Item = State> + Send + Sync + Unpin>>>;

#[async_trait]
pub trait Collect: Transact + Send + Sync {
    type Selector: Clone
        + DeserializeOwned
        + Serialize
        + TryFrom<Value, Error = error::TCError>
        + Send
        + Sync
        + 'static;

    type Item: Clone
        + DeserializeOwned
        + Serialize
        + TryFrom<Value, Error = error::TCError>
        + Send
        + Sync
        + 'static;

    async fn get(self: Arc<Self>, txn: Arc<Txn>, selector: Self::Selector) -> GetResult;

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        value: Self::Item,
    ) -> TCResult<()>;
}

#[async_trait]
pub trait Persist: Archive + Collect {}

pub enum State {
    Index(Arc<index::Index>),
    Value(Value),
}

impl State {
    pub async fn get(&self, _txn: &Arc<Txn>, _selector: Value) -> GetResult {
        Err(error::not_implemented())
    }

    pub async fn put(&self, _txn: &Arc<Txn>, _selector: Value, _data: State) -> TCResult<State> {
        Err(error::not_implemented())
    }
}

impl From<Arc<index::Index>> for State {
    fn from(i: Arc<index::Index>) -> State {
        State::Index(i)
    }
}

impl From<Value> for State {
    fn from(v: Value) -> State {
        State::Value(v)
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            State::Index(_) => write!(f, "(index)"),
            State::Value(v) => write!(f, "state: {}", v),
        }
    }
}
