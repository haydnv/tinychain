use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::ser::Serialize;

use crate::error;
use crate::internal::archive::Archive;
use crate::transaction::{Transact, Txn};
use crate::value::{TCResult, TCStream, Value};

mod dir;
pub mod file;
pub mod graph;
pub mod index;
mod table;

pub type Dir = dir::Dir;

// TODO: can this `Box<dyn...>` be replaced by `impl Stream<...>`?
pub type GetResult = TCResult<TCStream<State>>;

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
