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

pub mod btree;
mod dir;
pub mod file;
pub mod graph;
mod table;
pub mod tensor;

pub type Dir = dir::Dir;
pub type GetResult = TCResult<TCStream<State>>;

#[async_trait]
pub trait Collect: Transact + Send + Sync {
    type Selector: Clone + TryFrom<Value, Error = error::TCError> + Send + Sync + 'static;

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
    Index(Arc<btree::BTree>),
    Value(Value),
}

impl From<Arc<btree::BTree>> for State {
    fn from(i: Arc<btree::BTree>) -> State {
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
