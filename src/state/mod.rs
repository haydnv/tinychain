use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::file::File;
use crate::object::Object;
use crate::transaction::{Transact, Transaction};
use crate::value::{Args, PathSegment, TCResult, TCValue};

mod graph;
mod schema;
mod table;

pub type Graph = graph::Graph;
pub type Table = table::Table;

#[async_trait]
pub trait Collection {
    type Key: TryFrom<TCValue>;
    type Value: TryFrom<TCValue>;

    async fn get(self: &Arc<Self>, txn: Arc<Transaction>, key: &Self::Key)
        -> TCResult<Self::Value>;

    async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        key: Self::Key,
        state: Self::Value,
    ) -> TCResult<Arc<Self>>;
}

#[async_trait]
pub trait Persistent: Collection + File {
    type Config: TryFrom<TCValue>;

    async fn create(txn: Arc<Transaction>, config: Self::Config) -> TCResult<Arc<Self>>;
}

#[derive(Clone)]
pub enum State {
    Graph(Arc<Graph>),
    Table(Arc<Table>),
    Object(Object),
    Value(TCValue),
}

impl State {
    pub async fn get(&self, txn: Arc<Transaction>, key: TCValue) -> TCResult<State> {
        match self {
            State::Graph(g) => Ok(g.clone().get(txn, &key).await?.into()),
            State::Table(t) => Ok(t.clone().get(txn, &key.try_into()?).await?.into()),
            _ => Err(error::bad_request("Cannot GET from", self)),
        }
    }

    pub fn is_value(&self) -> bool {
        match self {
            State::Value(_) => true,
            _ => false,
        }
    }

    pub async fn put(
        &self,
        txn: Arc<Transaction>,
        key: TCValue,
        value: TCValue,
    ) -> TCResult<State> {
        match self {
            State::Graph(g) => Ok(g.clone().put(txn, key, value).await?.into()),
            State::Table(t) => Ok(t
                .clone()
                .put(txn, key.try_into()?, value.try_into()?)
                .await?
                .into()),
            _ => Err(error::bad_request("Cannot PUT to", self)),
        }
    }

    pub async fn post(
        &self,
        txn: Arc<Transaction>,
        method: &PathSegment,
        args: Args,
    ) -> TCResult<State> {
        match self {
            State::Object(o) => o.post(txn, method, args).await,
            other => Err(error::method_not_allowed(format!(
                "{} does not support POST",
                other
            ))),
        }
    }
}

impl From<Arc<Graph>> for State {
    fn from(graph: Arc<Graph>) -> State {
        State::Graph(graph)
    }
}

impl From<Arc<Table>> for State {
    fn from(table: Arc<Table>) -> State {
        State::Table(table)
    }
}

impl<T: Into<TCValue>> From<T> for State {
    fn from(value: T) -> State {
        State::Value(value.into())
    }
}

impl TryFrom<State> for Vec<TCValue> {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Vec<TCValue>> {
        match state {
            State::Value(value) => Ok(value.try_into()?),
            other => Err(error::bad_request("Expected a Value but found", other)),
        }
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            State::Graph(_) => write!(f, "(graph)"),
            State::Table(_) => write!(f, "(table)"),
            State::Object(object) => write!(f, "(object: {})", object.class()),
            State::Value(value) => write!(f, "value: {}", value),
        }
    }
}
