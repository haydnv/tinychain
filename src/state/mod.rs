use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::file::File;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCResult, TCValue};

mod graph;
mod schema;
mod table;

pub type Graph = graph::Graph;
pub type Table = table::Table;

#[async_trait]
pub trait Collection: Transactable {
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

#[async_trait]
pub trait Transactable: Send + Sync {
    async fn commit(&self, txn_id: &TransactionId);
}

#[derive(Clone)]
pub enum State {
    Graph(Arc<Graph>),
    Table(Arc<Table>),
    Value(TCValue),
}

impl State {
    pub async fn commit(&self, txn_id: &TransactionId) {
        match self {
            State::Graph(g) => g.commit(txn_id).await,
            State::Table(t) => t.commit(txn_id).await,
            _ => {
                panic!("Tried to commit to a non-persistent state: {}", self);
            }
        }
    }

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

    pub async fn post(&self, _txn: Arc<Transaction>, _method: &Link) -> TCResult<State> {
        Err(error::not_implemented())
    }
}

impl From<()> for State {
    fn from(_: ()) -> State {
        State::Value(TCValue::None)
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

impl From<TCValue> for State {
    fn from(value: TCValue) -> State {
        State::Value(value)
    }
}

impl From<&TCValue> for State {
    fn from(value: &TCValue) -> State {
        State::Value(value.clone())
    }
}

impl From<Vec<TCValue>> for State {
    fn from(value: Vec<TCValue>) -> State {
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
            State::Value(value) => write!(f, "value: {}", value),
        }
    }
}
