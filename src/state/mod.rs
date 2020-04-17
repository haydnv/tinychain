use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use crate::context::{TCContext, TCExecutable, TCResult};
use crate::error;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

mod chain;
mod dir;
mod graph;
mod table;
mod tensor;

pub type Chain = chain::Chain;
pub type ChainContext = chain::ChainContext;
pub type Dir = dir::Dir;
pub type Table = table::Table;
pub type TableContext = table::TableContext;

#[derive(Clone, Hash)]
pub enum TCState {
    Chain(Arc<Chain>),
    Dir(Arc<Dir>),
    Graph(Arc<graph::Graph>),
    Table(Arc<Table>),
    Tensor(Arc<tensor::Tensor>),
    Value(TCValue),
}

impl TCState {
    pub async fn get(&self, txn: Arc<Transaction>, key: TCValue) -> TCResult<TCState> {
        match self {
            TCState::Chain(c) => c.clone().get(txn, key).await,
            TCState::Dir(d) => d.clone().get(txn, key).await,
            TCState::Graph(g) => g.clone().get(txn, key).await,
            TCState::Table(t) => t.clone().get(txn, key).await,
            TCState::Tensor(t) => t.clone().get(txn, key).await,
            _ => Err(error::bad_request("Cannot GET from", self)),
        }
    }

    pub async fn put(
        &self,
        txn: Arc<Transaction>,
        key: TCValue,
        value: TCState,
    ) -> TCResult<TCState> {
        match self {
            TCState::Chain(c) => c.clone().put(txn, key, value).await,
            TCState::Dir(d) => d.clone().put(txn, key, value).await,
            TCState::Graph(g) => g.clone().put(txn, key, value).await,
            TCState::Table(t) => t.clone().put(txn, key, value).await,
            TCState::Tensor(t) => t.clone().put(txn, key, value).await,
            _ => Err(error::bad_request("Cannot PUT to", self)),
        }
    }

    pub async fn post(&self, txn: Arc<Transaction>, action: &Link) -> TCResult<TCState> {
        match self {
            TCState::Graph(g) => g.clone().post(txn, action).await,
            TCState::Tensor(t) => t.clone().post(txn, action).await,
            _ => Err(error::bad_request("Cannot POST to", self)),
        }
    }
}

impl From<()> for TCState {
    fn from(_: ()) -> TCState {
        TCState::Value(TCValue::None)
    }
}

impl From<Arc<Chain>> for TCState {
    fn from(chain: Arc<Chain>) -> TCState {
        TCState::Chain(chain)
    }
}

impl From<&Arc<Table>> for TCState {
    fn from(table: &Arc<Table>) -> TCState {
        TCState::Table(table.clone())
    }
}

impl From<TCValue> for TCState {
    fn from(value: TCValue) -> TCState {
        TCState::Value(value)
    }
}

impl From<&TCValue> for TCState {
    fn from(value: &TCValue) -> TCState {
        TCState::Value(value.clone())
    }
}

impl From<Vec<TCValue>> for TCState {
    fn from(value: Vec<TCValue>) -> TCState {
        TCState::Value(value.into())
    }
}

impl TryFrom<TCState> for Arc<Chain> {
    type Error = error::TCError;

    fn try_from(state: TCState) -> TCResult<Arc<Chain>> {
        match state {
            TCState::Chain(chain) => Ok(chain),
            other => Err(error::bad_request("Expected a Chain but found", other)),
        }
    }
}

impl TryFrom<TCState> for Vec<TCValue> {
    type Error = error::TCError;

    fn try_from(state: TCState) -> TCResult<Vec<TCValue>> {
        match state {
            TCState::Value(value) => Ok(value.try_into()?),
            other => Err(error::bad_request("Expected a Value but found", other)),
        }
    }
}

impl fmt::Display for TCState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCState::Chain(_) => write!(f, "(chain)"),
            TCState::Dir(_) => write!(f, "(dir)"),
            TCState::Graph(_) => write!(f, "(graph)"),
            TCState::Table(_) => write!(f, "(table)"),
            TCState::Tensor(_) => write!(f, "(tensor)"),
            TCState::Value(value) => write!(f, "value: {}", value),
        }
    }
}
