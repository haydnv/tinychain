use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use crate::context::{TCContext, TCExecutable, TCResult};
use crate::error;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

pub mod chain;
pub mod dir;
pub mod graph;
pub mod table;
pub mod tensor;

#[derive(Clone, Hash)]
pub enum TCState {
    None,
    Chain(Arc<chain::Chain>),
    Dir(Arc<dir::Dir>),
    Graph(Arc<graph::Graph>),
    Table(Arc<table::Table>),
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
            TCState::Chain(c) => c.clone().post(txn, action).await,
            TCState::Graph(g) => g.clone().post(txn, action).await,
            TCState::Table(t) => t.clone().post(txn, action).await,
            TCState::Tensor(t) => t.clone().post(txn, action).await,
            _ => Err(error::bad_request("Cannot POST to", self)),
        }
    }
}

impl From<()> for TCState {
    fn from(_: ()) -> TCState {
        TCState::None
    }
}

impl From<Arc<chain::Chain>> for TCState {
    fn from(chain: Arc<chain::Chain>) -> TCState {
        TCState::Chain(chain)
    }
}

impl From<Link> for TCState {
    fn from(link: Link) -> TCState {
        TCState::Value(TCValue::Link(link))
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

impl TryFrom<TCState> for Arc<chain::Chain> {
    type Error = error::TCError;

    fn try_from(state: TCState) -> TCResult<Arc<chain::Chain>> {
        match state {
            TCState::Chain(chain) => Ok(chain),
            other => Err(error::bad_request("Expected a Chain but found", other)),
        }
    }
}

impl fmt::Display for TCState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCState::None => write!(f, "None"),
            TCState::Chain(_) => write!(f, "(chain)"),
            TCState::Dir(_) => write!(f, "(dir)"),
            TCState::Graph(_) => write!(f, "(graph)"),
            TCState::Table(_) => write!(f, "(table)"),
            TCState::Tensor(_) => write!(f, "(tensor)"),
            TCState::Value(value) => write!(f, "value: {}", value),
        }
    }
}
