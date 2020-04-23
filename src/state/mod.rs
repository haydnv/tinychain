use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use crate::context::{TCContext, TCResult};
use crate::error;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCValue};

mod graph;
mod table;

pub type Table = table::Table;
pub type TableContext = table::TableContext;

#[derive(Clone)]
pub enum State {
    Graph(Arc<graph::Graph>),
    Table(Arc<Table>),
    Value(TCValue),
}

impl State {
    pub async fn commit(&self, txn_id: TransactionId) {
        match self {
            State::Graph(g) => g.commit(txn_id).await,
            State::Table(t) => t.commit(txn_id).await,
            _ => {
                eprintln!("Tried to commit to a Value!");
            }
        }
    }

    pub async fn get(&self, txn: Arc<Transaction>, key: &TCValue) -> TCResult<State> {
        match self {
            State::Graph(g) => g.clone().get(txn, key).await,
            State::Table(t) => t.clone().get(txn, key).await,
            _ => Err(error::bad_request("Cannot GET from", self)),
        }
    }

    pub async fn put(
        &self,
        txn: Arc<Transaction>,
        key: TCValue,
        value: State,
    ) -> TCResult<State> {
        match self {
            State::Graph(g) => g.clone().put(txn, key, value).await,
            State::Table(t) => t.clone().put(txn, key, value).await,
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
