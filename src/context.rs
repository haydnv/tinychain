use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::host::Host;
use crate::state::block::Block;
use crate::state::chain::Chain;
use crate::state::table::Table;
use crate::transaction::Transaction;

pub type TCResult<T> = Result<T, error::TCError>;

#[derive(Clone, Deserialize, Serialize, Hash)]
pub enum TCValue {
    Int32(i32),
    Link(String),
    r#String(String),
    Vector(Vec<TCValue>),
}

impl TCValue {
    pub fn link_string(&self) -> TCResult<String> {
        match self {
            TCValue::Link(s) => Ok(s.clone()),
            other => Err(error::bad_request("Expected link but found", other)),
        }
    }

    pub fn vector(&self) -> TCResult<Vec<TCValue>> {
        match self {
            TCValue::Vector(vec) => Ok(vec.clone()),
            other => Err(error::bad_request("Expected vector but found", other)),
        }
    }
}

impl fmt::Debug for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: tc://{}", l),
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector of length {}", v.len()),
        }
    }
}

#[derive(Hash)]
pub enum TCState {
    Block(Arc<Block>),
    Chain(Arc<Chain>),
    Table(Arc<Table>),
    Value(TCValue),
}

impl fmt::Display for TCState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCState::Block(_) => write!(f, "(block)"),
            TCState::Chain(_) => write!(f, "(chain)"),
            TCState::Table(_) => write!(f, "(table)"),
            TCState::Value(v) => write!(f, "value: {}", v),
        }
    }
}

#[async_trait]
pub trait TCContext: Send + Sync {
    async fn get(self: Arc<Self>, _path: String) -> TCResult<Arc<TCState>> {
        Err(error::method_not_allowed())
    }

    async fn post(
        self: Arc<Self>,
        _host: Arc<Host>,
        _method: String,
        _txn: Arc<Transaction>,
    ) -> TCResult<Arc<TCState>> {
        Err(error::method_not_allowed())
    }
}
