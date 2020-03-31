use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::table::Table;
use crate::transaction::Transaction;

pub type TCResult<T> = Result<T, error::TCError>;

#[derive(Clone, Deserialize, Serialize, Hash)]
pub enum TCValue {
    Int32(i32),
    r#String(String),
    Vector(Vec<TCValue>),
}

impl fmt::Display for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector of length {}", v.len()),
        }
    }
}

#[derive(Hash)]
pub enum TCState {
    Value(TCValue),
    Table(Arc<Table>),
}

impl fmt::Display for TCState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
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

    fn post(self: Arc<Self>, _method: String, _txn: Arc<Transaction>) -> TCResult<Arc<TCState>> {
        Err(error::method_not_allowed())
    }
}
