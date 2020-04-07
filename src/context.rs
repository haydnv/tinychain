use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

const LINK_BLACKLIST: [&str; 11] = ["..", "~", "$", "&", "?", "|", "{", "}", "//", ":", "="];

pub type TCResult<T> = Result<T, error::TCError>;

#[derive(Clone, Hash)]
pub enum TCResponse {
    State(Arc<TCState>),
    Value(TCValue),
}

impl TCResponse {
    pub fn to_state(&self) -> TCResult<Arc<TCState>> {
        match self {
            TCResponse::State(state) => Ok(state.clone()),
            other => Err(error::bad_request("Expected state but found", other)),
        }
    }
}

impl fmt::Display for TCResponse {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCResponse::State(state) => write!(f, "{}", state),
            TCResponse::Value(value) => write!(f, "{}", value),
        }
    }
}

#[async_trait]
pub trait TCContext: Send + Sync {
    async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<TCResponse>;

    async fn put(self: Arc<Self>, txn: Arc<Transaction>, value: TCValue) -> TCResult<()>;
}

#[async_trait]
pub trait TCExecutable: Send + Sync {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: Link) -> TCResult<TCResponse>;
}
