use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::state::block::Block;
use crate::state::chain::Chain;
use crate::state::table::Table;
use crate::transaction::Transaction;

pub type TCResult<T> = Result<T, error::TCError>;

#[derive(Clone, Deserialize, Serialize, Hash)]
pub struct Link {
    to: String,
}

impl Link {
    pub fn to(destination: String) -> TCResult<Link> {
        if !destination.starts_with('/') {
            Err(error::bad_request(
                "Expected an absolute path starting with '/' but found",
                destination,
            ))
        } else {
            Ok(Link { to: destination })
        }
    }

    pub fn from(&self, context: &str) -> TCResult<Link> {
        if self.to.starts_with(context) {
            Link::to(self.to[context.len()..].to_string())
        } else {
            Err(error::bad_request(
                format!("Cannot link {} from", self).as_str(),
                context,
            ))
        }
    }

    pub fn segments(&self) -> Vec<String> {
        self.to[1..].split('/').map(|s| s.to_string()).collect()
    }
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "tc://{}", self.to)
    }
}

#[derive(Clone, Deserialize, Serialize, Hash)]
pub enum TCValue {
    None,
    Int32(i32),
    Link(Link),
    r#String(String),
    Vector(Vec<TCValue>),
}

impl TCValue {
    pub fn link(&self) -> TCResult<Link> {
        match self {
            TCValue::Link(l) => Ok(l.clone()),
            other => Err(error::bad_request("Expected link but found", other)),
        }
    }

    pub fn string(&self) -> TCResult<String> {
        match self {
            TCValue::r#String(s) => Ok(s.clone()),
            other => Err(error::bad_request("Expected string but found", other)),
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
            TCValue::None => write!(f, "None"),
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: {}", l),
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

impl TCState {
    pub fn chain(self: Arc<Self>) -> TCResult<Arc<Chain>> {
        match &*self {
            TCState::Chain(chain) => Ok(chain.clone()),
            other => Err(error::bad_request("Expected chain but found", other)),
        }
    }

    pub fn from_chain(chain: Arc<Chain>) -> Arc<TCState> {
        Arc::new(TCState::Chain(chain))
    }

    pub fn from_string(s: String) -> Arc<TCState> {
        Arc::new(TCState::Value(TCValue::r#String(s)))
    }

    pub fn none() -> Arc<TCState> {
        Arc::new(TCState::Value(TCValue::None))
    }

    pub fn value(self: Arc<Self>) -> TCResult<TCValue> {
        match &*self {
            TCState::Value(val) => Ok(val.clone()),
            other => Err(error::bad_request("Expected value but found", other)),
        }
    }
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
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _path: Link) -> TCResult<Arc<TCState>> {
        Err(error::method_not_allowed())
    }

    async fn post(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        _method: &str,
    ) -> TCResult<Arc<TCState>> {
        Err(error::method_not_allowed())
    }
}
