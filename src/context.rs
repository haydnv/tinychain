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
    pub fn as_str(&self) -> &str {
        &self.to
    }

    pub fn to(destination: &str) -> TCResult<Link> {
        if !destination.starts_with('/') {
            Err(error::bad_request(
                "Expected an absolute path starting with '/' but found",
                destination,
            ))
        } else if destination != "/" && destination.ends_with('/') {
            Err(error::bad_request(
                "Trailing slash is not allowed",
                destination,
            ))
        } else {
            Ok(Link {
                to: destination.to_string(),
            })
        }
    }

    pub fn from(&self, context: &str) -> TCResult<Link> {
        if self.to.starts_with(context) {
            Link::to(&self.to[context.len()..].to_string())
        } else {
            Err(error::bad_request(
                format!("Cannot link {} from", self).as_str(),
                context,
            ))
        }
    }

    pub fn segments(&self) -> Vec<&str> {
        self.to[1..].split('/').collect()
    }
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to)
    }
}

#[derive(Clone, Deserialize, Serialize, Hash)]
pub enum TCValue {
    None,
    Bytes(Vec<u8>),
    Int32(i32),
    Link(Link),
    r#String(String),
    Vector(Vec<TCValue>),
}

impl TCValue {
    pub fn from_bytes(b: Vec<u8>) -> TCValue {
        TCValue::Bytes(b)
    }

    pub fn from_string(s: &str) -> TCValue {
        TCValue::r#String(s.to_string())
    }

    pub fn to_bytes(&self) -> TCResult<Vec<u8>> {
        match self {
            TCValue::Bytes(b) => Ok(b.clone()),
            other => Err(error::bad_request("Expected bytes but found", other)),
        }
    }

    pub fn to_link(&self) -> TCResult<Link> {
        match self {
            TCValue::Link(l) => Ok(l.clone()),
            other => Err(error::bad_request("Expected link but found", other)),
        }
    }

    pub fn to_string(&self) -> TCResult<String> {
        match self {
            TCValue::r#String(s) => Ok(s.clone()),
            other => Err(error::bad_request("Expected string but found", other)),
        }
    }

    pub fn to_vec(&self) -> TCResult<Vec<TCValue>> {
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
            TCValue::Bytes(b) => write!(f, "binary of length {}", b.len()),
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
    pub fn from_block(block: Arc<Block>) -> Arc<TCState> {
        Arc::new(TCState::Block(block))
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

    pub fn to_block(self: Arc<Self>) -> TCResult<Arc<Block>> {
        match &*self {
            TCState::Block(block) => Ok(block.clone()),
            other => Err(error::bad_request("Expected block but found", other)),
        }
    }

    pub fn to_chain(self: Arc<Self>) -> TCResult<Arc<Chain>> {
        match &*self {
            TCState::Chain(chain) => Ok(chain.clone()),
            other => Err(error::bad_request("Expected chain but found", other)),
        }
    }

    pub fn to_value(self: Arc<Self>) -> TCResult<TCValue> {
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

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _value: TCValue) -> TCResult<()> {
        Err(error::method_not_allowed())
    }

    async fn post(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        _method: Link,
    ) -> TCResult<Arc<TCState>> {
        Err(error::method_not_allowed())
    }
}
