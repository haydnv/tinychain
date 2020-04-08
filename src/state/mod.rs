use std::fmt;
use std::sync::Arc;

use crate::context::TCResult;
use crate::error;
use crate::value::TCValue;

pub mod chain;
pub mod dir;
pub mod graph;
pub mod table;
pub mod tensor;

#[derive(Clone, Hash)]
pub enum TCState {
    Chain(Arc<chain::Chain>),
    Dir(Arc<dir::Dir>),
    Graph(Arc<graph::Graph>),
    Table(Arc<table::Table>),
    Tensor(Arc<tensor::Tensor>),
    Value(TCValue),
}

impl TCState {
    pub fn to_chain(&self) -> TCResult<Arc<chain::Chain>> {
        match self {
            TCState::Chain(chain) => Ok(chain.clone()),
            other => Err(error::bad_request("Expected chain but found", other)),
        }
    }

    pub fn to_value(&self) -> TCResult<TCValue> {
        match self {
            TCState::Value(value) => Ok(value.clone()),
            other => Err(error::bad_request("Expected value but found", other)),
        }
    }
}

impl From<Arc<chain::Chain>> for TCState {
    fn from(chain: Arc<chain::Chain>) -> TCState {
        TCState::Chain(chain)
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
