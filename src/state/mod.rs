use std::fmt;
use std::sync::Arc;

use crate::context::TCResult;
use crate::error;
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
    pub fn as_chain(&self) -> TCResult<Arc<chain::Chain>> {
        match self {
            TCState::Chain(chain) => Ok(chain.clone()),
            other => Err(error::bad_request("Expected chain but found", other)),
        }
    }

    pub fn as_value(&self) -> TCResult<TCValue> {
        match self {
            TCState::Value(value) => Ok(value.clone()),
            other => Err(error::bad_request("Expected value but found", other)),
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
