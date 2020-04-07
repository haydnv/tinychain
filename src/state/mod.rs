use std::fmt;
use std::sync::Arc;

use crate::context::TCResult;
use crate::error;

pub mod block;
pub mod chain;
pub mod graph;
pub mod table;
pub mod tensor;

#[derive(Hash)]
pub enum TCState {
    Block(Arc<block::Block>),
    Chain(Arc<chain::Chain>),
    Graph(Arc<graph::Graph>),
    Table(Arc<table::Table>),
    Tensor(Arc<tensor::Tensor>),
}

impl TCState {
    pub fn from_block(block: Arc<block::Block>) -> Arc<TCState> {
        Arc::new(TCState::Block(block))
    }

    pub fn from_chain(chain: Arc<chain::Chain>) -> Arc<TCState> {
        Arc::new(TCState::Chain(chain))
    }

    pub fn from_table(table: Arc<table::Table>) -> Arc<TCState> {
        Arc::new(TCState::Table(table))
    }

    pub fn to_block(self: Arc<Self>) -> TCResult<Arc<block::Block>> {
        match &*self {
            TCState::Block(block) => Ok(block.clone()),
            other => Err(error::bad_request("Expected block but found", other)),
        }
    }

    pub fn to_chain(self: Arc<Self>) -> TCResult<Arc<chain::Chain>> {
        match &*self {
            TCState::Chain(chain) => Ok(chain.clone()),
            other => Err(error::bad_request("Expected chain but found", other)),
        }
    }
}

impl fmt::Display for TCState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCState::Block(_) => write!(f, "(block)"),
            TCState::Chain(_) => write!(f, "(chain)"),
            TCState::Graph(_) => write!(f, "(graph)"),
            TCState::Table(_) => write!(f, "(table)"),
            TCState::Tensor(_) => write!(f, "(tensor)"),
        }
    }
}
