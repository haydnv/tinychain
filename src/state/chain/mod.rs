use std::fmt;

use async_trait::async_trait;

use error::*;
use generic::*;

use crate::state::scalar::OpRef;
use crate::TxnId;

mod block;
pub mod sync;

pub use block::ChainBlock;

const PREFIX: PathLabel = path_label(&["state", "chain"]);

#[async_trait]
pub trait ChainInstance {
    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()>;
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ChainType {
    Sync,
}

impl Class for ChainType {
    type Instance = Chain;
}

impl NativeClass for ChainType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                "sync" => Some(Self::Sync),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        unimplemented!()
    }
}

impl fmt::Display for ChainType {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

#[derive(Clone)]
pub enum Chain {
    Sync(sync::SyncChain),
}

impl Instance for Chain {
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Sync(_) => ChainType::Sync,
        }
    }
}

#[async_trait]
impl ChainInstance for Chain {
    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()> {
        match self {
            Self::Sync(chain) => chain.append(txn_id, op_ref).await,
        }
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}
