use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;

use error::*;
use generic::*;
use safecast::CastFrom;
use transact::{Transact, TxnId};
use value::Value;

use crate::fs;
use crate::scalar::OpRef;

mod block;
mod sync;

pub use block::ChainBlock;
pub use sync::*;

const PREFIX: PathLabel = path_label(&["state", "chain"]);

pub const EXT: &str = "chain";

#[derive(Clone)]
pub enum Schema {
    Value(Value),
}

impl CastFrom<Value> for Schema {
    fn cast_from(value: Value) -> Self {
        Self::Value(value)
    }
}

#[derive(Clone)]
pub enum Subject {
    Value(fs::File<Bytes>),
}

#[async_trait]
impl Transact for Subject {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Value(file) => file.commit(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Value(file) => file.finalize(txn_id).await,
        }
    }
}

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
        let suffix = match self {
            Self::Sync => "sync",
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
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

#[async_trait]
impl Transact for Chain {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Sync(chain) => chain.commit(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Sync(chain) => chain.finalize(txn_id).await,
        }
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}
