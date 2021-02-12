//! A [`Chain`] responsible for recovering a `State` from a failed transaction.
//! INCOMPLETE AND UNSTABLE.

use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;

use error::*;
use generic::*;
use safecast::CastFrom;
use transact::fs::File;
use transact::{Transact, Transaction, TxnId};
use value::Value;

use crate::fs;
use crate::route::*;
use crate::scalar::OpRef;
use crate::state::State;
use crate::txn::Txn;

mod block;
mod sync;

pub use block::ChainBlock;
pub use sync::*;

const CHAIN: Label = label("chain");
const SUBJECT: Label = label("subject");
const PREFIX: PathLabel = path_label(&["state", "chain"]);

/// The file extension of a directory of [`ChainBlock`]s on disk.
pub const EXT: &str = "chain";

/// The schema of a [`Chain`], used when constructing a new `Chain` or loading a `Chain` from disk.
#[derive(Clone)]
pub enum Schema {
    Value(Value),
}

impl CastFrom<Value> for Schema {
    fn cast_from(value: Value) -> Self {
        Self::Value(value)
    }
}

/// The state whose transactional integrity is protected by a [`Chain`].
#[derive(Clone)]
pub enum Subject {
    Value(fs::File<Bytes>),
}

impl Subject {
    async fn at(&self, txn_id: &TxnId) -> TCResult<State> {
        match self {
            Self::Value(file) => {
                let block_id = SUBJECT.into();
                let block = file.get_block(txn_id, &block_id).await?;
                let value: Value = serde_json::from_slice(&block)
                    .map_err(|e| TCError::internal(format!("block corrupted! {}", e)))?;

                Ok(value.into())
            }
        }
    }
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

/// Trait defining methods common to any instance of a [`Chain`], such as a [`SyncChain`].
#[async_trait]
pub trait ChainInstance {
    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()>;

    fn subject(&self) -> &Subject;
}

/// The type of a [`Chain`].
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

/// A data structure responsible for maintaining the transactional integrity of its [`Subject`].
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

    fn subject(&self) -> &Subject {
        match self {
            Self::Sync(chain) => chain.subject(),
        }
    }
}

#[async_trait]
impl Public for Chain {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        let subject = self.subject().at(txn.id()).await?;
        subject.get(txn, path, key).await
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        let subject = self.subject().at(txn.id()).await?;
        subject.put(txn, path, key, value).await
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], params: Map<State>) -> TCResult<State> {
        let subject = self.subject().at(txn.id()).await?;
        subject.post(txn, path, params).await
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
