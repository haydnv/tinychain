//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.
//! INCOMPLETE AND UNSTABLE.

use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use log::debug;
use safecast::{CastFrom, TryCastFrom};

use tc_error::*;
use tc_transact::fs::File;
use tc_transact::{Transact, TxnId};
use tc_value::Value;
use tcgeneric::*;

use crate::fs;
use crate::scalar::OpRef;
use crate::state::State;

mod block;
mod sync;

pub use block::ChainBlock;
pub use sync::*;

const CHAIN: Label = label("chain");
const PREFIX: PathLabel = path_label(&["state", "chain"]);
const SUBJECT: Label = label("subject");

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
    /// Return the state of this subject as of the given [`TxnId`].
    pub async fn at(&self, txn_id: &TxnId) -> TCResult<State> {
        debug!("Subject::at {}", txn_id);

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

    /// Set the state of this `Subject` to `value` at the given [`TxnId`].
    pub async fn put(&self, txn_id: &TxnId, key: Value, value: State) -> TCResult<()> {
        match self {
            Self::Value(file) => {
                if key.is_some() {
                    return Err(TCError::bad_request("Value has no such property", key));
                }

                let new_value = Value::try_cast_from(value, |v| {
                    TCError::bad_request("cannot update a Value to", v)
                })?;

                let new_json = serde_json::to_string(&new_value)
                    .map_err(|e| TCError::bad_request("error serializing Value", e))?;

                let block_id = SUBJECT.into();
                let mut block = file.get_block_mut(txn_id, &block_id).await?;
                debug!(
                    "set new Value of chain subject to {} ({}) at {}",
                    new_value, new_json, txn_id
                );
                *block = Bytes::from(new_json);

                Ok(())
            }
        }
    }
}

#[async_trait]
impl Transact for Subject {
    async fn commit(&self, txn_id: &TxnId) {
        debug!(
            "commit subject with value {} at {}",
            self.at(txn_id).await.unwrap(),
            txn_id
        );

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
    /// Append the given [`OpRef`] to the latest block in this `Chain`.
    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()>;

    /// Borrow the [`Subject`] of this [`Chain`] immutably.
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
