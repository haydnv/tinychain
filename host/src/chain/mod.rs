//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.
//! INCOMPLETE AND UNSTABLE.

use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::TryFutureExt;
use log::debug;
use safecast::{CastFrom, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::fs::{File, Persist};
use tc_transact::{Transact, TxnId};
use tcgeneric::*;

use crate::fs;
use crate::scalar::{Link, Scalar, Value};
use crate::state::State;
use crate::txn::Txn;

mod block;
mod blockchain;
mod sync;

pub use block::ChainBlock;
pub use blockchain::*;
pub use sync::*;

const CHAIN: Label = label("chain");
const PREFIX: PathLabel = path_label(&["state", "chain"]);
const SUBJECT: Label = label("subject");
const ERR_INVALID_SCHEMA: &str = "invalid Chain schema";

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
    Value(fs::File<Value>),
}

impl Subject {
    /// Return the state of this subject as of the given [`TxnId`].
    pub async fn at(&self, txn_id: &TxnId) -> TCResult<State> {
        debug!("Subject::at {}", txn_id);

        match self {
            Self::Value(file) => {
                let value = file.read_block(txn_id, &SUBJECT.into()).await?;
                Ok(value.deref().clone().into())
            }
        }
    }

    /// Set the state of this `Subject` to `value` at the given [`TxnId`].
    pub async fn put(&self, txn_id: TxnId, key: Value, value: State) -> TCResult<()> {
        match self {
            Self::Value(file) => {
                if key.is_some() {
                    return Err(TCError::bad_request("Value has no such property", key));
                }

                let new_value = Value::try_cast_from(value, |v| {
                    TCError::bad_request("cannot update a Value to", v)
                })?;

                let mut block = file.write_block(txn_id, SUBJECT.into()).await?;

                debug!(
                    "set new Value of chain subject to {} at {}",
                    new_value, txn_id
                );
                *block = new_value;

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
    /// Append the given PUT op to the latest block in this `Chain`.
    async fn append(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: Scalar,
    ) -> TCResult<()>;

    /// Borrow the [`Subject`] of this [`Chain`] immutably.
    fn subject(&self) -> &Subject;

    /// Replicate this [`Chain`] from the [`Chain`] at the given [`Link`].
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()>;
}

/// The type of a [`Chain`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ChainType {
    Block,
    Sync,
}

impl Class for ChainType {
    type Instance = Chain;
}

impl NativeClass for ChainType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 3 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                "block" => Some(Self::Block),
                "sync" => Some(Self::Sync),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::Block => "block",
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
    Block(blockchain::BlockChain),
    Sync(sync::SyncChain),
}

impl Instance for Chain {
    type Class = ChainType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Block(_) => ChainType::Block,
            Self::Sync(_) => ChainType::Sync,
        }
    }
}

#[async_trait]
impl ChainInstance for Chain {
    async fn append(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: Scalar,
    ) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.append(txn_id, path, key, value).await,
            Self::Sync(chain) => chain.append(txn_id, path, key, value).await,
        }
    }

    fn subject(&self) -> &Subject {
        match self {
            Self::Block(chain) => chain.subject(),
            Self::Sync(chain) => chain.subject(),
        }
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        match self {
            Self::Block(chain) => chain.replicate(txn, source).await,
            Self::Sync(chain) => chain.replicate(txn, source).await,
        }
    }
}

#[async_trait]
impl Transact for Chain {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Block(chain) => chain.commit(txn_id).await,
            Self::Sync(chain) => chain.commit(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Block(chain) => chain.finalize(txn_id).await,
            Self::Sync(chain) => chain.finalize(txn_id).await,
        }
    }
}

pub async fn load(class: ChainType, schema: Value, dir: fs::Dir, txn_id: TxnId) -> TCResult<Chain> {
    let schema = schema.try_cast_into(|v| TCError::bad_request(ERR_INVALID_SCHEMA, v))?;

    match class {
        ChainType::Block => {
            BlockChain::load(schema, dir, txn_id)
                .map_ok(Chain::Block)
                .await
        }
        ChainType::Sync => {
            SyncChain::load(schema, dir, txn_id)
                .map_ok(Chain::Sync)
                .await
        }
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "instance of {}", self.class())
    }
}
