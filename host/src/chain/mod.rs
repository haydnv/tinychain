//! A [`Chain`] responsible for recovering a [`State`] from a failed transaction.

use std::convert::TryInto;
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use log::debug;
use safecast::{CastFrom, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::fs::{Dir, File, Persist};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::*;

use crate::fs;
use crate::scalar::{Link, Scalar, Value};
use crate::state::{State, StateView};
use crate::txn::Txn;

mod block;
mod blockchain;
mod sync;

use blockchain::BlockSeq;

use crate::fs::FileEntry;
pub use block::ChainBlock;
pub use blockchain::BlockChain;
pub use sync::SyncChain;

const CHAIN: Label = label("chain");
const NULL_HASH: Vec<u8> = vec![];
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

#[async_trait]
impl de::FromStream for Schema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        Value::from_stream(cxt, decoder).map_ok(Self::Value).await
    }
}

impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Value(value) => value.into_stream(encoder),
        }
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

    async fn load(schema: &Schema, dir: &fs::Dir, txn_id: TxnId) -> TCResult<Self> {
        match schema {
            Schema::Value(value) => {
                let file: fs::File<Value> =
                    if let Some(file) = dir.get_file(&txn_id, &SUBJECT.into()).await? {
                        file.try_into()?
                    } else {
                        let file = dir
                            .create_file(txn_id, SUBJECT.into(), value.class().into())
                            .await?;

                        file.try_into()?
                    };

                if !file.contains_block(&txn_id, &SUBJECT.into()).await? {
                    debug!("chain writing new subject...");
                    file.create_block(txn_id, SUBJECT.into(), value.clone())
                        .await?;
                } else {
                    debug!("chain found existing subject");
                }

                Ok(Subject::Value(file))
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

#[async_trait]
impl de::FromStream for Subject {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let value = Value::from_stream((), decoder).await?;

        let file: FileEntry = txn
            .context()
            .create_file(*txn.id(), SUBJECT.into(), value.class().into())
            .map_err(de::Error::custom)
            .await?;

        let file: fs::File<Value> = file.try_into().map_err(de::Error::custom)?;
        file.create_block(*txn.id(), SUBJECT.into(), value)
            .map_err(de::Error::custom)
            .await?;

        Ok(Self::Value(file))
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

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Chain {
    type Txn = Txn;
    type View = ChainView;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::Block(chain) => chain.into_view(txn).map_ok(ChainView::Block).await,
            Self::Sync(chain) => {
                chain
                    .into_view(txn)
                    .map_ok(Box::new)
                    .map_ok(ChainView::Sync)
                    .await
            }
        }
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "instance of {}", self.class())
    }
}

pub enum ChainView {
    Block((Schema, BlockSeq)),
    Sync(Box<(Schema, StateView)>),
}

impl<'en> en::IntoStream<'en> for ChainView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Block(view) => view.into_stream(encoder),
            Self::Sync(view) => view.into_stream(encoder),
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
