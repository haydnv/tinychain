use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use futures_locks::RwLock;

use error::*;
use generic::*;
use transact::{IntoView, TxnId};

use crate::fs::{DirView, FileView};
use crate::scalar::OpRef;
use crate::txn::Txn;

mod block;
pub mod sync;

pub use block::ChainBlock;

const PREFIX: PathLabel = path_label(&["state", "chain"]);

#[async_trait]
pub trait ChainInstance {
    async fn file(&self, txn_id: &TxnId) -> TCResult<RwLock<FileView<ChainBlock>>>;

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
    async fn file(&self, txn_id: &TxnId) -> TCResult<RwLock<FileView<ChainBlock>>> {
        match self {
            Self::Sync(chain) => chain.file(txn_id).await,
        }
    }

    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()> {
        match self {
            Self::Sync(chain) => chain.append(txn_id, op_ref).await,
        }
    }
}

#[async_trait]
impl de::FromStream for Chain {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(ChainVisitor { txn }).await
    }
}

impl<'en> IntoView<'en, DirView> for Chain {
    type Txn = Txn;
    type View = ChainView;

    fn into_view(self, txn: Self::Txn) -> Self::View {
        ChainView { txn, chain: self }
    }
}

impl fmt::Display for Chain {
    fn fmt(&self, _f: &mut fmt::Formatter) -> fmt::Result {
        unimplemented!()
    }
}

pub struct ChainView {
    txn: Txn,
    chain: Chain,
}

impl<'en> en::IntoStream<'en> for ChainView {
    fn into_stream<E: en::Encoder<'en>>(self, _encoder: E) -> Result<E::Ok, E::Error> {
        unimplemented!()
    }
}

pub struct ChainVisitor {
    txn: Txn,
}

impl ChainVisitor {
    pub async fn visit_map_value<A: de::MapAccess>(
        self,
        class: ChainType,
        access: &mut A,
    ) -> Result<Chain, A::Error> {
        match class {
            ChainType::Sync => access.next_value(self.txn).map_ok(Chain::Sync).await,
        }
    }
}

impl From<Txn> for ChainVisitor {
    fn from(txn: Txn) -> Self {
        Self { txn }
    }
}

#[async_trait]
impl de::Visitor for ChainVisitor {
    type Value = Chain;

    fn expecting() -> &'static str {
        "a Chain"
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        if let Some(key) = access.next_key::<TCPathBuf>(()).await? {
            if let Some(class) = ChainType::from_path(&key) {
                self.visit_map_value(class, &mut access).await
            } else {
                Err(de::Error::invalid_value(key, "a Chain classpath"))
            }
        } else {
            Err(de::Error::invalid_length(0, "a Chain"))
        }
    }
}
