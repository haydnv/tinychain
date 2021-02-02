use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::TryFutureExt;
use transact::Transaction;

use error::*;
use generic::*;
use transact::fs::{BlockOwned, File};
use transact::{IntoView, TxnId};

use crate::scalar::OpRef;
use crate::txn::{FileEntry, Txn};

mod block;
pub mod sync;

pub use block::ChainBlock;

const PREFIX: PathLabel = path_label(&["state", "chain"]);

#[async_trait]
pub trait ChainInstance {
    fn file(&'_ self) -> &'_ File<ChainBlock>;

    fn len(&self) -> u64;

    async fn append(&self, txn_id: &TxnId, op_ref: OpRef) -> TCResult<()>;

    fn block_stream(
        &self,
        txn_id: TxnId,
    ) -> Box<dyn Stream<Item = TCResult<BlockOwned<ChainBlock>>> + Send + Unpin> {
        let file = self.file().clone();
        let blocks = stream::iter(0..self.len())
            .then(move |block_id| Box::pin(file.clone().get_block_owned(txn_id, block_id.into())));

        Box::new(blocks)
    }
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
    fn file(&'_ self) -> &'_ File<ChainBlock> {
        match self {
            Self::Sync(chain) => chain.file(),
        }
    }

    fn len(&self) -> u64 {
        match self {
            Self::Sync(chain) => chain.len(),
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

impl<'en> IntoView<'en, FileEntry> for Chain {
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
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let blocks = self
            .chain
            .block_stream(*self.txn.id())
            .map_err(en::Error::custom);

        encoder.encode_seq_stream(blocks)
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
