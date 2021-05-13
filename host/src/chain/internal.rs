use std::convert::TryFrom;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::stream::{self, StreamExt};
use futures::{join, TryFutureExt, TryStreamExt};
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::fs::{Block, BlockData, BlockId, Dir, File, Persist};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::{label, Instance, Label, TCPathBuf, TCTryStream};

use crate::fs;
use crate::scalar::{Scalar, Value};
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::data::{ChainBlock, Mutation};
use super::{ChainType, CHAIN, NULL_HASH};

const DATA: Label = label("data");

#[derive(Clone)]
pub struct ChainData {
    dir: fs::Dir,
    file: fs::File<ChainBlock>,
    latest: TxnLock<Mutable<u64>>,
}

impl ChainData {
    fn new(latest: u64, dir: fs::Dir, file: fs::File<ChainBlock>) -> Self {
        let latest = TxnLock::new("latest block ordinal", latest.into());
        Self { dir, latest, file }
    }

    pub async fn create(txn_id: TxnId, dir: fs::Dir, class: ChainType) -> TCResult<Self> {
        let file = dir.create_file(txn_id, CHAIN.into(), class).await?;
        let file = fs::File::<ChainBlock>::try_from(file)?;
        file.create_block(txn_id, 0u64.into(), ChainBlock::new(NULL_HASH))
            .await?;

        let dir = dir.create_dir(txn_id, DATA.into()).await?;

        Ok(Self::new(0, dir, file))
    }

    pub async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        let mut block = self.write_latest(txn_id).await?;
        block.append_delete(txn_id, path, key);
        Ok(())
    }

    pub async fn append_put(
        &self,
        txn_id: TxnId,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        if value.is_ref() {
            return Err(TCError::bad_request(
                "cannot update Chain with reference: {}",
                value,
            ));
        }

        let value_ref = match value {
            State::Collection(collection) => Err(TCError::not_implemented(format!(
                "update Chain with value {}",
                collection.class()
            ))),
            State::Scalar(value) => Ok(value),
            other if Scalar::can_cast_from(&other) => Ok(other.opt_cast_into().unwrap()),
            other => Err(TCError::bad_request(
                "Chain does not support value",
                other.class(),
            )),
        }?;

        let mut block = self.write_latest(txn_id).await?;
        block.append_put(txn_id, path, key, value_ref);
        Ok(())
    }

    pub async fn last_commit(&self, txn_id: TxnId) -> TCResult<Option<TxnId>> {
        let block = self.read_latest(txn_id).await?;
        Ok(block.mutations().keys().next().cloned())
    }

    pub async fn latest_block_id(&self, txn_id: &TxnId) -> TCResult<u64> {
        self.latest.read(txn_id).map_ok(|id| *id).await
    }

    pub async fn contains_block(&self, txn_id: &TxnId, block_id: &BlockId) -> TCResult<bool> {
        self.file.contains_block(txn_id, block_id).await
    }

    pub async fn create_next_block(&self, txn_id: TxnId) -> TCResult<fs::Block<ChainBlock>> {
        let mut latest = self.latest.write(txn_id).await?;
        let last_block = self.read_block(txn_id, (*latest).into()).await?;
        let hash = last_block.hash().await?;
        let block = ChainBlock::new(hash);

        (*latest) += 1;
        self.file
            .create_block(txn_id, (*latest).into(), block)
            .await
    }

    pub async fn read_block(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<fs::BlockRead<ChainBlock>> {
        self.file.read_block(txn_id, block_id).await
    }

    pub async fn write_block(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> TCResult<fs::BlockWrite<ChainBlock>> {
        self.file.write_block(txn_id, block_id).await
    }

    pub async fn read_latest(&self, txn_id: TxnId) -> TCResult<fs::BlockRead<ChainBlock>> {
        let latest = self.latest.read(&txn_id).await?;
        self.read_block(txn_id, (*latest).into()).await
    }

    pub async fn write_latest(&self, txn_id: TxnId) -> TCResult<fs::BlockWrite<ChainBlock>> {
        let latest = self.latest.read(&txn_id).await?;
        self.write_block(txn_id, (*latest).into()).await
    }

    pub async fn prepare_commit(&self, txn_id: &TxnId) {
        let latest = self.latest.read(txn_id).await.expect("latest block");

        self.file
            .sync_block(*txn_id, (*latest).into())
            .await
            .expect("prepare BlockChain commit");
    }
}

const SCHEMA: () = ();

#[async_trait]
impl Persist<fs::Dir, Txn> for ChainData {
    type Schema = ();
    type Store = fs::Dir;

    fn schema(&self) -> &() {
        &SCHEMA
    }

    async fn load(txn: &Txn, _schema: (), dir: fs::Dir) -> TCResult<Self> {
        let file = dir
            .get_file(txn.id(), &CHAIN.into())
            .await?
            .ok_or_else(|| TCError::internal("Chain has no history file"))?;

        let file = fs::File::<ChainBlock>::try_from(file)?;

        let dir = dir
            .get_dir(txn.id(), &DATA.into())
            .await?
            .ok_or_else(|| TCError::internal("Chain has no data directory"))?;

        let mut last_hash = Bytes::from(NULL_HASH);
        let mut latest = 0;

        loop {
            let block = file.read_block(*txn.id(), latest.into()).await?;
            if block.last_hash() == &last_hash {
                last_hash = block.last_hash().clone();
            } else {
                return Err(TCError::internal(format!(
                    "block {} hash does not match previous block",
                    latest
                )));
            }

            if file.contains_block(txn.id(), &(latest + 1).into()).await? {
                latest += 1;
            } else {
                break;
            }
        }

        Ok(ChainData::new(latest, dir, file))
    }

    async fn save(&self, _txn_id: TxnId, _dir: fs::Dir) -> TCResult<()> {
        Err(TCError::not_implemented("ChainData::save"))
    }
}

#[async_trait]
impl Transact for ChainData {
    async fn commit(&self, txn_id: &TxnId) {
        join!(self.file.commit(txn_id), self.dir.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.file.finalize(txn_id), self.dir.finalize(txn_id));
    }
}

#[async_trait]
impl de::FromStream for ChainData {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(ChainDataVisitor { txn }).await
    }
}

struct ChainDataVisitor {
    txn: Txn,
}

#[async_trait]
impl de::Visitor for ChainDataVisitor {
    type Value = ChainData;

    fn expecting() -> &'static str {
        "Chain history"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let txn_id = *self.txn.id();
        let dir = self.txn.context().clone();

        let file = dir
            .create_file(txn_id, CHAIN.into(), ChainType::default())
            .map_err(de::Error::custom)
            .await?;

        let file = fs::File::<ChainBlock>::try_from(file).map_err(de::Error::custom)?;

        let dir = dir
            .create_dir(txn_id, DATA.into())
            .map_err(de::Error::custom)
            .await?;

        if let Some(first_block) = seq.next_element(()).await? {
            file.create_block(txn_id, 0u64.into(), first_block)
                .map_err(de::Error::custom)
                .await?;
        } else {
            let first_block = ChainBlock::new(NULL_HASH);
            file.create_block(txn_id, 0u64.into(), first_block)
                .map_err(de::Error::custom)
                .await?;

            return Ok(ChainData::new(0, dir, file));
        }

        let chain = ChainData::new(0, dir, file);

        while let Some(block_data) = seq.next_element::<ChainBlock>(()).await? {
            let block = chain
                .create_next_block(txn_id)
                .map_err(de::Error::custom)
                .await?;

            let mut block = block.write().await;
            if block.last_hash() == block_data.last_hash() {
                *block = block_data;
            } else {
                let unexpected = base64::encode(block_data.last_hash());
                let expected = base64::encode(block.last_hash());

                return Err(de::Error::invalid_value(
                    format!("block with hash {}", unexpected),
                    format!("block with hash {}", expected),
                ));
            }
        }

        Ok(chain)
    }
}

pub type ChainDataView<'en> =
    en::SeqStream<TCError, ChainDataBlockView<'en>, TCTryStream<'en, ChainDataBlockView<'en>>>;

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for ChainData {
    type Txn = Txn;
    type View = ChainDataView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        let txn_id = *txn.id();
        let latest = self.latest.read(&txn_id).await?;

        let file = self.file.clone();
        let read_block = move |block_id| Box::pin(file.clone().read_block_owned(txn_id, block_id));

        let seq = stream::iter(0..((*latest) + 1))
            .map(BlockId::from)
            .then(read_block)
            .map_ok(move |block| {
                let txn = txn.clone();
                let map =
                    stream::iter(block.mutations().clone()).map(move |(past_txn_id, mutations)| {
                        let txn = txn.clone();
                        let mutations = stream::iter(mutations).then(move |op| {
                            let txn = txn.clone();
                            Box::pin(async move {
                                match op {
                                    Mutation::Delete(path, key) => {
                                        Ok(MutationView::Delete(path, key))
                                    }
                                    Mutation::Put(_path, _key, value) if value.is_ref() => {
                                        Err(TCError::not_implemented(
                                            "resolve reference in Mutation::Put",
                                        ))
                                    }
                                    Mutation::Put(path, key, value) => {
                                        let value =
                                            State::from(value).into_view(txn.clone()).await?;

                                        Ok(MutationView::Put(path, key, value))
                                    }
                                }
                            })
                        });

                        let mutations: TCTryStream<'en, MutationView<'en>> = Box::pin(mutations);
                        let mutations = en::SeqStream::from(mutations);
                        Ok((past_txn_id, mutations))
                    });

                let map: TCTryStream<'en, (TxnId, MutationViewSeq<'en>)> = Box::pin(map);
                (block.last_hash().clone(), en::MapStream::from(map))
            });

        let seq: TCTryStream<'en, ChainDataBlockView<'en>> = Box::pin(seq);
        Ok(en::SeqStream::from(seq))
    }
}

type MutationViewSeq<'en> =
    en::SeqStream<TCError, MutationView<'en>, TCTryStream<'en, MutationView<'en>>>;

type ChainDataBlockView<'en> = (
    Bytes,
    en::MapStream<
        TCError,
        TxnId,
        MutationViewSeq<'en>,
        TCTryStream<'en, (TxnId, MutationViewSeq<'en>)>,
    >,
);

pub enum MutationView<'en> {
    Delete(TCPathBuf, Value),
    Put(TCPathBuf, Value, StateView<'en>),
}

impl<'en> en::IntoStream<'en> for MutationView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Delete(path, key) => (path, key).into_stream(encoder),
            Self::Put(path, key, value) => (path, key, value).into_stream(encoder),
        }
    }
}
