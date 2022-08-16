use std::collections::BTreeMap;
use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use freqfs::{DirLock, FileReadGuard, FileWriteGuard};
use futures::stream::{self, StreamExt};
use futures::{join, try_join, TryFutureExt, TryStreamExt};
use log::{debug, error};
use safecast::*;

use tc_error::*;
use tc_transact::fs::{BlockData, Dir, File, Persist};
use tc_transact::lock::TxnLock;
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, Label, Map, TCBoxStream, TCBoxTryStream, TCPathBuf, Tuple};

use crate::chain::{null_hash, Subject, BLOCK_SIZE, CHAIN};
use crate::fs;
use crate::route::Public;
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::{ChainBlock, Mutation, Store};

const PENDING: &str = "pending.chain_block";
const STORE: Label = label("store");

#[derive(Clone)]
pub struct History {
    file: DirLock<fs::CacheBlock>,
    store: Store,
    latest: TxnLock<u64>,
    cutoff: TxnLock<TxnId>,
}

impl History {
    fn new(file: DirLock<fs::CacheBlock>, store: Store, latest: u64, cutoff: TxnId) -> Self {
        Self {
            file,
            store,
            latest: TxnLock::new("latest block ordinal", latest),
            cutoff: TxnLock::new("block transaction time cutoff", cutoff),
        }
    }

    pub async fn create(txn_id: TxnId, dir: fs::Dir) -> TCResult<Self> {
        let store = dir
            .create_dir(txn_id, STORE.into())
            .map_ok(Store::new)
            .await?;

        let file = dir.into_inner();
        let mut file_lock = file.write().await;
        let block = ChainBlock::new(null_hash().to_vec());

        file_lock
            .create_file(block_name(PENDING), block.clone(), Some(0))
            .map_err(fs::io_err)?;

        file_lock
            .create_file(block_name(0u64), block.clone(), Some(0))
            .map_err(fs::io_err)?;

        Ok(Self::new(file, store, 0, txn_id))
    }

    pub async fn append_delete(&self, txn_id: TxnId, path: TCPathBuf, key: Value) -> TCResult<()> {
        debug!("History::append_delete {} {} {}", txn_id, path, key);
        let mut block = self.write_latest(txn_id).await?;
        block.append_delete(txn_id, path, key);
        Ok(())
    }

    pub async fn append_put(
        &self,
        txn: &Txn,
        path: TCPathBuf,
        key: Value,
        value: State,
    ) -> TCResult<()> {
        let txn_id = *txn.id();
        let value = self.store.save_state(txn, value).await?;

        debug!(
            "History::append_put {} {} {:?} {:?}",
            txn_id, path, key, value
        );

        let mut block = self.write_latest(txn_id).await?;
        block.append_put(txn_id, path, key, value);

        Ok(())
    }

    async fn read_block(
        &self,
        block_id: u64,
    ) -> TCResult<<fs::File<ChainBlock> as File<ChainBlock>>::Read> {
        let file = self.file.read().await;
        let block = file
            .get_file(&block_name(block_id))
            .ok_or_else(|| TCError::not_found(format!("chain block {}", block_id)))?;

        block.read().map_err(fs::io_err).await
    }

    async fn write_block(
        &self,
        block_id: u64,
    ) -> TCResult<<fs::File<ChainBlock> as File<ChainBlock>>::Write> {
        let file = self.file.read().await;
        let block = file
            .get_file(&block_name(block_id))
            .ok_or_else(|| TCError::not_found(format!("chain block {}", block_id)))?;

        block.write().map_err(fs::io_err).await
    }

    pub async fn read_latest(
        &self,
        txn_id: TxnId,
    ) -> TCResult<<fs::File<ChainBlock> as File<ChainBlock>>::Read> {
        let latest = self.latest.read(txn_id).await?;
        self.read_block(*latest).await
    }

    pub async fn write_latest(
        &self,
        txn_id: TxnId,
    ) -> TCResult<<fs::File<ChainBlock> as File<ChainBlock>>::Write> {
        let latest = self.latest.read(txn_id).await?;
        self.write_block(*latest).await
    }

    pub async fn replicate(&self, txn: &Txn, subject: &Subject, other: Self) -> TCResult<()> {
        debug!("replicate chain history");

        let txn_id = *txn.id();

        let (latest, other_latest) =
            try_join!(self.latest.read(txn_id), other.latest.read(txn_id))?;

        if (*latest) > (*other_latest) {
            return Err(TCError::bad_request(
                "cannot replicate from chain with fewer blocks",
                *latest,
            ));
        }

        let mut latest_txn_id = None;

        const ERR_DIVERGENT: &str = "chain to replicate diverges at block";
        for i in 0u64..*latest {
            let (block, other) = try_join!(self.read_block(i), other.read_block(i))?;

            if &*block != &*other {
                return Err(TCError::bad_request(ERR_DIVERGENT, i));
            }

            if let Some(txn_id) = block.mutations.keys().last() {
                latest_txn_id = Some(*txn_id);
            }
        }

        let mut last_hash = {
            let (mut dest, source) =
                try_join!(self.write_block(*latest), other.read_block(*latest))?;

            if let Some(txn_id) = dest.mutations.keys().last() {
                latest_txn_id = Some(*txn_id);
            }

            for (txn_id, ops) in &source.mutations {
                if let Some(latest_txn_id) = &latest_txn_id {
                    if txn_id <= latest_txn_id {
                        continue;
                    }
                }

                for op in ops {
                    match op {
                        Mutation::Delete(path, key) => {
                            dest.append_delete(*txn_id, path.clone(), key.clone());
                            subject.delete(txn, &path, key.clone()).await?
                        }
                        Mutation::Put(path, key, value) => {
                            let value = other.store.resolve(txn, value.clone()).await?;
                            let value_ref = self.store.save_state(txn, value.clone()).await?;

                            dest.append_put(*txn_id, path.clone(), key.clone(), value_ref);
                            subject.put(txn, &path, key.clone(), value).await?
                        }
                    }
                }
            }

            let last_hash = dest.hash().to_vec();
            if &last_hash[..] != &source.hash()[..] {
                return Err(TCError::internal(ERR_DIVERGENT));
            }

            last_hash
        };

        let mut this_file = self.file.write().await;
        for block_id in (*latest + 1)..(*other_latest + 1) {
            let source = other.read_block(block_id).await?;

            let dest = this_file
                .create_file(
                    block_name(block_id),
                    ChainBlock::new(last_hash.to_vec()),
                    Some(last_hash.len()),
                )
                .map_err(fs::io_err)?;

            let mut dest: FileWriteGuard<_, ChainBlock> = dest.write().map_err(fs::io_err).await?;

            for (txn_id, ops) in &source.mutations {
                for op in ops {
                    match op {
                        Mutation::Delete(path, key) => {
                            dest.append_delete(*txn_id, path.clone(), key.clone());
                            subject.delete(txn, &path, key.clone()).await?
                        }
                        Mutation::Put(path, key, value) => {
                            let value = other.store.resolve(txn, value.clone()).await?;
                            let value_ref = self.store.save_state(txn, value.clone()).await?;

                            dest.append_put(*txn_id, path.clone(), key.clone(), value_ref);
                            subject.put(txn, &path, key.clone(), value).await?
                        }
                    }
                }
            }

            last_hash = dest.hash().to_vec();
            if &last_hash[..] != &source.hash()[..] {
                return Err(TCError::internal(ERR_DIVERGENT));
            }
        }

        Ok(())
    }
}

const SCHEMA: () = ();

#[async_trait]
impl Persist<fs::Dir> for History {
    type Schema = ();
    type Store = fs::Dir;
    type Txn = Txn;

    fn schema(&self) -> &() {
        &SCHEMA
    }

    async fn load(txn: &Txn, _schema: (), dir: fs::Dir) -> TCResult<Self> {
        let txn_id = txn.id();

        // if there's no data in the data dir, it may not have been sync'd to the filesystem
        // so just create a new one
        let store = dir
            .get_or_create_dir(*txn_id, STORE.into())
            .map_ok(Store::new)
            .await?;

        let dir = dir.into_inner().read().await;
        let file = dir
            .get_dir(&format!("{}.{}", CHAIN, ChainBlock::ext()))
            .ok_or_else(|| TCError::internal("Chain has no history file"))?;

        let file_lock = file.read().await;

        let mut cutoff = *txn_id;
        let mut latest = 0;
        let mut last_hash = Bytes::from(null_hash().to_vec());

        while let Some(block) = file_lock.get_file(&format!("{}.{}", latest, ChainBlock::ext())) {
            let block: FileReadGuard<_, ChainBlock> = block.read().map_err(fs::io_err).await?;

            if block.last_hash() == &last_hash {
                last_hash = block.last_hash().clone();
            } else {
                return Err(TCError::internal(format!(
                    "block {} hash does not match previous block",
                    latest
                )));
            }

            cutoff = block.mutations.keys().last().copied().unwrap_or(cutoff);
            latest += 1;
        }

        Ok(Self::new(file.clone(), store, latest, cutoff))
    }
}

#[async_trait]
impl Transact for History {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit chain history {}", txn_id);

        self.store.commit(txn_id).await;

        let (mut latest, mut cutoff) =
            try_join!(self.latest.write(*txn_id), self.cutoff.write(*txn_id))
                .expect("BlockChain state");

        assert!(
            txn_id >= &cutoff,
            "cannot commit transaction {} since a block has already been committed at {}",
            txn_id,
            *cutoff
        );

        let mut file = self.file.write().await;
        let mut pending: FileWriteGuard<_, ChainBlock> = file
            .get_file(&block_name(PENDING))
            .expect("pending transactions")
            .write()
            .await
            .expect("pending transaction lock");

        if let Some(mutations) = pending.mutations.remove(txn_id) {
            let latest_block = file.get_file(&block_name(*latest)).expect("latest block");

            let mut latest_block: FileWriteGuard<_, ChainBlock> =
                latest_block.write().await.expect("latest block write lock");

            latest_block.mutations.insert(*txn_id, mutations);

            if latest_block.size().await.expect("block size") > BLOCK_SIZE {
                *cutoff = *txn_id;

                let hash = latest_block.hash();

                file.create_file(
                    block_name(*latest),
                    ChainBlock::new(hash.to_vec()),
                    Some(hash.len()),
                )
                .expect("new chain block");

                *latest += 1;
            }
        }

        self.latest.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(self.store.finalize(txn_id), self.latest.finalize(txn_id));

        let file = self.file.read().await;
        let mut pending: FileWriteGuard<_, ChainBlock> = file
            .get_file(&block_name(PENDING))
            .expect("pending transactions")
            .write()
            .await
            .expect("pending transaction lock");

        pending.mutations.remove(txn_id);
    }
}

#[async_trait]
impl de::FromStream for History {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(HistoryVisitor { txn }).await
    }
}

struct HistoryVisitor {
    txn: Txn,
}

#[async_trait]
impl de::Visitor for HistoryVisitor {
    type Value = History;

    fn expecting() -> &'static str {
        "Chain history"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let null_hash = null_hash();
        let txn_id = *self.txn.id();
        let dir = self.txn.context().clone();

        let store = dir
            .create_dir(txn_id, STORE.into())
            .map_ok(Store::new)
            .map_err(de::Error::custom)
            .await?;

        let file = dir
            .into_inner()
            .write()
            .await
            .create_dir(CHAIN.to_string())
            .map_err(de::Error::custom)?;

        let mut file_lock = file.write().await;

        file_lock
            .create_file(
                PENDING.to_string(),
                ChainBlock::new(null_hash.to_vec()),
                Some(0),
            )
            .map_err(de::Error::custom)?;

        let subcontext = |i: u64| self.txn.subcontext(i.into()).map_err(de::Error::custom);

        let mut i = 0u64;
        let mut last_hash = null_hash.clone();

        while let Some(state) = seq.next_element::<State>(subcontext(i).await?).await? {
            let (hash, block_data): (Bytes, Map<Tuple<State>>) =
                state.try_cast_into(|s| de::Error::invalid_type(s, "a chain block"))?;

            if &hash[..] != &last_hash[..] {
                return Err(de::Error::invalid_value(
                    format!("block with last hash {}", hex::encode(hash)),
                    format!("block with last hash {}", hex::encode(last_hash)),
                ));
            }

            let mutations = parse_block_state(&store, &self.txn, block_data)
                .map_err(de::Error::custom)
                .await?;

            let block = ChainBlock::with_mutations(hash, mutations);
            last_hash = block.hash();

            file_lock
                .create_file(block_name(i), block, None)
                .map_err(de::Error::custom)?;

            i += 1;
        }

        Ok(History::new(file, store, i, txn_id))
    }
}

async fn parse_block_state(
    store: &Store,
    txn: &Txn,
    block_data: Map<Tuple<State>>,
) -> TCResult<BTreeMap<TxnId, Vec<Mutation>>> {
    let mut mutations = BTreeMap::new();

    for (past_txn_id, ops) in block_data.into_iter() {
        let past_txn_id = past_txn_id.to_string().parse()?;

        let mut parsed = Vec::with_capacity(ops.len());

        for op in ops.into_iter() {
            if op.matches::<(TCPathBuf, Value)>() {
                let (path, key) = op.opt_cast_into().unwrap();
                parsed.push(Mutation::Delete(path, key));
            } else if op.matches::<(TCPathBuf, Value, State)>() {
                let (path, key, value) = op.opt_cast_into().unwrap();
                let value = store.save_state(txn, value).await?;
                parsed.push(Mutation::Put(path, key, value));
            } else {
                return Err(TCError::bad_request(
                    "unable to parse historical mutation",
                    op,
                ));
            }
        }

        mutations.insert(past_txn_id, parsed);
    }

    Ok(mutations)
}

pub type HistoryView<'en> =
    en::SeqStream<TCResult<HistoryBlockView<'en>>, TCBoxTryStream<'en, HistoryBlockView<'en>>>;

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for History {
    type Txn = Txn;
    type View = HistoryView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        debug!("History::into_view");

        let txn_id = *txn.id();
        let latest = self.latest.read(txn_id).await?;

        let file = self.file.read().await;

        let seq = stream::iter(0..((*latest) + 1))
            .map(move |block_id| {
                file.get_file(&block_name(block_id))
                    .ok_or_else(|| TCError::internal("missing chain block"))
            })
            .and_then(|block| Box::pin(async move { block.read().map_err(fs::io_err).await }))
            .map_ok(move |block: FileReadGuard<_, ChainBlock>| {
                let this = self.clone();
                let txn = txn.clone();
                let map =
                    stream::iter(block.mutations.clone()).map(move |(past_txn_id, mutations)| {
                        debug!("reading block mutations");

                        let this = this.clone();
                        let txn = txn.clone();
                        let mutations = stream::iter(mutations)
                            .then(move |op| Box::pin(load_history(this.clone(), op, txn.clone())));

                        let mutations: TCBoxTryStream<'en, MutationView<'en>> = Box::pin(mutations);
                        let mutations = en::SeqStream::from(mutations);
                        (past_txn_id, Ok(mutations))
                    });

                let map: TCBoxStream<'en, (TxnId, TCResult<MutationViewSeq<'en>>)> = Box::pin(map);
                (block.last_hash().clone(), en::MapStream::from(map))
            });

        let seq: TCBoxTryStream<'en, HistoryBlockView<'en>> = Box::pin(seq);
        Ok(en::SeqStream::from(seq))
    }
}

async fn load_history<'a>(history: History, op: Mutation, txn: Txn) -> TCResult<MutationView<'a>> {
    match op {
        Mutation::Delete(path, key) => Ok(MutationView::Delete(path, key)),
        Mutation::Put(path, key, value) if value.is_ref() => {
            debug!("historical mutation: PUT {}: {} <- {}", path, key, value);

            let value = history
                .store
                .resolve(&txn, value)
                .map_err(|err| {
                    error!("unable to load historical Chain data: {}", err);
                    err
                })
                .await?;

            let value = value.into_view(txn).await?;
            Ok(MutationView::Put(path, key, value))
        }
        Mutation::Put(path, key, value) => {
            let value = State::from(value).into_view(txn.clone()).await?;

            Ok(MutationView::Put(path, key, value))
        }
    }
}

type MutationViewSeq<'en> =
    en::SeqStream<TCResult<MutationView<'en>>, TCBoxTryStream<'en, MutationView<'en>>>;

type HistoryBlockView<'en> = (
    Bytes,
    en::MapStream<
        TxnId,
        TCResult<MutationViewSeq<'en>>,
        TCBoxStream<'en, (TxnId, TCResult<MutationViewSeq<'en>>)>,
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

fn block_name<I: fmt::Display>(block_id: I) -> String {
    format!("{}.{}", block_id, ChainBlock::ext())
}
