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
use tc_transact::fs::{BlockData, Dir, DirWrite, Persist};
use tc_transact::lock::TxnLock;
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, Label, Map, TCBoxStream, TCBoxTryStream, TCPathBuf, Tuple};

use crate::chain::{null_hash, Subject, BLOCK_SIZE, CHAIN};
use crate::fs;
use crate::fs::CacheBlock;
use crate::route::Public;
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::{ChainBlock, Mutation, Store};

const STORE: Label = label("store");
const PENDING: &str = "pending";
const WRITE_AHEAD: &str = "write_ahead";

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

    pub fn store(&self) -> &Store {
        &self.store
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
    ) -> TCResult<freqfs::FileReadGuard<CacheBlock, ChainBlock>> {
        let file = self.file.read().await;
        let block = file
            .get_file(&block_name(block_id))
            .ok_or_else(|| TCError::not_found(format!("chain block {}", block_id)))?;

        block.read().map_err(fs::io_err).await
    }

    async fn write_block(
        &self,
        block_id: u64,
    ) -> TCResult<freqfs::FileWriteGuard<CacheBlock, ChainBlock>> {
        let file = self.file.read().await;
        let block = file
            .get_file(&block_name(block_id))
            .ok_or_else(|| TCError::not_found(format!("chain block {}", block_id)))?;

        block.write().map_err(fs::io_err).await
    }

    pub async fn read_latest(
        &self,
        txn_id: TxnId,
    ) -> TCResult<freqfs::FileReadGuard<CacheBlock, ChainBlock>> {
        let latest = self.latest.read(txn_id).await?;
        self.read_block(*latest).await
    }

    pub async fn write_latest(
        &self,
        txn_id: TxnId,
    ) -> TCResult<freqfs::FileWriteGuard<CacheBlock, ChainBlock>> {
        let latest = self.latest.read(txn_id).await?;
        self.write_block(*latest).await
    }

    pub async fn read_log(&self) -> TCResult<freqfs::FileReadGuard<CacheBlock, ChainBlock>> {
        let log = {
            let file = self.file.read().await;
            file.get_file(&block_name(WRITE_AHEAD))
                .expect("write-ahead log")
        };

        log.read().map_err(fs::io_err).await
    }

    pub async fn replicate(&self, txn: &Txn, subject: &Subject, other: Self) -> TCResult<()> {
        let err_divergent =
            |block_id| TCError::bad_request("chain to replicate diverges at block", block_id);

        debug!("replicate chain history");

        let txn_id = *txn.id();

        let (latest, other_latest) =
            try_join!(self.latest.read(txn_id), other.latest.read(txn_id))?;

        debug!("chain to replicate ends with block {}", *other_latest);

        if (*latest) > (*other_latest) {
            return Err(TCError::bad_request(
                "cannot replicate from chain with fewer blocks",
                *latest,
            ));
        }

        let mut latest_txn_id = None;

        // handle blocks with ordinal < latest, which should be identical
        for i in 0u64..*latest {
            let (block, other) = try_join!(self.read_block(i), other.read_block(i))?;

            if &*block != &*other {
                return Err(err_divergent(i));
            }

            if let Some(txn_id) = block.mutations.keys().last() {
                latest_txn_id = Some(*txn_id);
            }
        }

        // handle the latest block, which may be shorter than the corresponding block to replicate
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

                assert!(!dest.mutations.contains_key(txn_id));
                replay_and_save(
                    &subject,
                    txn,
                    *txn_id,
                    ops,
                    &other.store,
                    &self.store,
                    &mut dest,
                )
                .await?;
            }

            let last_hash = dest.hash().to_vec();
            if &last_hash[..] != &source.hash()[..] {
                return Err(TCError::internal(err_divergent(*latest)));
            }

            last_hash
        };

        // if the other chain has the same number of blocks, replication is complete
        if *latest == *other_latest {
            return Ok(());
        }

        // otherwise, handle the remaining blocks
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
                assert!(!dest.mutations.contains_key(txn_id));

                replay_and_save(
                    &subject,
                    txn,
                    *txn_id,
                    ops,
                    &other.store,
                    &self.store,
                    &mut dest,
                )
                .await?;
            }

            last_hash = dest.hash().to_vec();
            if &last_hash[..] != &source.hash()[..] {
                return Err(TCError::internal(err_divergent(block_id)));
            }
        }

        Ok(())
    }

    pub async fn write_ahead(&self, txn_id: &TxnId) {
        self.store.commit(txn_id).await;

        let file = self.file.read().await;
        let mut pending: FileWriteGuard<_, ChainBlock> = file
            .get_file(&block_name(PENDING))
            .expect("pending transactions")
            .write()
            .await
            .expect("pending block write lock");

        if let Some(mutations) = pending.mutations.remove(txn_id) {
            let write_ahead = file
                .get_file(&block_name(WRITE_AHEAD))
                .expect("write-ahead log");

            {
                let mut write_ahead: FileWriteGuard<_, ChainBlock> = write_ahead
                    .write()
                    .await
                    .expect("write-ahead log write lock");

                write_ahead.mutations.insert(*txn_id, mutations);
            }

            write_ahead.sync(false).await.expect("sync write-ahead log");
        }
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
        let store = {
            let mut dir = dir.write(*txn_id).await?;
            dir.get_or_create_dir(STORE.into())
                .map_ok(Store::new)
                .await?
        };

        let file = {
            let mut dir = dir.into_inner().write().await;
            if let Some(file) = dir.get_dir(&block_name(CHAIN)) {
                file.clone()
            } else {
                dir.create_dir(block_name(CHAIN)).map_err(fs::io_err)?
            }
        };

        let mut file_lock = file.write().await;

        let mut cutoff = *txn_id;
        let mut latest = 0;
        let mut last_hash = Bytes::from(null_hash().to_vec());

        if file_lock.get_file(&block_name(PENDING)).is_none() {
            // the pending block may never be sync'd to the filesystem
            file_lock
                .create_file(
                    block_name(PENDING),
                    ChainBlock::new(last_hash.clone()),
                    Some(last_hash.len()),
                )
                .map_err(fs::io_err)?;
        }

        if file_lock.get_file(&block_name(WRITE_AHEAD)).is_none() {
            // if there were no mutations committed after the chain was created,
            // it may not have a write-ahead block on the filesystem, so just create one now
            file_lock
                .create_file(
                    block_name(WRITE_AHEAD),
                    ChainBlock::new(last_hash.clone()),
                    Some(last_hash.len()),
                )
                .map_err(fs::io_err)?;
        }

        if file_lock.get_file(&block_name(latest)).is_none() {
            // if there were no mutations committed after the chain was created,
            // it may not have an initial block on the filesystem, so just create one now
            file_lock
                .create_file(
                    block_name(latest),
                    ChainBlock::new(last_hash.clone()),
                    Some(last_hash.len()),
                )
                .map_err(fs::io_err)?;
        }

        while let Some(block) = file_lock.get_file(&block_name(latest)) {
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

        let latest = if latest == 0 { 0 } else { latest - 1 };
        Ok(Self::new(file.clone(), store, latest, cutoff))
    }
}

#[async_trait]
impl Transact for History {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit chain history {}", txn_id);

        // assume `self.store` has already been committed by calling `write_ahead`

        let mut file = self.file.write().await;
        let write_ahead = file
            .get_file(&block_name(WRITE_AHEAD))
            .expect("write-ahead log");

        let needs_sync = {
            let mut write_ahead: FileWriteGuard<_, ChainBlock> =
                write_ahead.write().await.expect("write-ahead lock");

            if let Some(mutations) = write_ahead.mutations.remove(txn_id) {
                let mut latest = self
                    .latest
                    .write(*txn_id)
                    .await
                    .expect("latest block ordinal");

                let latest_block = file.get_file(&block_name(*latest)).expect("latest block");

                {
                    let mut latest_block: FileWriteGuard<_, ChainBlock> =
                        latest_block.write().await.expect("latest block write lock");

                    latest_block.mutations.insert(*txn_id, mutations);

                    if latest_block.size().await.expect("block size") > BLOCK_SIZE {
                        let mut cutoff = self.cutoff.write(*txn_id).await.expect("block cutoff id");

                        assert!(
                            txn_id >= &cutoff,
                            "cannot commit transaction {} since a block has already been committed at {}",
                            txn_id,
                            *cutoff
                        );

                        *cutoff = *txn_id;

                        let hash = latest_block.hash();

                        let new_block = file
                            .create_file(
                                block_name(*latest),
                                ChainBlock::new(hash.to_vec()),
                                Some(hash.len()),
                            )
                            .expect("new chain block");

                        new_block.sync(true).await.expect("sync new chain block");

                        *latest += 1;
                    }
                }

                latest_block
                    .sync(false)
                    .await
                    .expect("sync latest chain block");

                true
            } else {
                false
            }
        };

        if needs_sync {
            write_ahead
                .sync(false)
                .await
                .expect("sync write-ahead log after commit");
        }

        join!(self.latest.commit(txn_id), self.cutoff.commit(txn_id));
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(
            self.store.finalize(txn_id),
            self.latest.finalize(txn_id),
            self.cutoff.finalize(txn_id)
        );

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

        let store = {
            let mut dir = dir.write(txn_id).map_err(de::Error::custom).await?;
            dir.create_dir(STORE.into())
                .map_ok(Store::new)
                .map_err(de::Error::custom)
                .await?
        };

        let file = {
            let mut dir = dir.into_inner().write().await;
            dir.create_dir(block_name(CHAIN))
                .map_err(de::Error::custom)?
        };

        let mut file_lock = file.write().await;

        file_lock
            .create_file(
                block_name(PENDING),
                ChainBlock::new(null_hash.to_vec()),
                Some(null_hash.len()),
            )
            .map_err(de::Error::custom)?;

        file_lock
            .create_file(
                block_name(WRITE_AHEAD),
                ChainBlock::new(null_hash.to_vec()),
                Some(null_hash.len()),
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

        let latest = if i == 0 { 0 } else { i - 1 };
        Ok(History::new(file, store, latest, txn_id))
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

async fn replay_and_save(
    subject: &Subject,
    txn: &Txn,
    txn_id: TxnId,
    ops: &[Mutation],
    source: &Store,
    dest: &Store,
    block: &mut ChainBlock,
) -> TCResult<()> {
    for op in ops {
        match op {
            Mutation::Delete(path, key) => {
                subject.delete(txn, path, key.clone()).await?;
                block.append_delete(txn_id, path.clone(), key.clone())
            }
            Mutation::Put(path, key, original_hash) => {
                let state = source.resolve(txn, original_hash.clone()).await?;
                subject.put(txn, path, key.clone(), state.clone()).await?;

                let computed_hash = dest.save_state(txn, state).await?;
                if &computed_hash != original_hash {
                    return Err(TCError::unsupported(format!(
                        "cannot replicate state with inconsistent hash {} vs {}",
                        original_hash, computed_hash
                    )));
                }

                block.append_put(txn_id, path.clone(), key.clone(), computed_hash)
            }
        }
    }

    Ok(())
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
