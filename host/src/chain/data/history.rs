use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt;
use std::iter;

use async_hash::generic_array::GenericArray;
use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use freqfs::{DirLock, DirWriteGuard, FileLock, FileReadGuard, FileReadGuardOwned, FileWriteGuard};
use futures::stream::{self, StreamExt};
use futures::{join, try_join, TryFutureExt, TryStreamExt};
use get_size::GetSize;
use log::{debug, error, trace};
use safecast::*;

use tc_error::*;
use tc_transact::fs::{Dir, Persist};
use tc_transact::lock::TxnLock;
use tc_transact::{AsyncHash, IntoView, Sha256, Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, Label, Map, TCBoxStream, TCBoxTryStream, Tuple};

use crate::chain::{null_hash, BLOCK_SIZE, CHAIN};
use crate::fs;
use crate::fs::CacheBlock;
use crate::route::{Public, Route};
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
            latest: TxnLock::new(latest),
            cutoff: TxnLock::new(cutoff),
        }
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    pub async fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        debug!("History::append_delete {} {}", txn_id, key);
        let mut block = self.write_pending().await?;
        block.append_delete(txn_id, key);
        Ok(())
    }

    pub async fn append_put(&self, txn: &Txn, key: Value, value: State) -> TCResult<()> {
        let txn_id = *txn.id();
        let value = self.store.save_state(txn, value).await?;

        debug!("History::append_put {} {} {:?}", txn_id, key, value);

        let mut block = self.write_pending().await?;
        block.append_put(txn_id, key, value);

        Ok(())
    }

    async fn read_block(
        &self,
        block_id: u64,
    ) -> TCResult<freqfs::FileReadGuardOwned<fs::CacheBlock, ChainBlock>> {
        let file = self.file.read().await;
        let block = file
            .get_file(&block_id)
            .ok_or_else(|| TCError::not_found(format!("chain block {}", block_id)))?;

        block.read_owned().map_err(TCError::from).await
    }

    async fn write_block(
        &self,
        block_id: u64,
    ) -> TCResult<freqfs::FileWriteGuardOwned<fs::CacheBlock, ChainBlock>> {
        let file = self.file.read().await;
        let block: &FileLock<fs::CacheBlock> = file
            .get_file(&block_id)
            .ok_or_else(|| TCError::not_found(format!("chain block {}", block_id)))?;

        block.write_owned().map_err(TCError::from).await
    }

    pub async fn read_pending(
        &self,
    ) -> TCResult<freqfs::FileReadGuardOwned<fs::CacheBlock, ChainBlock>> {
        let file = self.file.read().await;
        let block: &FileLock<fs::CacheBlock> = file
            .get_file(PENDING)
            .ok_or_else(|| unexpected!("BlockChain is missing its pending block"))?;

        block.read_owned().map_err(TCError::from).await
    }

    pub async fn write_pending(
        &self,
    ) -> TCResult<freqfs::FileWriteGuardOwned<fs::CacheBlock, ChainBlock>> {
        let file = self.file.read().await;
        let block: &FileLock<fs::CacheBlock> = file
            .get_file(PENDING)
            .ok_or_else(|| unexpected!("BlockChain is missing its pending block"))?;

        block.write_owned().map_err(TCError::from).await
    }

    pub async fn read_log(
        &self,
    ) -> TCResult<freqfs::FileReadGuardOwned<fs::CacheBlock, ChainBlock>> {
        let log: FileLock<fs::CacheBlock> = {
            let file = self.file.read().await;
            file.get_file(WRITE_AHEAD).expect("write-ahead log").clone()
        };

        log.into_read().map_err(TCError::from).await
    }

    pub async fn replicate<T>(&self, txn: &Txn, subject: &T, other: Self) -> TCResult<()>
    where
        T: Route + fmt::Debug,
    {
        let err_divergent =
            |block_id| bad_request!("chain to replicate diverges at block {}", block_id);

        debug!("replicate chain history");

        let (latest, other_latest) =
            try_join!(self.latest.read(*txn.id()), other.latest.read(*txn.id()))?;

        debug!("chain to replicate ends with block {}", *other_latest);

        if (*latest) > (*other_latest) {
            return Err(bad_request!(
                "a Chain with {} blocks cannot replicate a Chain with {} blocks",
                *latest,
                *other_latest,
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
                    subject,
                    txn,
                    *txn_id,
                    ops,
                    &other.store,
                    &self.store,
                    &mut dest,
                )
                .await?;
            }

            let last_hash = dest.current_hash().to_vec();
            if &last_hash[..] != &source.current_hash()[..] {
                return Err(unexpected!("{}", err_divergent(*latest)));
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

            let block = ChainBlock::new(last_hash.to_vec());
            let size_hint = block.get_size();
            let dest = this_file.create_file(block_id.to_string(), block, size_hint)?;

            let mut dest: FileWriteGuard<ChainBlock> = dest.write().await?;

            for (txn_id, ops) in &source.mutations {
                assert!(!dest.mutations.contains_key(txn_id));

                replay_and_save(
                    subject,
                    txn,
                    *txn_id,
                    ops,
                    &other.store,
                    &self.store,
                    &mut dest,
                )
                .await?;
            }

            last_hash = dest.current_hash().to_vec();
            if &last_hash[..] != &source.current_hash()[..] {
                return Err(unexpected!("{}", err_divergent(block_id)));
            }
        }

        Ok(())
    }

    pub async fn write_ahead(&self, txn_id: TxnId) {
        self.store.commit(txn_id).await;

        let file = self.file.read().await;
        let pending: &FileLock<fs::CacheBlock> =
            file.get_file(PENDING).expect("pending transactions");
        let mut pending: FileWriteGuard<ChainBlock> =
            pending.write().await.expect("pending block write lock");

        if let Some(mutations) = pending.mutations.remove(&txn_id) {
            let write_ahead: &FileLock<fs::CacheBlock> =
                file.get_file(WRITE_AHEAD).expect("write-ahead log");

            {
                let mut write_ahead: FileWriteGuard<ChainBlock> = write_ahead
                    .write()
                    .await
                    .expect("write-ahead log write lock");

                write_ahead.mutations.insert(txn_id, mutations);
            }

            write_ahead.sync().await.expect("sync write-ahead log");
        }
    }
}

#[async_trait]
impl Persist<fs::CacheBlock> for History {
    type Txn = Txn;
    type Schema = ();

    async fn create(txn_id: TxnId, _schema: Self::Schema, dir: fs::Dir) -> TCResult<Self> {
        let store = dir
            .create_dir(txn_id, STORE.into())
            .map_ok(Store::new)
            .await?;

        let file = dir
            .into_inner()
            .try_write()
            .and_then(|mut dir| dir.create_dir(CHAIN.to_string()))?;

        let mut file_lock = file.try_write()?;

        let cutoff = txn_id;
        let latest = 0;

        create_block(&mut file_lock, PENDING)?;
        create_block(&mut file_lock, WRITE_AHEAD)?;
        create_block(&mut file_lock, latest)?;

        Ok(Self::new(file.clone(), store, latest, cutoff))
    }

    async fn load(txn_id: TxnId, _schema: Self::Schema, dir: fs::Dir) -> TCResult<Self> {
        let store = dir
            .get_or_create_dir(txn_id, STORE.into())
            .map_ok(Store::new)
            .await?;

        let file = dir
            .into_inner()
            .try_write()
            .and_then(|mut dir| dir.get_or_create_dir(CHAIN.to_string()))?;

        let mut file_lock = file.try_write()?;

        let mut cutoff = txn_id;
        let mut latest = 0;

        get_or_create_block(&mut file_lock, PENDING.to_string())?;
        get_or_create_block(&mut file_lock, WRITE_AHEAD.to_string())?;
        get_or_create_block(&mut file_lock, latest.to_string())?;

        while let Some(_block) = file_lock.get_file(&latest) {
            latest += 1;
        }

        let mut latest = 0;
        let mut last_hash = Bytes::from(null_hash().to_vec());
        while let Some(block) = file_lock.get_file(&latest) {
            let block: FileReadGuard<ChainBlock> = block.read().await?;

            if block.last_hash() == &last_hash {
                last_hash = block.last_hash().clone();
            } else {
                return Err(unexpected!(
                    "block {} hash does not match previous block",
                    latest
                ));
            }

            cutoff = block.mutations.keys().last().copied().unwrap_or(cutoff);
            latest += 1;
        }

        let latest = if latest == 0 { 0 } else { latest - 1 };
        Ok(Self::new(file.clone(), store, latest, cutoff))
    }

    fn dir(&self) -> DirLock<fs::CacheBlock> {
        self.file.clone()
    }
}

#[async_trait]
impl AsyncHash<fs::CacheBlock> for History {
    type Txn = Txn;

    async fn hash(self, txn: &Self::Txn) -> TCResult<Output<Sha256>> {
        let latest_block_id = self.latest.read(*txn.id()).await?;
        let latest_block = self.read_block(*latest_block_id).await?;

        let latest_block = if latest_block.mutations.is_empty() {
            if *latest_block_id == 0 {
                latest_block
            } else {
                self.read_block(*latest_block_id - 1).await?
            }
        } else {
            latest_block
        };

        if let Some(past_txn_id) = latest_block.mutations.keys().next() {
            if past_txn_id > txn.id() {
                return Err(conflict!(
                    "requested a hash {} too far before the present {}",
                    past_txn_id,
                    txn.id()
                ));
            }
        }

        let log = self.read_log().await?;
        if let Some(mutations) = log.mutations.get(txn.id()) {
            let mutations = latest_block
                .mutations
                .iter()
                .take_while(|(past_txn_id, _)| *past_txn_id <= txn.id())
                .chain(iter::once((txn.id(), mutations)));

            Ok(ChainBlock::hash(latest_block.last_hash(), mutations))
        } else {
            let pending = self.read_pending().await?;
            if let Some(mutations) = pending.mutations.get(txn.id()) {
                let mutations = latest_block
                    .mutations
                    .iter()
                    .take_while(|(past_txn_id, _)| *past_txn_id <= txn.id())
                    .chain(iter::once((txn.id(), mutations)));

                Ok(ChainBlock::hash(latest_block.last_hash(), mutations))
            } else {
                // TODO: validate the length of the hash before calling clone_from_slice
                Ok(GenericArray::clone_from_slice(latest_block.last_hash()))
            }
        }
    }
}

fn get_or_create_block(
    cache: &mut DirWriteGuard<fs::CacheBlock>,
    name: String,
) -> TCResult<FileLock<fs::CacheBlock>> {
    if let Some(file) = cache.get_file(&name) {
        Ok(file.clone())
    } else {
        create_block(cache, name)
    }
}

fn create_block<I: fmt::Display>(
    cache: &mut DirWriteGuard<fs::CacheBlock>,
    name: I,
) -> TCResult<FileLock<fs::CacheBlock>> {
    let last_hash = Bytes::from(null_hash().to_vec());

    let block = ChainBlock::new(last_hash.clone());
    let size_hint = block.get_size();

    cache
        .create_file(name.to_string(), block, size_hint)
        .map_err(TCError::from)
}

#[async_trait]
impl Transact for History {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) {
        debug!("commit chain history {}", txn_id);

        // assume `self.store` has already been committed by calling `write_ahead`

        let mut file = self.file.write().await;
        trace!("got write lock on chain history file");

        let write_ahead = file.get_file(WRITE_AHEAD).expect("write-ahead log").clone();

        let needs_sync = {
            let mut write_ahead: FileWriteGuard<ChainBlock> =
                write_ahead.write().await.expect("write-ahead lock");

            trace!("locked write-ahead block for writing");

            if let Some(mutations) = write_ahead.mutations.remove(&txn_id) {
                trace!("locking latest block ordinal for writing...");

                let mut latest = self
                    .latest
                    .write(txn_id)
                    .await
                    .expect("latest block ordinal");

                trace!("locked latest block ordinal for writing");

                let latest_block = file.get_file(&*latest).expect("latest block").clone();

                {
                    let mut latest_block: FileWriteGuard<ChainBlock> =
                        latest_block.write().await.expect("latest block write lock");

                    trace!("locked latest ChainBlock for writing");

                    latest_block.mutations.insert(txn_id, mutations);

                    if latest_block.size().await.expect("block size") > BLOCK_SIZE {
                        let mut cutoff = self.cutoff.write(txn_id).await.expect("block cutoff id");

                        trace!("locked block cutoff ID for writing");

                        assert!(
                            &txn_id >= &*cutoff,
                            "cannot commit transaction {} since a block has already been committed at {}",
                            txn_id,
                            *cutoff
                        );

                        *cutoff = txn_id;

                        let hash = latest_block.current_hash();

                        let block = ChainBlock::new(hash.to_vec());
                        let size_hint = block.get_size();
                        let new_block = file
                            .create_file(latest.to_string(), block, size_hint)
                            .expect("new chain block");

                        new_block.sync().await.expect("sync new chain block");

                        trace!("sync'd new ChainBlock to disk");

                        *latest += 1;
                    }
                }

                latest_block.sync().await.expect("sync latest chain block");

                trace!("sync'd last ChainBlock to disk");

                true
            } else {
                false
            }
        };

        if needs_sync {
            write_ahead
                .sync()
                .await
                .expect("sync write-ahead log after commit");
            trace!("sync'd write-ahead block to disk");
        }

        self.latest.commit(txn_id);
        self.cutoff.commit(txn_id);
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let file = self.file.read().await;
        let mut pending: FileWriteGuard<ChainBlock> = file
            .get_file(PENDING)
            .expect("pending transactions")
            .write()
            .await
            .expect("pending transaction lock");

        pending.mutations.remove(txn_id);

        self.latest.rollback(txn_id);
        self.cutoff.rollback(txn_id);
        self.store.rollback(txn_id).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.latest.finalize(*txn_id);
        self.cutoff.finalize(*txn_id);
        self.store.finalize(txn_id).await;

        let file = self.file.read().await;
        let mut pending: FileWriteGuard<ChainBlock> = file
            .get_file(PENDING)
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

        let store = {
            let dir = self
                .txn
                .context()
                .write()
                .await
                .create_dir(STORE.to_string())
                .map_err(de::Error::custom)?;

            let dir = fs::Dir::load(txn_id, dir)
                .map_err(de::Error::custom)
                .await?;

            Store::new(dir)
        };

        let file = {
            let mut dir = self.txn.context().write().await;
            dir.create_dir(CHAIN.to_string())
                .map_err(de::Error::custom)?
        };

        let mut file_lock = file.write().await;

        let block = ChainBlock::new(null_hash.to_vec());
        let size_hint = block.get_size();
        file_lock
            .create_file(PENDING.into(), block, size_hint)
            .map_err(de::Error::custom)?;

        let block = ChainBlock::new(null_hash.to_vec());
        let size_hint = block.get_size();
        file_lock
            .create_file(WRITE_AHEAD.into(), block, size_hint)
            .map_err(de::Error::custom)?;

        let subcontext = |i: u64| self.txn.subcontext(i.into()).map_err(de::Error::custom);

        let mut i = 0u64;
        let mut last_hash = null_hash.clone();

        while let Some(state) = seq.next_element::<State>(subcontext(i).await?).await? {
            let (hash, block_data): (Bytes, Map<Tuple<State>>) = state
                .try_cast_into(|s| de::Error::invalid_type(format!("{s:?}"), "a chain block"))?;

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
            last_hash = block.current_hash();

            let size_hint = block.get_size();
            file_lock
                .create_file(i.to_string(), block, size_hint)
                .map_err(de::Error::custom)?;

            i += 1;
        }

        std::mem::drop(file_lock);

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
            if op.matches::<(Value,)>() {
                let (key,) = op.opt_cast_into().unwrap();
                parsed.push(Mutation::Delete(key));
            } else if op.matches::<(Value, State)>() {
                let (key, value) = op.opt_cast_into().unwrap();
                let value = store.save_state(txn, value).await?;
                parsed.push(Mutation::Put(key, value));
            } else {
                return Err(unexpected!("unable to parse historical mutation {:?}", op,));
            }
        }

        mutations.insert(past_txn_id, parsed);
    }

    Ok(mutations)
}

async fn replay_and_save<T>(
    subject: &T,
    txn: &Txn,
    txn_id: TxnId,
    ops: &[Mutation],
    source: &Store,
    dest: &Store,
    block: &mut ChainBlock,
) -> TCResult<()>
where
    T: Route + fmt::Debug,
{
    for op in ops {
        match op {
            Mutation::Delete(key) => {
                subject.delete(txn, &[], key.clone()).await?;
                block.append_delete(txn_id, key.clone())
            }
            Mutation::Put(key, original_hash) => {
                let state = source.resolve(*txn.id(), original_hash.clone()).await?;
                subject.put(txn, &[], key.clone(), state.clone()).await?;

                let computed_hash = dest.save_state(txn, state).await?;
                if &computed_hash != original_hash {
                    return Err(bad_request!(
                        "cannot replicate state with inconsistent hash {:?} vs {:?}",
                        original_hash,
                        computed_hash
                    ));
                }

                block.append_put(txn_id, key.clone(), computed_hash)
            }
        }
    }

    Ok(())
}

pub type HistoryView<'en> =
    en::SeqStream<TCResult<HistoryBlockView<'en>>, TCBoxTryStream<'en, HistoryBlockView<'en>>>;

#[async_trait]
impl<'en> IntoView<'en, fs::CacheBlock> for History {
    type Txn = Txn;
    type View = HistoryView<'en>;

    async fn into_view(self, txn: Txn) -> TCResult<Self::View> {
        debug!("History::into_view");

        let latest = self.latest.read(*txn.id()).await?;

        let file = self.file.read_owned().await;

        let seq = stream::iter(0..((*latest) + 1))
            .map(move |block_id| {
                file.get_file(&block_id)
                    .cloned()
                    .ok_or_else(|| unexpected!("missing chain block"))
            })
            .and_then(|block| {
                Box::pin(async move { block.read_owned().map_err(TCError::from).await })
            })
            .map_ok(move |block: FileReadGuardOwned<CacheBlock, ChainBlock>| {
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
        Mutation::Delete(key) => Ok(MutationView::Delete(key)),
        Mutation::Put(key, value) if value.is_ref() => {
            debug!("historical mutation: PUT {} <- {:?}", key, value);

            let value = history
                .store
                .resolve(*txn.id(), value)
                .map_err(|err| {
                    error!("unable to load historical Chain data: {}", err);
                    err
                })
                .await?;

            let value = value.into_view(txn).await?;
            Ok(MutationView::Put(key, value))
        }
        Mutation::Put(key, value) => {
            let value = State::from(value).into_view(txn.clone()).await?;

            Ok(MutationView::Put(key, value))
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
    Delete(Value),
    Put(Value, StateView<'en>),
}

impl<'en> en::IntoStream<'en> for MutationView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Delete(key) => (key,).into_stream(encoder),
            Self::Put(key, value) => (key, value).into_stream(encoder),
        }
    }
}
