use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::iter;
use std::sync::Arc;

use async_hash::generic_array::GenericArray;
use async_hash::{Output, Sha256};
use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use freqfs::*;
use futures::stream::{self, StreamExt};
use futures::{join, try_join, TryFutureExt, TryStreamExt};
use get_size::GetSize;
use log::{debug, error, info, trace};
use safecast::*;
use tokio::sync::{mpsc, oneshot, RwLock};

use tc_collection::btree::Node as BTreeNode;
use tc_collection::tensor::{DenseCacheFile, Node as TensorNode};
use tc_collection::Collection;
use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::fs;
use tc_transact::lock::TxnTaskQueue;
use tc_transact::public::{Public, Route, StateInstance};
use tc_transact::{AsyncHash, IntoView, Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, Label, Map, TCBoxStream, TCBoxTryStream, ThreadSafe, Tuple};

use crate::{new_queue, null_hash, BLOCK_SIZE, CHAIN};

use super::block::{ChainBlock, MutationPending, MutationRecord};
use super::store::{Store, StoreEntry, StoreEntryView};

const STORE: Label = label("store");
const WRITE_AHEAD: &str = "write_ahead";

struct Commit(TxnId, oneshot::Sender<()>);

impl Commit {
    fn new(txn_id: TxnId) -> (Self, oneshot::Receiver<()>) {
        let (tx, rx) = oneshot::channel();
        (Self(txn_id, tx), rx)
    }
}

impl Eq for Commit {}

impl PartialEq for Commit {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Ord for Commit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl PartialOrd for Commit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct WriteAhead(TxnId, Vec<MutationRecord>, oneshot::Sender<()>);

impl WriteAhead {
    fn new(txn_id: TxnId, mutations: Vec<MutationRecord>) -> (Self, oneshot::Receiver<()>) {
        let (tx, rx) = oneshot::channel();
        (Self(txn_id, mutations, tx), rx)
    }
}

pub struct History<State: StateInstance> {
    queue: TxnTaskQueue<MutationPending<State::Txn, State::FE>, TCResult<MutationRecord>>,
    file: DirLock<State::FE>,
    store: Store<State::Txn, State::FE>,
    latest: Arc<RwLock<u64>>,
    cutoff: Arc<RwLock<TxnId>>,
    write_ahead_log: mpsc::UnboundedSender<WriteAhead>,
    commit_log: mpsc::UnboundedSender<Commit>,
}

impl<State: StateInstance> Clone for History<State> {
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
            file: self.file.clone(),
            store: self.store.clone(),
            latest: self.latest.clone(),
            cutoff: self.cutoff.clone(),
            write_ahead_log: self.write_ahead_log.clone(),
            commit_log: self.commit_log.clone(),
        }
    }
}

impl<State> History<State>
where
    State: StateInstance,
    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
{
    fn new(
        file: DirLock<State::FE>,
        writeahead_block: FileLock<State::FE>,
        store: Store<State::Txn, State::FE>,
        latest: u64,
        cutoff: TxnId,
    ) -> Self {
        debug_assert!(file.try_read().expect("history").contains(&latest));

        let latest = Arc::new(RwLock::new(latest));
        let cutoff = Arc::new(RwLock::new(cutoff));

        let queue = new_queue::<State>(store.clone());
        let write_ahead_log = spawn_writeahead_thread::<State>(writeahead_block);
        let commit_log = spawn_commit_thread::<State>(file.clone(), cutoff.clone(), latest.clone());

        Self {
            queue,
            file,
            store,
            latest,
            cutoff,
            write_ahead_log,
            commit_log,
        }
    }
}

impl<State: StateInstance> History<State> {
    pub fn store(&self) -> &Store<State::Txn, State::FE> {
        &self.store
    }
}

impl<State> History<State>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
{
    pub fn append_delete(&self, txn_id: TxnId, key: Value) -> TCResult<()> {
        debug!("History::append_delete {} {}", txn_id, key);
        self.queue
            .push(txn_id, MutationPending::Delete(key))
            .map_err(TCError::from)
    }

    async fn read_block(
        &self,
        block_id: u64,
    ) -> TCResult<freqfs::FileReadGuardOwned<State::FE, ChainBlock>> {
        let file = self.file.read().await;
        let block = file
            .get_file(&block_id)
            .ok_or_else(|| TCError::not_found(format!("chain block {}", block_id)))?;

        block.read_owned().map_err(TCError::from).await
    }

    async fn write_block(
        &self,
        block_id: u64,
    ) -> TCResult<freqfs::FileWriteGuardOwned<State::FE, ChainBlock>> {
        let file = self.file.read().await;
        let block: &FileLock<State::FE> = file
            .get_file(&block_id)
            .ok_or_else(|| TCError::not_found(format!("chain block {}", block_id)))?;

        block.write_owned().map_err(TCError::from).await
    }

    pub async fn read_log(&self) -> TCResult<freqfs::FileReadGuardOwned<State::FE, ChainBlock>> {
        let log: FileLock<State::FE> = {
            let file = self.file.read().await;
            file.get_file(WRITE_AHEAD).expect("write-ahead log").clone()
        };

        log.into_read().map_err(TCError::from).await
    }

    pub async fn write_ahead(&self, txn_id: TxnId) {
        let handles = self.queue.commit(txn_id).await;

        let mutations = handles
            .into_iter()
            .collect::<TCResult<Vec<_>>>()
            .expect("mutations");

        if mutations.is_empty() {
            return;
        }

        self.store.commit(txn_id).await;

        let (message, rx) = WriteAhead::new(txn_id, mutations);

        self.write_ahead_log
            .send(message)
            .expect("send write-ahead message");

        rx.await.expect("write-ahead confirmation");
    }
}

impl<State> History<State>
where
    State: StateInstance,
    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
{
    pub fn append_put(&self, txn: State::Txn, key: Value, value: State) -> TCResult<()>
    where
        Collection<State::Txn, State::FE>: TryCastFrom<State>,
        Scalar: TryCastFrom<State>,
    {
        debug!("History::append_put {} {} {:?}", txn.id(), key, value);

        let value = StoreEntry::try_from_state(value)?;

        self.queue
            .push(*txn.id(), MutationPending::Put(txn, key, value))
            .map_err(TCError::from)
    }

    pub async fn replicate<T>(&self, txn: &State::Txn, subject: &T, other: Self) -> TCResult<()>
    where
        State: From<Collection<State::Txn, State::FE>> + From<Scalar>,
        T: Route<State> + fmt::Debug,
        Collection<State::Txn, State::FE>: TryCastFrom<State>,
        Scalar: TryCastFrom<State>,
    {
        let err_divergent =
            |block_id| bad_request!("chain to replicate diverges at block {}", block_id);

        info!("replicate {subject:?} from chain history {other:?}");

        let (latest, other_latest) = join!(self.latest.read(), other.latest.read());

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
            let (mut dest, source) = try_join!(
                self.write_block(*latest)
                    .map_err(|cause| internal!("missing chain block {}: {cause}", *latest)),
                other
                    .read_block(*latest)
                    .map_err(|cause| bad_request!("invalid source Chain: {cause}"))
            )?;

            if let Some(txn_id) = dest.mutations.keys().last() {
                latest_txn_id = Some(*txn_id);
            }

            trace!("the latest txn id in this chain history is {:?} (compare to {:?} in the history to replicate from)", latest_txn_id, source.mutations.keys().last());

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
                return Err(internal!("{}", err_divergent(*latest)));
            }

            last_hash
        };

        // if the other chain has the same number of blocks, replication is complete
        if *latest == *other_latest {
            trace!("the chain to replicate from has the same hash as this chain");
            return Ok(());
        } else {
            trace!("the chain to replicate from has a different hash than this chain...");
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
                return Err(internal!("{}", err_divergent(block_id)));
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<State> fs::Persist<State::FE> for History<State>
where
    State: StateInstance,
    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
{
    type Txn = State::Txn;
    type Schema = ();

    async fn create(
        txn_id: TxnId,
        _schema: Self::Schema,
        dir: fs::Dir<State::FE>,
    ) -> TCResult<Self> {
        debug!("History::create");

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

        let writeahead_block = create_block(&mut file_lock, WRITE_AHEAD)?;
        create_block(&mut file_lock, latest)?;

        std::mem::drop(file_lock);

        Ok(Self::new(
            file.clone(),
            writeahead_block,
            store,
            latest,
            cutoff,
        ))
    }

    async fn load(txn_id: TxnId, _schema: Self::Schema, dir: fs::Dir<State::FE>) -> TCResult<Self> {
        debug!("History::load");

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

        let writeahead_block = get_or_create_block(&mut file_lock, WRITE_AHEAD.to_string())?;

        let mut last_hash = Bytes::from(null_hash().to_vec());
        while let Some(block) = file_lock.get_file(&latest) {
            let block: FileReadGuard<ChainBlock> = block.read().await?;

            if block.last_hash() == &last_hash {
                last_hash = block.last_hash().clone();
            } else {
                return Err(internal!(
                    "block {} hash does not match previous block",
                    latest
                ));
            }

            cutoff = block.mutations.keys().last().copied().unwrap_or(cutoff);
            latest += 1;
        }

        let latest = if latest == 0 {
            create_block(&mut file_lock, latest.to_string())?;
            0
        } else {
            latest - 1
        };

        assert!(
            file_lock.contains(&latest),
            "Chain is missing block {latest}"
        );

        std::mem::drop(file_lock);

        Ok(Self::new(
            file.clone(),
            writeahead_block,
            store,
            latest,
            cutoff,
        ))
    }

    fn dir(&self) -> DirLock<State::FE> {
        self.file.clone()
    }
}

#[async_trait]
impl<State> AsyncHash for History<State>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let latest_block_id = self.latest.read().await;
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
            if *past_txn_id > txn_id {
                return Err(conflict!(
                    "requested a hash {} too far before the present {}",
                    past_txn_id,
                    txn_id,
                ));
            }
        }

        let log = self.read_log().await?;
        if let Some(mutations) = log.mutations.get(&txn_id) {
            let mutations = latest_block
                .mutations
                .iter()
                .take_while(|(past_txn_id, _)| *past_txn_id <= &txn_id)
                .chain(iter::once((&txn_id, mutations)));

            Ok(ChainBlock::hash(latest_block.last_hash(), mutations))
        } else {
            let pending = self.queue.peek(&txn_id).await?;

            if let Some(pending_mutations) = pending {
                if let Some(err) = pending_mutations
                    .iter()
                    .map(Result::as_ref)
                    .filter_map(Result::err)
                    .next()
                {
                    return Err(err.clone());
                }

                let pending_mutations = pending_mutations
                    .iter()
                    .map(Result::as_ref)
                    .filter_map(Result::ok);

                let mutations = latest_block
                    .mutations
                    .iter()
                    .take_while(|(past_txn_id, _)| *past_txn_id <= &txn_id);

                Ok(ChainBlock::pending_hash(
                    latest_block.last_hash(),
                    mutations,
                    &txn_id,
                    pending_mutations,
                ))
            } else {
                // TODO: validate the length of the hash before calling clone_from_slice
                Ok(GenericArray::clone_from_slice(latest_block.last_hash()))
            }
        }
    }
}

#[async_trait]
impl<State> Transact for History<State>
where
    State: StateInstance,
    State::FE: AsType<ChainBlock> + for<'a> fs::FileSave<'a>,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) {
        debug!("commit chain history {}", txn_id);

        // assume `self.store` has already been committed by calling `write_ahead`

        let (message, rx) = Commit::new(txn_id);
        self.commit_log.send(message).expect("send commit message");
        rx.await.expect("commit confirmation");
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.queue.rollback(txn_id);
        self.store.rollback(txn_id).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.queue.finalize(*txn_id);
        self.store.finalize(txn_id).await;
    }
}

#[async_trait]
impl<'en, State> IntoView<'en, State::FE> for History<State>
where
    State: StateInstance,
    State::FE: DenseCacheFile + AsType<ChainBlock> + AsType<BTreeNode> + AsType<TensorNode>,
{
    type Txn = State::Txn;
    type View = HistoryView<'en>;

    async fn into_view(self, txn: State::Txn) -> TCResult<Self::View> {
        debug!("History::into_view");

        let latest = self.latest.clone().read_owned().await;
        let file = self.file.read_owned().await;

        let seq = stream::iter(0..((*latest) + 1))
            .map(move |block_id| {
                file.get_file(&block_id)
                    .cloned()
                    .ok_or_else(|| internal!("missing chain block"))
            })
            .and_then(|block| {
                Box::pin(async move { block.read_owned().map_err(TCError::from).await })
            })
            .map_ok(move |block: FileReadGuardOwned<State::FE, ChainBlock>| {
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

#[async_trait]
impl<State> de::FromStream for History<State>
where
    State: StateInstance + de::FromStream<Context = State::Txn> + From<Scalar>,
    State::FE: DenseCacheFile
        + AsType<ChainBlock>
        + AsType<BTreeNode>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    (Bytes, Map<Tuple<State>>): TryCastFrom<State>,
    (Value,): TryCastFrom<State>,
    (Value, State): TryCastFrom<State>,
{
    type Context = State::Txn;

    async fn from_stream<D: de::Decoder>(
        txn: State::Txn,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_seq(HistoryVisitor { txn }).await
    }
}

impl<State: StateInstance> fmt::Debug for History<State> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a chain history")
    }
}

struct HistoryVisitor<State: StateInstance> {
    txn: State::Txn,
}

#[async_trait]
impl<State> de::Visitor for HistoryVisitor<State>
where
    State: StateInstance + de::FromStream<Context = State::Txn> + From<Scalar>,
    State::FE: DenseCacheFile
        + AsType<ChainBlock>
        + AsType<BTreeNode>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    (Bytes, Map<Tuple<State>>): TryCastFrom<State>,
    (Value,): TryCastFrom<State>,
    (Value, State): TryCastFrom<State>,
{
    type Value = History<State>;

    fn expecting() -> &'static str {
        "Chain history"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let null_hash = null_hash();
        let txn_id = *self.txn.id();

        let cxt = self.txn.context().map_err(de::Error::custom).await?;

        let store = {
            let dir = {
                let mut cxt = cxt.write().await;
                cxt.create_dir(STORE.to_string())
                    .map_err(de::Error::custom)?
            };

            let dir = fs::Dir::load(txn_id, dir)
                .map_err(de::Error::custom)
                .await?;

            Store::new(dir)
        };

        let file = {
            let mut cxt = cxt.write().await;
            cxt.create_dir(CHAIN.to_string())
                .map_err(de::Error::custom)?
        };

        let mut guard = file.write().await;

        let block = ChainBlock::new(null_hash.to_vec());
        let size_hint = block.get_size();
        let writeahead_block = guard
            .create_file(WRITE_AHEAD.into(), block, size_hint)
            .map_err(de::Error::custom)?;

        let mut i = 0u64;
        let mut last_hash = null_hash.clone();

        while let Some(state) = seq.next_element::<State>(self.txn.subcontext(i)).await? {
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
            guard
                .create_file(i.to_string(), block, size_hint)
                .map_err(de::Error::custom)?;

            i += 1;
        }

        let latest = if i == 0 {
            create_block(&mut guard, 0).map_err(de::Error::custom)?;
            0
        } else {
            i - 1
        };

        assert!(guard.contains(&latest));

        std::mem::drop(guard);

        Ok(History::new(file, writeahead_block, store, latest, txn_id))
    }
}

fn spawn_writeahead_thread<State: StateInstance>(
    log: FileLock<State::FE>,
) -> mpsc::UnboundedSender<WriteAhead>
where
    State::FE: AsType<ChainBlock> + for<'a> FileSave<'a>,
{
    let (tx, mut rx) = mpsc::unbounded_channel();

    let mut oneshot_buffer = Vec::new();

    fn handle_message(
        block: &mut ChainBlock,
        buffer: &mut Vec<oneshot::Sender<()>>,
        message: WriteAhead,
    ) -> bool {
        let WriteAhead(txn_id, mutations, tx) = message;
        debug_assert!(!mutations.is_empty());
        buffer.push(tx);
        block.mutations.insert(txn_id, mutations);
        true
    }

    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            {
                let mut block = log.write().await.expect("write-ahead log");

                handle_message(&mut *block, &mut oneshot_buffer, msg);

                while let Ok(msg) = rx.try_recv() {
                    handle_message(&mut *block, &mut oneshot_buffer, msg);
                }
            }

            log.sync().await.expect("sync write-ahead log");

            for tx in oneshot_buffer.drain(..) {
                tx.send(()).expect("confirm write-ahead");
            }
        }
    });

    tx
}

fn spawn_commit_thread<State: StateInstance>(
    blocks: DirLock<State::FE>,
    cutoff: Arc<RwLock<TxnId>>,
    latest: Arc<RwLock<u64>>,
) -> mpsc::UnboundedSender<Commit>
where
    State::FE: AsType<ChainBlock> + for<'a> FileSave<'a>,
{
    let write_ahead = blocks
        .try_read()
        .expect("blocks")
        .get_file(WRITE_AHEAD)
        .expect("write-ahead log")
        .clone();

    let (tx, mut rx) = mpsc::unbounded_channel();

    let mut message_buffer = Vec::new();
    let mut oneshot_buffer = Vec::new();

    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let mut latest = latest.write().await;
            let mut cutoff = cutoff.write().await;
            let mut write_ahead = write_ahead.write().await.expect("write-ahead log");
            let mut blocks = blocks.write().await;

            message_buffer.push(msg);

            while let Ok(msg) = rx.try_recv() {
                message_buffer.push(msg);
            }

            message_buffer.sort();

            let mut latest_block = blocks.get_file(&*latest).cloned().expect("latest block");
            let mut latest_block_mut = latest_block.write_owned().await.expect("latest block");

            for Commit(txn_id, tx) in message_buffer.drain(..) {
                let mutations = if let Some(mutations) = write_ahead.mutations.remove(&txn_id) {
                    mutations
                } else {
                    tx.send(()).expect("confirm empty commit");
                    continue;
                };

                // this condition is technically possible but should be extraordinarily rare
                // it prevents replicas from diverging in case of out-of-order commits
                assert!(
                    &txn_id >= &*cutoff,
                    "cannot commit transaction {} since a block has already been committed at {}",
                    txn_id,
                    *cutoff
                );

                oneshot_buffer.push(tx);

                latest_block_mut.mutations.insert(txn_id, mutations);

                if latest_block_mut.get_size() > BLOCK_SIZE {
                    *cutoff = txn_id;

                    let waiters = oneshot_buffer.drain(..).collect::<Vec<_>>();

                    tokio::spawn(async move {
                        latest_block.sync().await.expect("sync block");

                        for waiter in waiters {
                            waiter.send(()).expect("confirm commit");
                        }
                    });

                    let hash = latest_block_mut.current_hash();

                    *latest += 1;

                    let block = ChainBlock::new(hash.to_vec());
                    let size_hint = block.get_size();

                    latest_block = blocks
                        .create_file(latest.to_string(), block, size_hint)
                        .expect("new chain block");

                    latest_block_mut = latest_block.write_owned().await.expect("latest block");
                }
            }

            std::mem::drop(latest_block_mut);

            if !oneshot_buffer.is_empty() {
                latest_block.sync().await.expect("sync block");

                for waiter in oneshot_buffer.drain(..) {
                    waiter.send(()).expect("confirm commit");
                }
            }
        }
    });

    tx
}

async fn parse_block_state<State>(
    store: &Store<State::Txn, State::FE>,
    txn: &State::Txn,
    block_data: Map<Tuple<State>>,
) -> TCResult<BTreeMap<TxnId, Vec<MutationRecord>>>
where
    State: StateInstance,
    State::FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode>,
    State: StateInstance + From<Scalar>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    (Value,): TryCastFrom<State>,
    (Value, State): TryCastFrom<State>,
{
    let mut mutations = BTreeMap::new();

    for (past_txn_id, ops) in block_data.into_iter() {
        let past_txn_id = past_txn_id.as_str().parse()?;

        let mut parsed = Vec::with_capacity(ops.len());

        for op in ops.into_iter() {
            if op.matches::<(Value,)>() {
                let (key,) = op.opt_cast_into().expect("GET op");
                parsed.push(MutationRecord::Delete(key));
            } else if op.matches::<(Value, State)>() {
                let (key, value) = op.opt_cast_into().expect("PUT op");
                let value = StoreEntry::try_from_state(value)?;
                let value = store.save_state(txn, value).await?;
                parsed.push(MutationRecord::Put(key, value));
            } else {
                return Err(internal!("unable to parse historical mutation {:?}", op,));
            }
        }

        mutations.insert(past_txn_id, parsed);
    }

    Ok(mutations)
}

async fn replay_and_save<State, T>(
    subject: &T,
    txn: &State::Txn,
    txn_id: TxnId,
    ops: &[MutationRecord],
    source: &Store<State::Txn, State::FE>,
    dest: &Store<State::Txn, State::FE>,
    block: &mut ChainBlock,
) -> TCResult<()>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Scalar>,
    State::FE: DenseCacheFile + AsType<ChainBlock> + AsType<BTreeNode> + AsType<TensorNode>,
    T: Route<State> + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
{
    for op in ops {
        match op {
            MutationRecord::Delete(key) => {
                trace!("replay DELETE {} at {}", key, txn_id);

                subject.delete(txn, &[], key.clone()).await?;
                block.append_delete(txn_id, key.clone())
            }
            MutationRecord::Put(key, original_hash) => {
                let state = source.resolve(*txn.id(), original_hash.clone()).await?;

                trace!("replay PUT {}: {:?} at {}", key, state, txn_id);

                subject
                    .put(txn, &[], key.clone(), state.clone().into_state())
                    .await?;

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

async fn load_history<'en, State>(
    history: History<State>,
    op: MutationRecord,
    txn: State::Txn,
) -> TCResult<MutationView<'en>>
where
    State: StateInstance,
    State::FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode>,
{
    match op {
        MutationRecord::Delete(key) => Ok(MutationView::Delete(key)),
        MutationRecord::Put(key, value) => {
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
    Put(Value, StoreEntryView<'en>),
}

impl<'en> en::IntoStream<'en> for MutationView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Delete(key) => (key,).into_stream(encoder),
            Self::Put(key, value) => (key, value).into_stream(encoder),
        }
    }
}

#[inline]
fn get_or_create_block<FE>(cache: &mut DirWriteGuard<FE>, name: String) -> TCResult<FileLock<FE>>
where
    FE: AsType<ChainBlock> + ThreadSafe,
{
    if let Some(file) = cache.get_file(&name) {
        Ok(file.clone())
    } else {
        create_block(cache, name)
    }
}

#[inline]
fn create_block<FE, I: fmt::Display>(
    cache: &mut DirWriteGuard<FE>,
    name: I,
) -> TCResult<FileLock<FE>>
where
    FE: AsType<ChainBlock> + ThreadSafe,
{
    let last_hash = Bytes::from(null_hash().to_vec());

    let block = ChainBlock::new(last_hash.clone());
    let size_hint = block.get_size();

    cache
        .create_file(name.to_string(), block, size_hint)
        .map_err(TCError::from)
}
