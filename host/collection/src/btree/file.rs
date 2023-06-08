use std::marker::PhantomData;
use std::ops::Deref;
use std::string::ToString;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use b_table::b_tree;
use destream::de;
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::{DirLock, DirWriteGuard, FileLoad};
use futures::{future, try_join, Stream, TryFutureExt, TryStreamExt};
use log::{debug, trace};
use safecast::AsType;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Inner, Persist, Restore, VERSIONS};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::ValueCollator;
use tcgeneric::{label, Instance, Label, TCBoxTryStream, ThreadSafe};

use super::schema::BTreeSchema;
use super::slice::BTreeSlice;
use super::stream::Keys;
use super::{BTreeInstance, BTreeType, BTreeWrite, Key, Node, Range};

const CANON: Label = label("canon");
const DELETES: Label = label("deletes");
const INSERTS: Label = label("inserts");

type Version<FE> = b_tree::BTreeLock<BTreeSchema, ValueCollator, FE>;
type VersionReadGuard<FE> = b_tree::BTreeReadGuard<BTreeSchema, ValueCollator, FE>;
type VersionWriteGuard<FE> = b_tree::BTreeWriteGuard<BTreeSchema, ValueCollator, FE>;

type Semaphore = tc_transact::lock::Semaphore<b_tree::Collator<ValueCollator>, Arc<Range>>;

struct Delta<FE> {
    deletes: Version<FE>,
    inserts: Version<FE>,
}

impl<FE> Clone for Delta<FE> {
    fn clone(&self) -> Self {
        Self {
            deletes: self.deletes.clone(),
            inserts: self.inserts.clone(),
        }
    }
}

impl<FE> Delta<FE>
where
    FE: AsType<Node> + ThreadSafe,
{
    fn create(
        schema: BTreeSchema,
        collator: ValueCollator,
        mut dir: DirWriteGuard<FE>,
    ) -> TCResult<Self> {
        let deletes = dir.create_dir(DELETES.to_string())?;
        let inserts = dir.create_dir(INSERTS.to_string())?;

        Ok(Self {
            deletes: Version::create(schema.clone(), collator.clone(), deletes)?,
            inserts: Version::create(schema, collator, inserts)?,
        })
    }

    async fn read(self) -> (VersionReadGuard<FE>, VersionReadGuard<FE>) {
        // acquire these locks in order to avoid the risk of a deadlock
        let inserts = self.inserts.into_read().await;
        let deletes = self.deletes.into_read().await;
        (inserts, deletes)
    }

    async fn write(self) -> (VersionWriteGuard<FE>, VersionWriteGuard<FE>) {
        // acquire these locks in order to avoid the risk of a deadlock
        let inserts = self.inserts.into_write().await;
        let deletes = self.deletes.into_write().await;
        (inserts, deletes)
    }

    async fn merge_into<'a>(
        self,
        mut keys: TCBoxTryStream<'a, Key>,
        collator: b_tree::Collator<ValueCollator>,
        range: Arc<Range>,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Key>> {
        trace!("merge delta");

        let (inserted, deleted) = self.read().await;

        trace!("merge delta acquired a read lock on the version to merge");

        if inserted.is_empty(&range).await? {
            trace!("no inserts to merge");
        } else {
            let inserted = inserted.keys(range.clone(), reverse).map_err(TCError::from);
            keys = Box::pin(collate::try_merge(collator.clone(), keys, inserted));
        }

        if deleted.is_empty(&range).await? {
            trace!("no deletes to merge");
        } else {
            let deleted = deleted.keys(range, reverse).map_err(TCError::from);
            keys = Box::pin(collate::try_diff(collator.clone(), keys, deleted));
        }

        Ok(keys)
    }
}

struct State<FE> {
    commits: OrdHashSet<TxnId>,
    deltas: OrdHashMap<TxnId, Delta<FE>>,
    pending: OrdHashMap<TxnId, Delta<FE>>,
    versions: DirLock<FE>,
    finalized: Option<TxnId>,
}

impl<FE> State<FE>
where
    FE: AsType<Node> + ThreadSafe,
{
    #[inline]
    fn pending_version(
        &mut self,
        txn_id: TxnId,
        schema: &BTreeSchema,
        collator: &ValueCollator,
    ) -> TCResult<Delta<FE>> {
        if let Some(version) = self.pending.get(&txn_id) {
            debug_assert!(!self.commits.contains(&txn_id));
            Ok(version.clone())
        } else if self.commits.contains(&txn_id) {
            Err(conflict!("{} has already been committed", txn_id))
        } else if self.finalized.as_ref() > Some(&txn_id) {
            Err(conflict!("{} has already been finalized", txn_id))
        } else {
            let dir = {
                let mut versions = self.versions.try_write()?;
                versions.create_dir(txn_id.to_string())?
            };

            let version = Delta::create(schema.clone(), collator.clone(), dir.try_write()?)?;
            self.pending.insert(txn_id, version.clone());
            Ok(version)
        }
    }
}

/// A B+Tree which supports concurrent transactional access
pub struct BTreeFile<Txn, FE> {
    dir: DirLock<FE>,
    state: Arc<RwLock<State<FE>>>,
    canon: Version<FE>,
    semaphore: Semaphore,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE> Clone for BTreeFile<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            state: self.state.clone(),
            canon: self.canon.clone(),
            semaphore: self.semaphore.clone(),
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE> BTreeFile<Txn, FE> {
    fn new(dir: DirLock<FE>, canon: Version<FE>, versions: DirLock<FE>) -> Self {
        let semaphore = Semaphore::new(canon.collator().clone());
        let state = State {
            commits: OrdHashSet::new(),
            deltas: OrdHashMap::new(),
            pending: OrdHashMap::new(),
            versions,
            finalized: None,
        };

        Self {
            dir,
            state: Arc::new(RwLock::new(state)),
            canon,
            semaphore,
            phantom: PhantomData,
        }
    }

    pub(super) fn collator(&self) -> &b_tree::Collator<ValueCollator> {
        self.canon.collator()
    }
}

impl<'a, Txn, FE> BTreeFile<Txn, FE>
where
    FE: AsType<Node> + ThreadSafe,
{
    async fn into_keys(
        self,
        txn_id: TxnId,
        range: Arc<Range>,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Key>> {
        trace!("BTreeFile::into_keys {:?}", range);

        let collator = self.collator().clone();

        // read-lock the canonical version BEFORE locking self.state,
        // to avoid a deadlock or conflict with Self::finalize
        let canon = self.canon.into_read().await;

        trace!("BTreeFile::into_keys locked the canonical version for reading");

        let (deltas, pending) = {
            let state = self.state.read().expect("state");
            let deltas = state
                .deltas
                .iter()
                .take_while(|(id, _)| *id <= &txn_id)
                .map(|(_, delta)| delta)
                .cloned()
                .collect::<Vec<_>>();

            let pending = state.pending.get(&txn_id).cloned();

            (deltas, pending)
        };

        trace!("BTreeFile::into_keys got a list of delta versions");

        let keys = canon.keys(range.clone(), reverse).map_err(TCError::from);

        let mut keys: TCBoxTryStream<'static, Key> = Box::pin(keys);

        for delta in deltas {
            trace!("BTreeFile::into_keys merging delta version...");

            keys = delta
                .merge_into(keys, collator.clone(), range.clone(), reverse)
                .await?;
        }

        if let Some(pending) = pending {
            trace!("BTreeFile::into_keys merging pending delta...");

            keys = pending
                .merge_into(keys, collator.clone(), range.clone(), reverse)
                .await?;
        }

        Ok(keys)
    }

    pub(super) async fn into_stream(
        self,
        txn_id: TxnId,
        range: Arc<Range>,
        reverse: bool,
    ) -> TCResult<Keys<'a>> {
        debug!("BTreeFile::into_stream");

        let permit = self.semaphore.read(txn_id, range.clone()).await?;

        trace!("got permit for {:?}", range);

        let keys = self
            .into_keys(txn_id, permit.deref().clone(), reverse)
            .await?;

        Ok(Keys::new(permit, keys))
    }
}

impl<Txn, FE> Instance for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: Send + Sync,
{
    type Class = BTreeType;

    fn class(&self) -> Self::Class {
        BTreeType::File
    }
}

#[async_trait]
impl<Txn, FE> BTreeInstance for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Slice = BTreeSlice<Txn, FE>;

    fn schema(&self) -> &BTreeSchema {
        self.canon.schema()
    }

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        debug!("BTreeFile::count");

        let keys = self.clone().keys(txn_id).await?;
        keys.try_fold(0u64, |count, _key| future::ready(Ok(count + 1)))
            .await
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        debug!("BTreeFile::is_empty");

        let mut keys = self.clone().keys(txn_id).await?;
        keys.try_next().map_ok(|key| key.is_none()).await
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<Keys<'a>>
    where
        Self: 'a,
    {
        debug!("BTreeFile::keys");

        self.into_stream(txn_id, Arc::new(Range::default()), false)
            .await
    }

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice> {
        let range = self.schema().validate_range(range)?;
        Ok(BTreeSlice::new(self, range, reverse))
    }
}

#[async_trait]
impl<Txn, FE> BTreeWrite for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()> {
        debug!("BTreeFile::delete {:?}", range);

        let range = Arc::new(self.schema().validate_range(range)?);
        let _permit = self.semaphore.write(txn_id, range.clone()).await?;

        let (deltas, pending) = {
            let mut state = self.state.write().expect("state");

            let deltas = state
                .deltas
                .iter()
                .take_while(|(id, _)| *id < &txn_id)
                .map(|(_, delta)| delta)
                .cloned()
                .collect::<Vec<_>>();

            let pending = state.pending_version(txn_id, self.schema(), self.collator().inner())?;
            (deltas, pending)
        };

        let canon = self.canon.read().await;
        let mut keys: TCBoxTryStream<Key> =
            Box::pin(canon.keys(range.clone(), false).map_err(TCError::from));

        for delta in deltas {
            let collator = self.collator().clone();

            keys = delta
                .merge_into(keys, collator, range.clone(), false)
                .await?;
        }

        if range.is_default() {
            let (mut inserts, mut deletes) = pending.write().await;

            inserts.truncate().await?;

            while let Some(key) = keys.try_next().await? {
                deletes.insert(key).await?;
            }
        } else {
            let inserts = pending.inserts.read().await;
            let mut deletes = pending.deletes.write().await;

            let collator = self.collator().clone();
            let inserted = inserts.keys(range, false).map_err(TCError::from);
            keys = Box::pin(collate::try_merge(collator, keys, inserted));

            while let Some(key) = keys.try_next().await? {
                deletes.insert(key).await?;
            }
        }

        Ok(())
    }

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        let key = b_tree::Schema::validate(self.schema(), key)?;
        debug!("BTreeFile::insert {:?} at {}", key, txn_id);

        let _permit = self
            .semaphore
            .write(txn_id, Arc::new(Range::from_prefix(key.clone())))
            .await?;

        trace!("BTreeFile::insert got permit");

        let delta = {
            let mut state = self.state.write().expect("state");
            state.pending_version(txn_id, self.schema(), self.collator().inner())?
        };

        trace!("BTreeFile::insert got pending delta");

        let (mut inserts, mut deletes) = delta.write().await;

        trace!("BTreeFile::insert locked pending delta for writing");

        deletes.delete(&key).await?;
        inserts.insert(key).await?;

        trace!("BTreeFile::insert is complete");

        Ok(())
    }

    async fn try_insert_from<S>(&self, txn_id: TxnId, mut keys: S) -> TCResult<()>
    where
        S: Stream<Item = TCResult<Key>> + Send + Unpin,
    {
        let _permit = self
            .semaphore
            .write(txn_id, Arc::new(Range::default()))
            .await?;

        let delta = {
            let mut state = self.state.write().expect("state");
            state.pending_version(txn_id, self.schema(), self.collator().inner())?
        };

        let (mut inserts, mut deletes) = delta.write().await;

        while let Some(key) = keys.try_next().await? {
            deletes.delete(&key).await?;
            inserts.insert(key).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl<Txn, FE> Transact for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("BTreeFile::commit {}", txn_id);

        let mut state = self.state.write().expect("state");

        if state.finalized.as_ref() > Some(&txn_id) {
            panic!("cannot commit finalized version {}", txn_id);
        } else if !state.commits.insert(txn_id) {
            log::warn!("duplicate commit at {}", txn_id);
        } else if let Some(pending) = state.pending.remove(&txn_id) {
            state.deltas.insert(txn_id, pending);
        }

        self.semaphore.finalize(&txn_id, false);
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("BTreeFile::rollback {}", txn_id);

        let mut state = self.state.write().expect("state");

        if state.finalized.as_ref() > Some(txn_id) {
            panic!("tried to roll back finalized version {}", txn_id);
        } else if state.commits.contains(txn_id) {
            panic!("tried to roll back committed version {}", txn_id);
        }

        state.pending.remove(txn_id);

        self.semaphore.finalize(txn_id, false);
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("BTreeFile::finalize {}", txn_id);

        let mut canon = self.canon.write().await;

        let deltas = {
            let mut state = self.state.write().expect("state");

            if state.finalized.as_ref() > Some(txn_id) {
                return;
            }

            let mut deltas = Vec::with_capacity(state.deltas.len());

            while let Some(version_id) = state.pending.keys().next().copied() {
                if &version_id <= txn_id {
                    state.pending.pop_first();
                } else {
                    break;
                }
            }

            while let Some(version_id) = state.commits.first().map(|id| **id) {
                if &version_id <= txn_id {
                    state.commits.pop_first();
                } else {
                    break;
                }
            }

            while let Some(version_id) = state.deltas.keys().next().copied() {
                if &version_id <= txn_id {
                    let version = state.deltas.pop_first().expect("version");
                    deltas.push(version);
                } else {
                    break;
                }
            }

            state.finalized = Some(*txn_id);

            deltas
        };

        for delta in deltas {
            let (inserted, deleted) = delta.read().await;
            canon.merge(inserted).await.expect("commit inserts");
            canon.delete_all(deleted).await.expect("commit deletes");
        }

        self.semaphore.finalize(txn_id, true);
    }
}

#[async_trait]
impl<Txn, FE> Persist<FE> for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
    Node: FileLoad,
{
    type Txn = Txn;
    type Schema = BTreeSchema;

    async fn create(_txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self> {
        debug!("BTreeFile::create");

        let dir = store.into_inner();
        let collator = ValueCollator::default();

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let versions = dir.create_dir(VERSIONS.to_string())?;
            let canon = dir.create_dir(CANON.to_string())?;
            let canon = Version::create(schema, collator, canon)?;
            (canon, versions)
        };

        Ok(Self::new(dir, canon, versions))
    }

    async fn load(_txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self> {
        debug!("BTreeFile::load");

        let dir = store.into_inner();
        let collator = ValueCollator::default();

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let versions = dir.get_or_create_dir(VERSIONS.to_string())?;
            let canon = dir.get_or_create_dir(CANON.to_string())?;
            let canon = Version::load(schema, collator, canon)?;
            (canon, versions)
        };

        Ok(Self::new(dir, canon, versions))
    }

    fn dir(&self) -> Inner<FE> {
        self.dir.clone()
    }
}

#[async_trait]
impl<Txn, FE, I> CopyFrom<FE, I> for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
    I: BTreeInstance + 'static,
    Node: freqfs::FileLoad,
{
    async fn copy_from(
        txn: &<Self as Persist<FE>>::Txn,
        store: Dir<FE>,
        instance: I,
    ) -> TCResult<Self> {
        debug!("BTreeFile::copy_from");

        let txn_id = *txn.id();
        let dir = store.into_inner();
        let schema = instance.schema().clone();
        let collator = ValueCollator::default();

        let mut keys = instance.keys(txn_id).await?;

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let canon = dir.create_dir(CANON.to_string())?;
            let versions = dir.create_dir(VERSIONS.to_string())?;
            (canon, versions)
        };

        let version = {
            let mut dir = versions.write().await;
            dir.create_dir(txn_id.to_string())?
        };

        let (deletes, inserts) = {
            let mut version = version.write().await;
            let deletes = version.create_dir(DELETES.to_string())?;
            let inserts = version.create_dir(INSERTS.to_string())?;
            (deletes, inserts)
        };

        let inserts = Version::create(schema.clone(), collator.clone(), inserts)?;

        {
            let mut inserts = inserts.write().await;
            while let Some(key) = keys.try_next().await? {
                inserts.insert(key).await?;
            }
        }

        let deletes = Version::create(schema.clone(), collator.clone(), deletes)?;

        let delta = Delta { deletes, inserts };

        let canon = Version::create(schema, collator, canon)?;

        let semaphore = Semaphore::with_reservation(
            txn_id,
            canon.collator().clone(),
            Arc::new(Range::default()),
        );

        Ok(Self {
            dir,
            canon,
            state: Arc::new(RwLock::new(State {
                versions,
                deltas: OrdHashMap::new(),
                commits: OrdHashSet::new(),
                pending: std::iter::once((txn_id, delta)).collect(),
                finalized: None,
            })),
            semaphore,
            phantom: PhantomData,
        })
    }
}

#[async_trait]
impl<Txn, FE> Restore<FE> for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
    Node: freqfs::FileLoad,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        debug!("BTreeFile::restore");

        let _permit = self
            .semaphore
            .write(txn_id, Arc::new(Range::default()))
            .await?;

        let collator = self.canon.collator().inner();

        let schema = if self.schema() == backup.schema() {
            self.schema()
        } else {
            return Err(bad_request!(
                "cannot restore a BTree with schema {:?} from one with schema {:?}",
                self.schema(),
                backup.schema()
            ));
        };

        let canon = self.canon.read().await;

        let (deltas, pending) = {
            let mut state = self.state.write().expect("state");

            let deltas = state
                .deltas
                .iter()
                .take_while(|(id, _)| *id < &txn_id)
                .map(|(_, delta)| delta)
                .cloned()
                .collect::<Vec<_>>();

            let pending = state.pending_version(txn_id, schema, collator)?;

            (deltas, pending)
        };

        let (mut inserts, mut deletes) = pending.write().await;

        try_join!(inserts.truncate(), deletes.truncate())?;

        deletes.merge(canon).await?;

        for delta in deltas {
            let (inserted, deleted) = delta.read().await;
            deletes.merge(inserted).await?;
            deletes.delete_all(deleted).await?;
        }

        let mut to_insert = backup.clone().keys(txn_id).await?;
        while let Some(key) = to_insert.try_next().await? {
            deletes.delete(&key).await?;
            inserts.insert(key).await?;
        }

        Ok(())
    }
}

struct BTreeVisitor<Txn, FE> {
    txn: Txn,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> BTreeVisitor<Txn, FE> {
    fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<Txn, FE> de::Visitor for BTreeVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Value = BTreeFile<Txn, FE>;

    fn expecting() -> &'static str {
        "a BTree"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        debug!("BTreeVisitor::visit_seq");

        let txn_id = *self.txn.id();
        let collator = ValueCollator::default();

        trace!("decode schema");
        let schema = seq.expect_next::<BTreeSchema>(()).await?;
        trace!("decoded schema");

        let (canon, versions) = {
            trace!("lock txn context dir to create canon and versions directories...");

            let mut dir = self.txn.context().write().await;

            let canon = dir
                .create_dir(CANON.to_string())
                .map_err(de::Error::custom)?;

            let versions = dir
                .create_dir(VERSIONS.to_string())
                .map_err(de::Error::custom)?;

            (canon, versions)
        };

        let version = {
            trace!("lock versions dir to create version {} dir...", txn_id);

            let mut dir = versions.write().await;
            dir.create_dir(txn_id.to_string())
                .map_err(de::Error::custom)?
        };

        let (deletes, inserts) = {
            trace!(
                "lock version {} dir to create deletes & inserts log dirs...",
                txn_id
            );

            let mut dir = version.write().await;

            let deletes = dir
                .create_dir(DELETES.to_string())
                .map_err(de::Error::custom)?;

            let inserts = dir
                .create_dir(INSERTS.to_string())
                .map_err(de::Error::custom)?;

            (deletes, inserts)
        };

        debug!("decode inserts log as a b_tree::BTreeLock...");
        let cxt = (schema.clone(), collator.clone(), inserts.clone());
        let inserts = if let Some(inserts) = seq.next_element(cxt).await? {
            inserts
        } else {
            trace!("there is no inserts log to be decoded");
            Version::create(schema.clone(), collator.clone(), inserts).map_err(de::Error::custom)?
        };

        trace!("create deletes log");
        let deletes = Version::create(schema.clone(), collator.clone(), deletes)
            .map_err(de::Error::custom)?;

        let version = Delta { inserts, deletes };

        trace!("create canonical b_tree::BTreeLock");
        let canon = Version::create(schema, collator, canon).map_err(de::Error::custom)?;

        let collator = canon.collator().clone();
        let semaphore = Semaphore::with_reservation(txn_id, collator, Arc::new(Range::default()));

        trace!("decoded BTreeFile");

        Ok(BTreeFile {
            dir: self.txn.context().clone(),
            state: Arc::new(RwLock::new(State {
                commits: OrdHashSet::with_capacity(0),
                deltas: OrdHashMap::with_capacity(0),
                pending: std::iter::once((txn_id, version)).collect(),
                versions,
                finalized: None,
            })),
            canon,
            semaphore,
            phantom: PhantomData,
        })
    }
}

#[async_trait]
impl<Txn, FE> de::FromStream for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(BTreeVisitor::new(txn)).await
    }
}
