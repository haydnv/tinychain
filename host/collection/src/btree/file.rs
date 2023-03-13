use std::marker::PhantomData;
use std::ops::Deref;
use std::string::ToString;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use b_table::b_tree;
use destream::de;
use ds_ext::link::{label, Label};
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::{DirLock, DirWriteGuard, FileLoad};
use futures::{future, join, try_join, TryFutureExt, TryStreamExt};
use safecast::AsType;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Inner, Persist, Restore};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::ValueCollator;
use tcgeneric::{Instance, TCBoxTryStream, ThreadSafe};

use super::schema::Schema;
use super::slice::BTreeSlice;
use super::stream::Keys;
use super::{BTreeInstance, BTreeType, Key, Node, Range};

const CANON: Label = label("canon");
const DELETES: Label = label("deletes");
const INSERTS: Label = label("inserts");
const VERSIONS: Label = label("versions");

type Version<FE> = b_tree::BTreeLock<Schema, ValueCollator, FE>;
type VersionReadGuard<FE> = b_tree::BTreeReadGuard<Schema, ValueCollator, FE>;
type VersionWriteGuard<FE> = b_tree::BTreeWriteGuard<Schema, ValueCollator, FE>;

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
        schema: Schema,
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
        join!(self.inserts.into_read(), self.deletes.into_read())
    }

    async fn write(self) -> (VersionWriteGuard<FE>, VersionWriteGuard<FE>) {
        join!(self.inserts.into_write(), self.deletes.into_write())
    }

    async fn merge_into<'a>(
        self,
        mut keys: TCBoxTryStream<'a, Key>,
        collator: b_tree::Collator<ValueCollator>,
        range: Arc<Range>,
        reverse: bool,
    ) -> TCBoxTryStream<'a, Key> {
        let (deleted, inserted) = join!(self.deletes.into_read(), self.inserts.into_read());

        keys = Box::pin(collate::try_merge(
            collator.clone(),
            keys,
            inserted.keys(range.clone(), reverse).map_err(TCError::from),
        ));

        keys = Box::pin(collate::try_diff(
            collator.clone(),
            keys,
            deleted.keys(range, reverse).map_err(TCError::from),
        ));

        keys
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
        schema: &Schema,
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
            let mut versions = self.versions.try_write()?;
            let dir = versions.create_dir(txn_id.to_string())?;
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
    ) -> TCBoxTryStream<'a, Key> {
        let collator = self.collator().clone();

        // read-lock the canonical version BEFORE locking self.state,
        // to avoid a deadlock or conflict with Self::finalize
        let canon = self.canon.into_read().await;

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

        let keys = canon.keys(range.clone(), reverse).map_err(TCError::from);

        let mut keys: TCBoxTryStream<'static, Key> = Box::pin(keys);

        for delta in deltas {
            keys = delta
                .merge_into(keys, collator.clone(), range.clone(), reverse)
                .await;
        }

        if let Some(pending) = pending {
            keys = pending
                .merge_into(keys, collator.clone(), range.clone(), reverse)
                .await;
        }

        keys
    }

    pub(super) async fn into_stream(
        self,
        txn_id: TxnId,
        range: Arc<Range>,
        reverse: bool,
    ) -> TCResult<Keys<'a>> {
        let permit = self.semaphore.read(txn_id, range).await?;
        let keys = self
            .into_keys(txn_id, permit.deref().clone(), reverse)
            .await;

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

    fn schema(&self) -> &Schema {
        self.canon.schema()
    }

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let keys = self.clone().keys(txn_id).await?;
        keys.try_fold(0u64, |count, _key| future::ready(Ok(count + 1)))
            .await
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        let mut keys = self.clone().keys(txn_id).await?;
        keys.try_next().map_ok(|key| key.is_none()).await
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<Keys<'a>>
    where
        Self: 'a,
    {
        self.into_stream(txn_id, Arc::new(Range::default()), false)
            .await
    }

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice> {
        Ok(BTreeSlice::new(self, range, reverse))
    }
}

impl<Txn, FE> BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    /// Delete the given [`Range`] from this [`BTreeFile`] at `txn_id`.
    pub async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()> {
        let range = Arc::new(self.schema().validate_range(range)?);
        let _permit = self.semaphore.write(txn_id, range.clone()).await?;

        let delta = {
            let mut state = self.state.write().expect("state");
            state.pending_version(txn_id, self.schema(), self.collator().inner())?
        };

        let (mut inserts, mut deletes) = delta.write().await;

        let deleted = BTreeSlice::new(self.clone(), range, false);
        let mut deleted = deleted.keys(txn_id).await?;

        // there's not much point in trying to parallelize this
        while let Some(key) = deleted.try_next().await? {
            try_join!(inserts.delete(key.clone().into()), deletes.insert(key))?;
        }

        Ok(())
    }

    /// Insert the given `key` into this [`BTreeFile`] at `txn_id`.
    pub async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        let key = b_tree::Schema::validate(self.schema(), key)?;

        let _permit = self
            .semaphore
            .write(txn_id, Arc::new(Range::from_prefix(key.clone())))
            .await?;

        let delta = {
            let mut state = self.state.write().expect("state");
            state.pending_version(txn_id, self.schema(), self.collator().inner())?
        };

        let (mut inserts, mut deletes) = delta.write().await;

        try_join!(inserts.insert(key.clone()), deletes.delete(key))?;

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
        let mut state = self.state.write().expect("state");

        if state.finalized.as_ref() > Some(&txn_id) {
            panic!("tried to commit finalized version {}", txn_id);
        } else if state.commits.contains(&txn_id) {
            return log::warn!("duplicate commit at {}", txn_id);
        } else if let Some(pending) = state.pending.remove(&txn_id) {
            state.deltas.insert(txn_id, pending);
        }

        self.semaphore.finalize(&txn_id, false);
        state.commits.insert(txn_id);
    }

    async fn rollback(&self, txn_id: &TxnId) {
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
            let (deleted, inserted) = delta.read().await;
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
    FE: AsType<Node> + ThreadSafe,
    Node: FileLoad,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self> {
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
    FE: AsType<Node> + ThreadSafe,
    I: BTreeInstance + 'static,
    Node: freqfs::FileLoad,
{
    async fn copy_from(
        txn: &<Self as Persist<FE>>::Txn,
        store: Dir<FE>,
        instance: I,
    ) -> TCResult<Self> {
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
    FE: AsType<Node> + ThreadSafe,
    Node: freqfs::FileLoad,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        let _permit = self
            .semaphore
            .write(txn_id, Arc::new(Range::default()))
            .await?;

        let collator = self.canon.collator().inner();

        let schema = if self.schema() == backup.schema() {
            self.schema()
        } else {
            return Err(bad_request!(
                "cannot restore a Table with schema {:?} from one with schema {:?}",
                self.schema(),
                backup.schema()
            ));
        };

        let delta = {
            let mut state = self.state.write().expect("state");
            state.pending_version(txn_id, schema, collator)?
        };

        let (mut deletes, mut inserts) = delta.write().await;

        let mut to_delete = self
            .clone()
            .into_keys(txn_id, Arc::new(Range::default()), false)
            .await;

        while let Some(key) = to_delete.try_next().await? {
            try_join!(deletes.insert(key.clone()), inserts.delete(key))?;
        }

        let mut to_insert = backup.clone().keys(txn_id).await?;
        while let Some(key) = to_insert.try_next().await? {
            try_join!(deletes.delete(key.clone()), inserts.insert(key))?;
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
        let txn_id = *self.txn.id();
        let collator = ValueCollator::default();
        let schema = seq.expect_next::<Schema>(()).await?;

        let (canon, versions) = {
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
            let mut dir = versions.write().await;
            dir.create_dir(txn_id.to_string())
                .map_err(de::Error::custom)?
        };

        let (deletes, inserts) = {
            let mut dir = version.write().await;

            let deletes = dir
                .create_dir(DELETES.to_string())
                .map_err(de::Error::custom)?;

            let inserts = dir
                .create_dir(INSERTS.to_string())
                .map_err(de::Error::custom)?;

            (deletes, inserts)
        };

        let inserts = seq
            .expect_next((schema.clone(), collator.clone(), inserts))
            .await?;

        let deletes = Version::create(schema.clone(), collator.clone(), deletes)
            .map_err(de::Error::custom)?;

        let version = Delta { inserts, deletes };

        let canon = Version::create(schema, collator, canon).map_err(de::Error::custom)?;

        let collator = canon.collator().clone();
        let semaphore = Semaphore::with_reservation(txn_id, collator, Arc::new(Range::default()));

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
