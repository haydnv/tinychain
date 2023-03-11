use std::marker::PhantomData;
use std::string::ToString;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
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
use super::{BTreeInstance, BTreeType, Key, Node, Range};

const CANON: Label = label("canon");
const DELETES: Label = label("deletes");
const INSERTS: Label = label("inserts");
const VERSIONS: Label = label("versions");

type Version<FE> = b_tree::BTreeLock<Schema, ValueCollator, FE>;
type VersionReadGuard<FE> = b_tree::BTreeReadGuard<Schema, ValueCollator, FE>;
type VersionWriteGuard<FE> = b_tree::BTreeWriteGuard<Schema, ValueCollator, FE>;
type Semaphore = tc_transact::lock::Semaphore<b_tree::Collator<ValueCollator>, Range>;

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
    FE: AsType<Node> + Send + Sync,
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

    async fn write(&self) -> (VersionWriteGuard<FE>, VersionWriteGuard<FE>) {
        join!(self.inserts.write(), self.deletes.write())
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
    FE: AsType<Node> + Send + Sync,
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
    FE: AsType<Node> + Send + Sync,
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

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        Err(not_implemented!("BTreeFile::keys"))
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
    async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()> {
        let range = self.schema().validate_range(range)?;
        let permit = self.semaphore.write(txn_id, range.clone()).await?;

        let delta = {
            let mut state = self.state.write().expect("state");
            state.pending_version(txn_id, self.schema(), self.collator().inner())?
        };

        let (mut inserts, mut deletes) = delta.write().await;

        let deleted = self.clone().slice(range, false)?;
        let mut deleted = deleted.keys(txn_id).await?;

        // there's not much point in trying to parallelize this
        while let Some(key) = deleted.try_next().await? {
            try_join!(inserts.delete(key.clone().into()), deletes.insert(key))?;
        }

        Ok(())
    }

    async fn upsert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        let key = b_tree::Schema::validate(self.schema(), key)?;

        let permit = self
            .semaphore
            .write(txn_id, Range::from_prefix(key.clone()))
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
    FE: Send + Sync,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        todo!()
    }

    async fn rollback(&self, txn_id: &TxnId) {
        todo!()
    }

    async fn finalize(&self, txn_id: &TxnId) {
        todo!()
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
        Err(not_implemented!("BTreeFile::copy_from"))
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
        Err(not_implemented!("BTreeFile::restore"))
    }
}

#[async_trait]
impl<Txn, FE> de::FromStream for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: Send + Sync,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        todo!()
    }
}