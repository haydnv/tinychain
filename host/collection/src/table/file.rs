use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use b_table::{b_tree, TableLock};
use destream::de;
use ds_ext::{Id, OrdHashMap, OrdHashSet};
use freqfs::{DirLock, DirWriteGuard};
use futures::{future, TryFutureExt, TryStreamExt};
use log::trace;
use safecast::AsType;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Inner, Persist, Restore, VERSIONS};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Value, ValueCollator};
use tcgeneric::{label, Instance, Label, TCBoxTryStream, ThreadSafe};

use crate::btree::{Node, Schema as IndexSchema};

use super::stream::Rows;
use super::view::{Limited, Selection, TableSlice as Slice};
use super::{
    Key, Range, Row, Schema, TableInstance, TableOrder, TableRead, TableSlice, TableStream,
    TableType, TableWrite, Values,
};

const CANON: Label = label("canon");
const DELETES: Label = label("deletes");
const INSERTS: Label = label("inserts");

type Version<FE> = TableLock<Schema, IndexSchema, ValueCollator, FE>;
type VersionReadGuard<FE> = b_table::TableReadGuard<Schema, IndexSchema, ValueCollator, FE>;
type VersionWriteGuard<FE> = b_table::TableWriteGuard<Schema, IndexSchema, ValueCollator, FE>;

type Semaphore = tc_transact::lock::Semaphore<ValueCollator, Range>;

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

// TODO: should this code be consolidated with b_tree::Delta?
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
        range: b_table::Range<Id, Value>,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Key>> {
        trace!("merge delta");

        let rows = self.inserts.rows(range.clone(), reverse).await?;
        let inserted = rows.map_err(TCError::from);
        keys = Box::pin(collate::try_merge(collator.clone(), keys, inserted));

        let rows = self.deletes.rows(range, reverse).await?;
        let deleted = rows.map_err(TCError::from);
        keys = Box::pin(collate::try_diff(collator.clone(), keys, deleted));

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

/// A relational database table which supports a primary key and multiple indices
pub struct TableFile<Txn, FE> {
    dir: DirLock<FE>,
    canon: Version<FE>,
    state: Arc<RwLock<State<FE>>>,
    semaphore: Semaphore,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE> Clone for TableFile<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            canon: self.canon.clone(),
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE> TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    fn new(dir: DirLock<FE>, canon: Version<FE>, versions: DirLock<FE>) -> Self {
        let semaphore = Semaphore::new(Arc::new(canon.collator().inner().clone()));

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
}

impl<Txn, FE> TableFile<Txn, FE> {
    async fn into_rows<'a>(
        self,
        txn_id: TxnId,
        range: Range,
        order: Vec<Id>,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Row>> {
        todo!()
    }
}

impl<Txn, FE> Instance for TableFile<Txn, FE>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        TableType::Table
    }
}

impl<Txn, FE> TableInstance for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    fn schema(&self) -> &Schema {
        self.canon.schema()
    }
}

impl<Txn, FE> TableOrder for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type OrderBy = Slice<Txn, FE>;
    type Reverse = Slice<Txn, FE>;

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        Slice::new(self, Arc::new(Range::default()), Arc::new(columns), reverse)
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        Slice::new(self, Arc::new(Range::default()), Arc::new(vec![]), true)
    }
}

#[async_trait]
impl<Txn, FE> TableRead for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn read(&self, txn_id: TxnId, key: Key) -> TCResult<Option<Vec<Value>>> {
        let key = b_table::Schema::validate_key(self.schema(), key)?;
        let range = self.schema().range_from_key(key.clone())?;
        let _permit = self.semaphore.read(txn_id, range).await?;

        let (deltas, pending) = {
            let state = self.state.read().expect("state");

            let deltas = state
                .deltas
                .iter()
                .take_while(|(id, _)| *id <= &txn_id)
                .map(|(_, delta)| delta)
                .cloned()
                .collect::<Vec<_>>();

            (deltas, state.pending.get(&txn_id).cloned())
        };

        if let Some(pending) = pending {
            let (inserted, deleted) = pending.read().await;

            if let Some(row) = inserted.get(&key).await? {
                return Ok(Some(row));
            } else if deleted.contains(&key).await? {
                return Ok(None);
            }
        }

        for delta in deltas.into_iter().rev() {
            let (inserted, deleted) = delta.read().await;

            if let Some(row) = inserted.get(&key).await? {
                return Ok(Some(row));
            } else if deleted.contains(&key).await? {
                return Ok(None);
            }
        }

        let canon = self.canon.read().await;
        canon.get(&key).map_err(TCError::from).await
    }
}

impl<Txn, FE> TableSlice for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Slice = Slice<Txn, FE>;

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        Slice::new(self, Arc::new(range), Arc::new(vec![]), false)
    }
}

#[async_trait]
impl<Txn, FE> TableStream for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Limit = Limited<Self>;
    type Selection = Selection<Self>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        let mut rows = self.rows(txn_id).await?;
        rows.try_fold(0, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    fn limit(self, limit: u64) -> Self::Limit {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>> {
        todo!()
    }
}

#[async_trait]
impl<Txn, FE> TableWrite for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn delete(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        let key = b_table::Schema::validate_key(self.schema(), key)?;
        let range = self.schema().range_from_key(key.clone())?;
        let _permit = self.semaphore.write(txn_id, range).await?;

        // read-lock the canonical version BEFORE locking self.state,
        // to avoid a deadlock or conflict with Self::finalize
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

            let pending =
                state.pending_version(txn_id, self.schema(), self.canon.collator().inner())?;

            (deltas, pending)
        };

        let mut row = None;
        for delta in deltas {
            let (inserted, deleted) = delta.read().await;

            if deleted.contains(&key).await? {
                return Ok(());
            } else if let Some(insert) = inserted.get(&key).await? {
                row = Some(insert);
                break;
            }
        }

        let mut row = if let Some(row) = row {
            row
        } else if let Some(row) = canon.get(&key).await? {
            row
        } else {
            return Ok(());
        };

        let (mut inserts, mut deletes) = pending.write().await;

        let values = row.drain(key.len()..).collect();
        debug_assert_eq!(key, row[..key.len()]);

        inserts.delete(&key).await?;
        deletes.upsert(key, values).await?;

        Ok(())
    }

    async fn upsert(&self, txn_id: TxnId, key: Key, values: Values) -> TCResult<()> {
        let key = b_table::Schema::validate_key(self.schema(), key)?;
        let values = b_table::Schema::validate_values(self.schema(), values)?;

        let range = self.schema().range_from_key(key.clone())?;
        let _permit = self.semaphore.write(txn_id, range).await?;

        let pending = {
            let mut state = self.state.write().expect("state");
            state.pending_version(txn_id, self.schema(), self.canon.collator().inner())?
        };

        let (mut inserts, mut deletes) = pending.write().await;

        deletes.delete(&key).await?;
        inserts.upsert(key, values).await?;

        Ok(())
    }
}

// TODO: can this logic be consolidated with impl Transact for BTreeFile?
#[async_trait]
impl<Txn, FE> Transact for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        let mut state = self.state.write().expect("state");

        if state.finalized.as_ref() > Some(&txn_id) {
            panic!("cannot commit finalized version {}", txn_id);
        } else if !state.commits.insert(txn_id) {
            log::warn!("duplicate commit at {}", txn_id);
        } else if let Some(delta) = state.pending.remove(&txn_id) {
            state.deltas.insert(txn_id, delta);
        }

        self.semaphore.finalize(&txn_id, false);
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
            let (inserted, deleted) = delta.read().await;
            canon.merge(inserted).await.expect("commit inserts");
            canon.delete_all(deleted).await.expect("commit deletes");
        }

        self.semaphore.finalize(txn_id, true);
    }
}

#[async_trait]
impl<Txn, FE> Persist<FE> for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
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

    async fn load(_txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
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

// TODO: can this be consolidated with impl CopyFrom for BTreeFile?
#[async_trait]
impl<Txn, FE, T> CopyFrom<FE, T> for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    T: TableStream + 'static,
{
    async fn copy_from(
        txn: &<Self as Persist<FE>>::Txn,
        store: Dir<FE>,
        instance: T,
    ) -> TCResult<Self> {
        let txn_id = *txn.id();
        let dir = store.into_inner();
        let schema = instance.schema().clone();
        let collator = ValueCollator::default();

        let mut rows = instance.rows(txn_id).await?;

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
            let key_len = b_table::Schema::key(&schema).len();
            let mut inserts = inserts.write().await;
            while let Some(mut key) = rows.try_next().await? {
                let values = key.drain(key_len..).collect();
                inserts.upsert(key, values).await?;
            }
        }

        let deletes = Version::create(schema.clone(), collator.clone(), deletes)?;

        let delta = Delta { deletes, inserts };

        let canon = Version::create(schema, collator.clone(), canon)?;

        let semaphore = Semaphore::with_reservation(txn_id, collator.into(), Range::default());

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

// TODO: can this be consolidated with impl Restore for BTreeFile?
#[async_trait]
impl<Txn, FE> Restore<FE> for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        let _permit = self.semaphore.write(txn_id, Range::default()).await?;

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

        let key_len = b_table::Schema::key(schema).len();

        let (mut inserts, mut deletes) = delta.write().await;

        let mut to_delete = self
            .clone()
            .into_rows(txn_id, Range::default(), vec![], false)
            .await?;

        while let Some(mut row) = to_delete.try_next().await? {
            let values = row.drain(key_len..).collect();
            let key = row;

            inserts.delete(&key).await?;
            deletes.upsert(key, values).await?;
        }

        let mut to_insert = backup.clone().rows(txn_id).await?;
        while let Some(mut row) = to_insert.try_next().await? {
            let values = row.drain(key_len..).collect();
            let key = row;

            deletes.delete(&key).await?;
            inserts.upsert(key, values).await?;
        }

        Ok(())
    }
}

struct TableVisitor<Txn, FE> {
    txn: Txn,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> TableVisitor<Txn, FE> {
    fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<Txn, FE> de::Visitor for TableVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Value = TableFile<Txn, FE>;

    fn expecting() -> &'static str {
        "a Table"
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

        let cxt = (schema.clone(), collator.clone(), inserts.clone());
        let inserts = if let Some(inserts) = seq.next_element(cxt).await? {
            inserts
        } else {
            Version::create(schema.clone(), collator.clone(), inserts).map_err(de::Error::custom)?
        };

        let deletes = Version::create(schema.clone(), collator.clone(), deletes)
            .map_err(de::Error::custom)?;

        let version = Delta { inserts, deletes };

        let canon = Version::create(schema, collator, canon).map_err(de::Error::custom)?;

        let collator = Arc::new(canon.collator().inner().clone());
        let semaphore = Semaphore::with_reservation(txn_id, collator, Range::default());

        Ok(TableFile {
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
impl<Txn, FE> de::FromStream for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(TableVisitor::new(txn)).await
    }
}

impl<Txn, FE> fmt::Debug for TableFile<Txn, FE>
where
    Self: TableInstance,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a relational database table with schema {:?}",
            self.schema()
        )
    }
}
