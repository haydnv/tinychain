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
        todo!()
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        todo!()
    }
}

#[async_trait]
impl<Txn, FE> TableRead for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn read(&self, txn_id: TxnId, key: Key) -> TCResult<Option<Vec<Value>>> {
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
        todo!()
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
        todo!()
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        todo!()
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
        todo!()
    }

    async fn update(&self, txn_id: TxnId, key: Key, values: Row) -> TCResult<()> {
        todo!()
    }

    async fn upsert(&self, txn_id: TxnId, key: Key, values: Values) -> TCResult<()> {
        todo!()
    }
}

#[async_trait]
impl<Txn, FE> Transact for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
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
        todo!()
    }
}

#[async_trait]
impl<Txn, FE> Restore<FE> for TableFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        todo!()
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
        todo!()
    }
}
