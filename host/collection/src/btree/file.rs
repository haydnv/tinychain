use std::marker::PhantomData;
use std::string::ToString;
use std::sync::Arc;

use async_trait::async_trait;
use destream::de;
use ds_ext::link::{label, Label};
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::{DirLock, FileLoad};
use safecast::AsType;
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Inner, Persist, Restore};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::ValueCollator;
use tcgeneric::{Instance, TCBoxTryStream, ThreadSafe};

use super::schema::Schema;
use super::slice::BTreeSlice;
use super::{BTreeInstance, BTreeType, Key, Node, Range};

const CANON: Label = label("canon");
const VERSIONS: Label = label("versions");

type Canon<FE> = b_tree::BTreeLock<Schema, ValueCollator, FE>;
type Semaphore = tc_transact::lock::Semaphore<ValueCollator, Range>;

struct Delta<FE> {
    deletes: Canon<FE>,
    inserts: Canon<FE>,
}

struct State<FE> {
    canon: Canon<FE>,
    commits: OrdHashSet<TxnId>,
    deltas: OrdHashMap<TxnId, Delta<FE>>,
    pending: OrdHashMap<TxnId, Delta<FE>>,
    versions: DirLock<FE>,
}

/// A B+Tree which supports concurrent transactional access
pub struct BTreeFile<Txn, FE> {
    dir: DirLock<FE>,
    schema: Arc<Schema>,
    semaphore: Semaphore,
    state: Arc<RwLock<State<FE>>>,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE> Clone for BTreeFile<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            schema: self.schema.clone(),
            semaphore: self.semaphore.clone(),
            state: self.state.clone(),
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE> BTreeFile<Txn, FE> {
    fn new(
        collator: ValueCollator,
        schema: Schema,
        dir: DirLock<FE>,
        canon: Canon<FE>,
        versions: DirLock<FE>,
    ) -> Self {
        Self {
            dir,
            schema: Arc::new(schema),
            state: Arc::new(RwLock::new(State {
                canon,
                commits: OrdHashSet::new(),
                deltas: OrdHashMap::new(),
                pending: OrdHashMap::new(),
                versions,
            })),
            semaphore: Semaphore::new(Arc::new(collator)),
            phantom: PhantomData,
        }
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
        &self.schema
    }

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        Err(not_implemented!("BTreeFile::count"))
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        Err(not_implemented!("BTreeFile::is_empty"))
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        Err(not_implemented!("BTreeFile::keys"))
    }

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice> {
        Err(not_implemented!("BTreeFile::slice"))
    }
}

impl<Txn, FE> BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + Send + Sync,
{
    async fn delete(&self, range: Range) -> TCResult<()> {
        Err(not_implemented!("BTreeFile::delete"))
    }

    async fn upsert(&self, key: Key) -> TCResult<()> {
        Err(not_implemented!("BTreeFile::upsert"))
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
            let canon = Canon::create(schema.clone(), collator.clone(), canon)?;
            (canon, versions)
        };

        Ok(Self::new(collator, schema, dir, canon, versions))
    }

    async fn load(_txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self> {
        let dir = store.into_inner();
        let collator = ValueCollator::default();

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let versions = dir.get_or_create_dir(VERSIONS.to_string())?;
            let canon = dir.get_or_create_dir(CANON.to_string())?;
            let canon = Canon::load(schema.clone(), collator.clone(), canon.clone())?;
            (canon, versions)
        };

        Ok(Self::new(collator, schema, dir, canon, versions))
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
