use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use destream::{de, en};
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::DirReadGuardOwned;
use safecast::AsType;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Inner, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::ValueCollator;
use tcgeneric::{Instance, TCBoxTryStream, ThreadSafe};

use super::schema::Schema;
use super::slice::BTreeSlice;
use super::{BTreeInstance, BTreeType, Key, Node, Range};

type Canon<FE> = b_tree::BTreeLock<Schema, ValueCollator, DirReadGuardOwned<FE>>;
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
}

/// A B+Tree which supports concurrent transactional access
pub struct BTreeFile<Txn, FE> {
    schema: Arc<Schema>,
    semaphore: Semaphore,
    state: Arc<RwLock<State<FE>>>,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE> Clone for BTreeFile<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            schema: self.schema.clone(),
            semaphore: self.semaphore.clone(),
            state: self.state.clone(),
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

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice> {
        Err(not_implemented!("BTreeFile::slice"))
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
    FE: ThreadSafe,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self> {
        Err(not_implemented!("BTreeFile::create"))
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: Dir<FE>) -> TCResult<Self> {
        Err(not_implemented!("BTreeFile::load"))
    }

    fn dir(&self) -> Inner<FE> {
        todo!()
    }
}

#[async_trait]
impl<Txn, FE, I> CopyFrom<FE, I> for BTreeFile<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: ThreadSafe,
    I: BTreeInstance + 'static,
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
    FE: ThreadSafe,
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
