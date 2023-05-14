use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use b_table::TableLock;
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::{DirLock, DirWriteGuard};
use safecast::AsType;
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::{Dir, Inner, Persist, VERSIONS};
use tc_transact::{Transaction, TxnId};
use tc_value::NumberCollator;
use tcgeneric::{label, Instance, Label, ThreadSafe};

use super::{Range, TensorType};

pub use fensor::sparse::{IndexSchema, Node, Schema};

const CANON: Label = label("canon");
const FILLED: Label = label("filled");
const ZEROS: Label = label("zeros");

type Version<FE> = TableLock<Schema, IndexSchema, NumberCollator, FE>;
type VersionReadGuard<FE> = b_table::TableReadGuard<Schema, IndexSchema, NumberCollator, FE>;
type VersionWriteGuard<FE> = b_table::TableWriteGuard<Schema, IndexSchema, NumberCollator, FE>;

type Semaphore = tc_transact::lock::Semaphore<NumberCollator, Range>;

#[derive(Clone)]
struct Delta<FE> {
    zeros: Version<FE>,
    filled: Version<FE>,
}

// TODO: should this code be consolidated with b_tree::Delta?
impl<FE> Delta<FE>
where
    FE: AsType<Node> + ThreadSafe,
{
    fn create(
        schema: Schema,
        collator: NumberCollator,
        mut dir: DirWriteGuard<FE>,
    ) -> TCResult<Self> {
        let zeros = dir.create_dir(ZEROS.to_string())?;
        let filled = dir.create_dir(FILLED.to_string())?;

        Ok(Self {
            zeros: Version::create(schema.clone(), collator.clone(), zeros)?,
            filled: Version::create(schema, collator, filled)?,
        })
    }

    async fn read(self) -> (VersionReadGuard<FE>, VersionReadGuard<FE>) {
        // acquire these locks in order to avoid the risk of a deadlock
        let filled = self.filled.into_read().await;
        let zeros = self.zeros.into_read().await;
        (filled, zeros)
    }

    async fn write(self) -> (VersionWriteGuard<FE>, VersionWriteGuard<FE>) {
        // acquire these locks in order to avoid the risk of a deadlock
        let filled = self.filled.into_write().await;
        let zeros = self.zeros.into_write().await;
        (filled, zeros)
    }
}

struct State<FE> {
    commits: OrdHashSet<TxnId>,
    deltas: OrdHashMap<TxnId, Delta<FE>>,
    pending: OrdHashMap<TxnId, Delta<FE>>,
    versions: DirLock<FE>,
    finalized: Option<TxnId>,
}

// TODO: should this be merged/consolidated with table::file::State?
impl<FE> State<FE>
where
    FE: AsType<Node> + ThreadSafe,
{
    #[inline]
    fn pending_version(
        &mut self,
        txn_id: TxnId,
        schema: &Schema,
        collator: &NumberCollator,
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

/// A tensor to hold sparse data, based on [`b_table::Table`]
pub struct SparseTensor<Txn, FE> {
    dir: DirLock<FE>,
    canon: Version<FE>,
    state: Arc<RwLock<State<FE>>>,
    semaphore: Semaphore,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE> Clone for SparseTensor<Txn, FE> {
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

impl<Txn, FE> SparseTensor<Txn, FE>
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

impl<Txn, FE> Instance for SparseTensor<Txn, FE>
where
    Self: Send + Sync,
{
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Sparse
    }
}

#[async_trait]
impl<Txn, FE> Persist<FE> for SparseTensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        let dir = store.into_inner();
        let collator = NumberCollator::default();

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
        let collator = NumberCollator::default();

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
