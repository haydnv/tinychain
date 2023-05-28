use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use ds_ext::{OrdHashMap, OrdHashSet};
use fensor::{
    CDatatype, Node, Shape, SparseAccess, SparseCow, SparseTable, SparseWrite, SparseWriteGuard,
    TensorInstance,
};
use freqfs::{DirLock, FileLoad};
use log::debug;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::fs::{Dir, Inner, Persist, VERSIONS};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{DType, Number, NumberCollator, NumberInstance, NumberType};
use tcgeneric::{label, Instance, Label, ThreadSafe};

use crate::tensor::{Range, TensorType};

use super::Schema;

const CANON: Label = label("canon");
const FILLED: Label = label("filled");
const ZEROS: Label = label("zeros");

type Semaphore = tc_transact::lock::Semaphore<NumberCollator, Range>;
type Version<FE, T> = SparseCow<FE, T, SparseAccess<FE, T>>;

struct State<FE, T> {
    commits: OrdHashSet<TxnId>,
    deltas: OrdHashMap<TxnId, Version<FE, T>>,
    pending: OrdHashMap<TxnId, Version<FE, T>>,
    versions: DirLock<FE>,
    finalized: Option<TxnId>,
}

impl<FE, T> State<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType,
{
    #[inline]
    fn pending_version(
        &mut self,
        txn_id: TxnId,
        canon: SparseAccess<FE, T>,
    ) -> TCResult<Version<FE, T>> {
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

            let mut dir = dir.try_write()?;
            let filled = dir.create_dir(FILLED.to_string())?;
            let zeros = dir.create_dir(ZEROS.to_string())?;
            let filled = SparseTable::create(filled, canon.shape().clone())?;
            let zeros = SparseTable::create(zeros, canon.shape().clone())?;

            let version = Version::create(canon, filled, zeros);
            self.pending.insert(txn_id, version.clone());
            Ok(version)
        }
    }
}

/// A tensor to hold sparse data, based on [`b_table::Table`]
pub struct SparseTensorTable<Txn, FE, T> {
    dir: DirLock<FE>,
    canon: SparseTable<FE, T>,
    state: Arc<RwLock<State<FE, T>>>,
    semaphore: Semaphore,
    phantom: PhantomData<(Txn, T)>,
}

impl<Txn, FE, T> Clone for SparseTensorTable<Txn, FE, T> {
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

impl<Txn, FE, T> SparseTensorTable<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    fn new(dir: DirLock<FE>, canon: SparseTable<FE, T>, versions: DirLock<FE>) -> Self {
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

impl<Txn, FE, T> Instance for SparseTensorTable<Txn, FE, T>
where
    Self: Send + Sync,
{
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Sparse
    }
}

impl<Txn, FE, T> TensorInstance for SparseTensorTable<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.canon.schema().shape()
    }
}

#[async_trait]
impl<Txn, FE, T> Persist<FE> for SparseTensorTable<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        let dir = store.into_inner();

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let versions = dir.create_dir(VERSIONS.to_string())?;
            let canon = dir.create_dir(CANON.to_string())?;
            let canon = SparseTable::create(canon, schema.shape().clone())?;
            (canon, versions)
        };

        Ok(Self::new(dir, canon, versions))
    }

    async fn load(_txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        let dir = store.into_inner();

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let versions = dir.get_or_create_dir(VERSIONS.to_string())?;
            let canon = dir.get_or_create_dir(CANON.to_string())?;
            let canon = SparseTable::load(canon, schema.shape().clone())?;
            (canon, versions)
        };

        Ok(Self::new(dir, canon, versions))
    }

    fn dir(&self) -> Inner<FE> {
        self.dir.clone()
    }
}

#[async_trait]
impl<Txn, FE, T> Transact for SparseTensorTable<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + FileLoad,
    T: CDatatype + DType + NumberInstance,
    Number: CastInto<T>,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("SparseTensor::commit {}", txn_id);

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
        debug!("SparseTensor::rollback {}", txn_id);

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
        debug!("SparseTensor::finalize {}", txn_id);

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
            canon.merge(delta).await.expect("write dense tensor delta");
        }

        self.semaphore.finalize(txn_id, true);
    }
}
