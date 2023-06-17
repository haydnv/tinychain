use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use collate::Collator;
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::DirLock;
use futures::TryFutureExt;
use ha_ndarray::{Array, CDatatype};
use log::debug;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::fs::{Dir, Inner, Persist, VERSIONS};
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{DType, Number, NumberType};
use tcgeneric::{label, Instance, Label, ThreadSafe};

use crate::tensor::sparse::{Blocks, Elements};
use crate::tensor::{
    Axes, Coord, Range, Semaphore, Shape, TensorInstance, TensorPermitRead, TensorPermitWrite,
    TensorRead, TensorType, TensorWrite, TensorWriteDual,
};

use super::access::{
    SparseAccess, SparseCow, SparseFile, SparseVersion, SparseWriteGuard, SparseWriteLock,
};
use super::{Node, Schema, SparseInstance, SparseSlice, SparseTensor};

const CANON: Label = label("canon");
const FILLED: Label = label("filled");
const ZEROS: Label = label("zeros");

type Version<Txn, FE, T> = SparseCow<FE, T, SparseAccess<Txn, FE, T>>;

struct Delta<FE, T> {
    filled: SparseFile<FE, T>,
    zeros: SparseFile<FE, T>,
}

impl<FE, T> Clone for Delta<FE, T> {
    fn clone(&self) -> Self {
        Delta {
            filled: self.filled.clone(),
            zeros: self.zeros.clone(),
        }
    }
}

struct State<Txn, FE, T> {
    commits: OrdHashSet<TxnId>,
    deltas: OrdHashMap<TxnId, Delta<FE, T>>,
    pending: OrdHashMap<TxnId, Delta<FE, T>>,
    versions: DirLock<FE>,
    finalized: Option<TxnId>,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE, T> State<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType,
{
    #[inline]
    fn latest_version(
        &self,
        txn_id: TxnId,
        canon: SparseAccess<Txn, FE, T>,
    ) -> TCResult<SparseAccess<Txn, FE, T>> {
        if self.finalized > Some(txn_id) {
            return Err(conflict!("sparse tensor is already finalized at {txn_id}"));
        }

        let mut version = canon.clone().into();
        for (_version_id, delta) in self
            .deltas
            .iter()
            .take_while(|(version_id, _delta)| *version_id <= &txn_id)
        {
            version = Version::create(version, delta.filled.clone(), delta.zeros.clone()).into();
        }

        Ok(version)
    }

    #[inline]
    fn pending_version(
        &mut self,
        txn_id: TxnId,
        canon: SparseAccess<Txn, FE, T>,
    ) -> TCResult<Version<Txn, FE, T>> {
        if self.commits.contains(&txn_id) {
            return Err(conflict!("{} has already been committed", txn_id));
        } else if self.finalized > Some(txn_id) {
            return Err(conflict!("sparse tensor is already finalized at {txn_id}"));
        }

        let mut version = canon.clone().into();
        for (_version_id, delta) in self
            .deltas
            .iter()
            .take_while(|(version_id, _delta)| *version_id < &txn_id)
        {
            version = Version::create(version, delta.filled.clone(), delta.zeros.clone()).into();
        }

        if let Some(delta) = self.pending.get(&txn_id) {
            Ok(Version::create(
                version,
                delta.filled.clone(),
                delta.zeros.clone(),
            ))
        } else {
            let dir = {
                let mut versions = self.versions.try_write()?;
                versions.create_dir(txn_id.to_string())?
            };

            let mut dir = dir.try_write()?;
            let filled = dir.create_dir(FILLED.to_string())?;
            let zeros = dir.create_dir(ZEROS.to_string())?;
            let filled = SparseFile::create(filled, canon.shape().clone())?;
            let zeros = SparseFile::create(zeros, canon.shape().clone())?;
            let delta = Delta { filled, zeros };

            self.pending.insert(txn_id, delta.clone());

            Ok(Version::create(version, delta.filled, delta.zeros))
        }
    }
}

/// A tensor to hold sparse data, based on [`b_table::Table`]
pub struct SparseBase<Txn, FE, T> {
    dir: DirLock<FE>,
    canon: SparseVersion<FE, T>,
    state: Arc<RwLock<State<Txn, FE, T>>>,
}

impl<Txn, FE, T> Clone for SparseBase<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            canon: self.canon.clone(),
            state: self.state.clone(),
        }
    }
}

impl<Txn, FE, T> SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType,
{
    fn access(&self, txn_id: TxnId) -> TCResult<SparseAccess<Txn, FE, T>> {
        let state = self.state.read().expect("sparse state");
        state.latest_version(txn_id, self.canon.clone().into())
    }
}

impl<Txn, FE, T> SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    fn new(dir: DirLock<FE>, canon: SparseFile<FE, T>, versions: DirLock<FE>) -> Self {
        let semaphore = Semaphore::new(Arc::new(Collator::default()));

        let state = State {
            commits: OrdHashSet::new(),
            deltas: OrdHashMap::new(),
            pending: OrdHashMap::new(),
            versions,
            finalized: None,
            phantom: PhantomData,
        };

        Self {
            dir,
            state: Arc::new(RwLock::new(state)),
            canon: SparseVersion::new(canon, semaphore),
        }
    }
}

impl<Txn, FE, T> Instance for SparseBase<Txn, FE, T>
where
    Self: Send + Sync,
{
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Sparse
    }
}

impl<Txn, FE, T> TensorInstance for SparseBase<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.canon.shape()
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseBase<Txn, FE, T>
where
    Txn: Send + Sync,
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.canon.read_permit(txn_id, range).await
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitWrite for SparseBase<Txn, FE, T>
where
    Txn: Send + Sync,
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.canon.write_permit(txn_id, range).await
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let version = self.access(txn_id)?;
        version.blocks(txn_id, range, order).await
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let version = self.access(txn_id)?;
        version.elements(txn_id, range, order).await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        let version = self.access(txn_id)?;
        version.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<'a, Txn, FE, T> SparseWriteLock<'a> for SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Guard = SparseBaseWriteGuard<'a, Txn, FE, T>;

    async fn write(&'a self) -> Self::Guard {
        SparseBaseWriteGuard { base: self }
    }
}

pub struct SparseBaseWriteGuard<'a, Txn, FE, T> {
    base: &'a SparseBase<Txn, FE, T>,
}

#[async_trait]
impl<'a, Txn, FE, T> SparseWriteGuard<T> for SparseBaseWriteGuard<'a, Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    async fn clear(&mut self, txn_id: TxnId, range: Range) -> TCResult<()> {
        let _write_permit = self.base.write_permit(txn_id, range.clone()).await?;

        let version = {
            let mut state = self.base.state.write().expect("dense state");
            state.pending_version(txn_id, self.base.canon.clone().into())?
        };

        let mut guard = version.write().await;
        guard.clear(txn_id, range).await
    }

    async fn overwrite<O>(&mut self, txn_id: TxnId, other: O) -> TCResult<()>
    where
        O: SparseInstance<DType = T> + TensorPermitRead,
    {
        // always acquire these permits in-order to avoid the risk of a deadlock
        let _write_permit = self.base.write_permit(txn_id, Range::default()).await?;
        let _read_permit = other.read_permit(txn_id, Range::default()).await?;

        let version = {
            let mut state = self.base.state.write().expect("dense state");
            state.pending_version(txn_id, self.base.canon.clone().into())?
        };

        let mut guard = version.write().await;
        guard.overwrite(txn_id, other).await
    }

    async fn write_value(&mut self, txn_id: TxnId, coord: Coord, value: T) -> TCResult<()> {
        let _permit = self
            .base
            .write_permit(txn_id, coord.to_vec().into())
            .await?;

        let version = {
            let mut state = self.base.state.write().expect("sparse state");
            state.pending_version(txn_id, self.base.canon.clone().into())?
        };

        let mut version = version.write().await;

        version
            .write_value(txn_id, coord, value.cast_into())
            .map_err(TCError::from)
            .await
    }
}

// #[async_trait]
// impl<Txn, FE, T> TensorRead for SparseBase<Txn, FE, T>
// where
//     Txn: Transaction<FE>,
//     FE: AsType<Node> + ThreadSafe,
//     T: CDatatype + DType + fmt::Debug,
//     Number: From<T> + CastInto<T>,
// {
//     async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
//         let _permit = self.read_permit(txn_id, coord.to_vec().into()).await?;
//
//         let version = {
//             let state = self.state.read().expect("sparse state");
//             state.latest_version(txn_id, self.canon.clone().into())?
//         };
//
//         version
//             .read_value(coord)
//             .map_ok(Number::from)
//             .map_err(TCError::from)
//             .await
//     }
// }
//
// #[async_trait]
// impl<Txn, FE, T> TensorWrite for SparseBase<Txn, FE, T>
// where
//     Txn: Transaction<FE>,
//     FE: AsType<Node> + ThreadSafe,
//     T: CDatatype + DType + fmt::Debug,
//     Number: From<T> + CastInto<T>,
// {
//     async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()> {
//         let _permit = self.write_permit(txn_id, range.clone().into()).await?;
//
//         let value = value.cast_into();
//
//         let version = {
//             let mut state = self.state.write().expect("sparse state");
//             state.pending_version(txn_id, self.canon.clone().into())?
//         };
//
//         let mut version = version.write().await;
//
//         for coord in range.affected() {
//             version.write_value(coord, value).await?;
//         }
//
//         Ok(())
//     }
//
//     async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
//         let _permit = self.write_permit(txn_id, coord.to_vec().into()).await?;
//
//         let version = {
//             let mut state = self.state.write().expect("sparse state");
//             state.pending_version(txn_id, self.canon.clone().into())?
//         };
//
//         let mut version = version.write().await;
//
//         version
//             .write_value(coord, value.cast_into())
//             .map_err(TCError::from)
//             .await
//     }
// }
// #[async_trait]
// impl<Txn, FE, A> TensorWriteDual<SparseTensor<FE, A>> for SparseBase<Txn, FE, A::DType>
// where
//     Txn: Transaction<FE>,
//     FE: AsType<Node> + ThreadSafe,
//     A: SparseInstance + TensorPermitRead,
//     A::DType: fmt::Debug,
//     Number: From<A::DType> + CastInto<A::DType>,
// {
//     async fn write(self, txn_id: TxnId, range: Range, value: SparseTensor<FE, A>) -> TCResult<()> {
//         // always acquire these permits in-order to avoid the risk of a deadlock
//         let _write_permit = self.write_permit(txn_id, range.clone()).await?;
//         let _read_permit = value.accessor.read_permit(txn_id, range.clone()).await?;
//
//         let version = {
//             let mut state = self.state.write().expect("dense state");
//             state.pending_version(txn_id, self.canon.clone().into())?
//         };
//
//         if range.is_empty() || range == Range::all(self.canon.shape()) {
//             let mut guard = version.write().await;
//             guard.overwrite(value.accessor).await
//         } else {
//             let slice = SparseSlice::new(version, range)?;
//             let mut guard = slice.write().await;
//             guard.overwrite(value.accessor).await
//         }
//     }
// }

#[async_trait]
impl<Txn, FE, T> Persist<FE> for SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        let dir = store.into_inner();

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let versions = dir.create_dir(VERSIONS.to_string())?;
            let canon = dir.create_dir(CANON.to_string())?;
            let canon = SparseFile::create(canon, schema.shape().clone())?;
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
            let canon = SparseFile::load(canon, schema.shape().clone())?;
            (canon, versions)
        };

        Ok(Self::new(dir, canon, versions))
    }

    fn dir(&self) -> Inner<FE> {
        self.dir.clone()
    }
}

#[async_trait]
impl<Txn, FE, T> Transact for SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
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

        self.canon.commit(&txn_id);
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

        self.canon.rollback(txn_id);
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
            canon
                .merge(*txn_id, delta.filled, delta.zeros)
                .await
                .expect("write dense tensor delta");
        }

        self.canon.finalize(txn_id);
    }
}

impl<Txn, FE, T> fmt::Debug for SparseBase<Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional sparse tensor",)
    }
}
