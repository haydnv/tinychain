use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use collate::Collator;
use destream::de;
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::{DirLock, FileLoad};
use futures::TryFutureExt;
use ha_ndarray::{Buffer, CDatatype};
use log::debug;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::fs::{Dir, Inner, Persist, VERSIONS};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{DType, Number, NumberInstance, NumberType};
use tcgeneric::{label, Instance, Label, ThreadSafe};

use crate::tensor::sparse::Node;
use crate::tensor::{
    Coord, Range, Semaphore, Shape, TensorDualIO, TensorInstance, TensorPermitRead,
    TensorPermitWrite, TensorRead, TensorType, TensorWrite,
};

use super::access::{DenseAccess, DenseCow, DenseFile, DenseSlice, DenseVersion};
use super::{DenseCacheFile, DenseInstance, DenseTensor, DenseWriteGuard, DenseWriteLock};

const CANON: Label = label("canon");

struct State<FE, T> {
    commits: OrdHashSet<TxnId>,
    deltas: OrdHashMap<TxnId, DirLock<FE>>,
    pending: OrdHashMap<TxnId, DirLock<FE>>,
    versions: DirLock<FE>,
    finalized: Option<TxnId>,
    dtype: PhantomData<T>,
}

impl<FE, T> State<FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>> + 'static,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    #[inline]
    fn latest_version(
        &self,
        txn_id: TxnId,
        canon: DenseAccess<FE, T>,
    ) -> TCResult<DenseAccess<FE, T>> {
        if self.finalized > Some(txn_id) {
            return Err(conflict!("dense tensor is already finalized at {txn_id}"));
        }

        let mut version = canon;
        for (_version_id, delta) in self
            .deltas
            .iter()
            .take_while(|(version_id, _delta)| *version_id <= &txn_id)
        {
            version = DenseCow::create(version, delta.clone()).into();
        }

        Ok(version)
    }

    #[inline]
    fn pending_version(
        &mut self,
        txn_id: TxnId,
        canon: DenseAccess<FE, T>,
    ) -> TCResult<DenseCow<FE, DenseAccess<FE, T>>> {
        if self.commits.contains(&txn_id) {
            return Err(conflict!("{txn_id} has already been committed"));
        } else if self.finalized > Some(txn_id) {
            return Err(conflict!("dense tensor is already finalized at {txn_id}"));
        }

        let mut version = canon;
        for (_version_id, delta) in self
            .deltas
            .iter()
            .take_while(|(version_id, _delta)| *version_id < &txn_id)
        {
            version = DenseCow::create(version, delta.clone()).into();
        }

        if let Some(delta) = self.pending.get(&txn_id) {
            Ok(DenseCow::create(version, delta.clone()))
        } else {
            let delta = {
                let mut versions = self.versions.try_write()?;
                versions.create_dir(txn_id.to_string())?
            };

            self.pending.insert(txn_id, delta.clone());
            Ok(DenseCow::create(version, delta.clone()))
        }
    }
}

pub struct DenseBase<Txn, FE, T> {
    dir: DirLock<FE>,
    canon: DenseVersion<FE, T>,
    state: Arc<RwLock<State<FE, T>>>,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE, T> Clone for DenseBase<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            canon: self.canon.clone(),
            state: self.state.clone(),
            phantom: self.phantom,
        }
    }
}

impl<Txn, FE, T> DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Buffer<T>> + ThreadSafe,
    T: CDatatype,
{
    fn new(dir: DirLock<FE>, canon: DenseFile<FE, T>, versions: DirLock<FE>) -> Self {
        let semaphore = Semaphore::new(Arc::new(Collator::default()));

        let state = State {
            commits: OrdHashSet::new(),
            deltas: OrdHashMap::new(),
            pending: OrdHashMap::new(),
            versions,
            finalized: None,
            dtype: PhantomData,
        };

        Self {
            dir,
            state: Arc::new(RwLock::new(state)),
            canon: DenseVersion::new(canon, semaphore),
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE, T> Instance for DenseBase<Txn, FE, T>
where
    Self: Send + Sync,
{
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Sparse
    }
}

impl<Txn, FE, T> TensorInstance for DenseBase<Txn, FE, T>
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
impl<Txn, FE, T> TensorRead for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node> + 'static,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        let _permit = self
            .canon
            .read_permit(txn_id, coord.to_vec().into())
            .await?;

        let version = {
            let state = self.state.read().expect("dense state");
            state.latest_version(txn_id, self.canon.clone().into())?
        };

        version
            .read_value(coord)
            .map_ok(Number::from)
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<Txn, FE, T> TensorWrite for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node> + 'static,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()> {
        let _permit = self
            .canon
            .write_permit(txn_id, range.clone().into())
            .await?;

        let version = {
            let mut state = self.state.write().expect("dense state");
            let canon = state.latest_version(txn_id, self.canon.clone().into())?;
            state.pending_version(txn_id, canon)?
        };

        let slice = DenseSlice::new(version, range)?;
        let slice = slice.write().await;
        slice
            .overwrite_value(value.cast_into())
            .map_err(TCError::from)
            .await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        let _permit = self
            .canon
            .write_permit(txn_id, coord.to_vec().into())
            .await?;

        let version = {
            let mut state = self.state.write().expect("dense state");
            let canon = state.latest_version(txn_id, self.canon.clone().into())?;
            state.pending_version(txn_id, canon)?
        };

        let version = version.write().await;

        version
            .write_value(coord, value.cast_into())
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<Txn, FE, T, A> TensorDualIO<DenseTensor<FE, A>> for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node> + 'static,
    T: CDatatype + DType + fmt::Debug,
    A: DenseInstance<DType = T> + TensorPermitRead,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    async fn write(self, txn_id: TxnId, range: Range, value: DenseTensor<FE, A>) -> TCResult<()> {
        // always acquire these permits in-order to avoid the risk of a deadlock
        let _write_permit = self.canon.write_permit(txn_id, range.clone()).await?;
        let _read_permit = value.accessor.read_permit(txn_id, range.clone()).await?;

        let version = {
            let mut state = self.state.write().expect("dense state");
            state.pending_version(txn_id, self.canon.clone().into())?
        };

        if range.is_empty() || range == Range::all(self.canon.shape()) {
            let guard = version.write().await;
            guard.overwrite(value.accessor).await
        } else {
            let slice = DenseSlice::new(version, range)?;
            let guard = slice.write().await;
            guard.overwrite(value.accessor).await
        }
    }
}

#[async_trait]
impl<Txn, FE, T> Persist<FE> for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + ThreadSafe + Clone,
    T: CDatatype + DType + NumberInstance,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Txn = Txn;
    type Schema = Shape;

    async fn create(_txn_id: TxnId, shape: Shape, store: Dir<FE>) -> TCResult<Self> {
        let dir = store.into_inner();

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let versions = dir.create_dir(VERSIONS.to_string())?;
            let canon = dir.create_dir(CANON.to_string())?;
            (canon, versions)
        };

        let canon = DenseFile::constant(canon, shape, T::zero()).await?;

        Ok(Self::new(dir, canon, versions))
    }

    async fn load(_txn_id: TxnId, shape: Shape, store: Dir<FE>) -> TCResult<Self> {
        let dir = store.into_inner();

        let (canon, versions) = {
            let mut dir = dir.write().await;
            let versions = dir.get_or_create_dir(VERSIONS.to_string())?;
            let canon = dir.get_or_create_dir(CANON.to_string())?;
            (canon, versions)
        };

        let canon = DenseFile::load(canon, shape).await?;

        Ok(Self::new(dir, canon, versions))
    }

    fn dir(&self) -> Inner<FE> {
        self.dir.clone()
    }
}

#[async_trait]
impl<Txn, FE, T> Transact for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Buffer<T>> + FileLoad,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("DenseTensor::commit {}", txn_id);

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
        debug!("DenseTensor::rollback {}", txn_id);

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
        debug!("DenseTensor::finalize {}", txn_id);

        let canon = self.canon.write().await;

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

        self.canon.finalize(txn_id);
    }
}

impl<Txn, FE, T> fmt::Debug for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Buffer<T>>,
    T: CDatatype + DType,
    DenseFile<FE, T>: TensorInstance,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dense tensor file with shape {:?}", self.canon.shape())
    }
}
