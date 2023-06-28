use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use collate::Collator;
use destream::de;
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::DirLock;
use futures::{join, try_join, TryFutureExt};
use ha_ndarray::{Array, Buffer, CDatatype};
use log::debug;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::fs::{Dir, Persist};
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::{fs, Transact, Transaction, TxnId};
use tc_value::{DType, Number, NumberType};
use tcgeneric::{label, Instance, Label, ThreadSafe};

use crate::tensor::dense::DenseCacheFile;
use crate::tensor::sparse::{Blocks, Elements};
use crate::tensor::{
    Axes, Coord, Range, Semaphore, Shape, TensorInstance, TensorPermitRead, TensorPermitWrite,
    TensorType, IMAG, REAL,
};

use super::access::{SparseAccess, SparseCow, SparseVersion, SparseWriteGuard, SparseWriteLock};
use super::file::SparseFile;
use super::{Node, Schema, SparseInstance};

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

        if let Some(delta) = self.pending.get(&txn_id) {
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
    FE: DenseCacheFile + AsType<Node> + AsType<Buffer<T>>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
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
    FE: DenseCacheFile + AsType<Node> + AsType<Buffer<T>>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
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
    FE: DenseCacheFile + AsType<Node> + AsType<Buffer<T>>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    async fn clear(&mut self, txn_id: TxnId, range: Range) -> TCResult<()> {
        let _write_permit = self.base.write_permit(txn_id, range.clone()).await?;

        let version = {
            let mut state = self.base.state.write().expect("sparse state");
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
            let mut state = self.base.state.write().expect("sparse state");
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

#[async_trait]
impl<Txn, FE, T> fs::Persist<FE> for SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, schema: Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let (dir, canon, versions) = fs_init(store).await?;
        let canon = SparseFile::create(canon, schema.shape().clone())?;
        Ok(Self::new(dir, canon, versions))
    }

    async fn load(_txn_id: TxnId, schema: Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let (dir, canon, versions) = fs_init(store).await?;
        let canon = SparseFile::load(canon, schema.shape().clone())?;
        Ok(Self::new(dir, canon, versions))
    }

    fn dir(&self) -> fs::Inner<FE> {
        self.dir.clone()
    }
}

#[async_trait]
impl<Txn, FE, T, O> fs::CopyFrom<FE, O> for SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
    T: CDatatype + DType,
    O: SparseInstance<DType = T>,
    Number: From<T> + CastInto<T>,
{
    async fn copy_from(
        txn: &<Self as Persist<FE>>::Txn,
        store: Dir<FE>,
        other: O,
    ) -> TCResult<Self> {
        let (dir, canon, versions) = fs_init(store).await?;
        let canon = SparseFile::copy_from(canon, *txn.id(), other).await?;
        Ok(Self::new(dir, canon, versions))
    }
}

#[async_trait]
impl<Txn, FE, T> fs::Restore<FE> for SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + AsType<Buffer<T>> + Clone,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        // always acquire these permits in-order to avoid the risk of a deadlock
        let _write_permit = self.write_permit(txn_id, Range::default()).await?;
        let _read_permit = backup.read_permit(txn_id, Range::default()).await?;

        let version = {
            let mut state = self.state.write().expect("sparse state");
            state.pending_version(txn_id, self.canon.clone().into())?
        };

        let mut guard = version.write().await;
        guard.overwrite(txn_id, backup.clone()).await
    }
}

#[async_trait]
impl<Txn, FE, T> de::FromStream for SparseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
    T: CDatatype + DType + de::FromStream<Context = ()> + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Context = (Txn, Shape);

    async fn from_stream<D: de::Decoder>(
        cxt: (Txn, Shape),
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let (txn, shape) = cxt;

        let (_dir_name, dir) = {
            let mut cxt = txn.context().write().await;
            cxt.create_dir_unique().map_err(de::Error::custom)?
        };

        let (dir, canon, versions) = dir_init(dir).map_err(de::Error::custom).await?;
        let canon = SparseFile::from_stream((canon, shape), decoder).await?;
        Ok(Self::new(dir, canon, versions))
    }
}

pub(super) struct SparseComplexBaseVisitor<Txn, FE, T> {
    re: (DirLock<FE>, SparseFile<FE, T>, DirLock<FE>),
    im: (DirLock<FE>, SparseFile<FE, T>, DirLock<FE>),
    txn: Txn,
}

impl<Txn, FE, T> SparseComplexBaseVisitor<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
    T: CDatatype,
{
    async fn new(txn: Txn, shape: Shape) -> TCResult<Self> {
        let (re, im) = {
            let dir = {
                let mut cxt = txn.context().write().await;
                let (_dir_name, dir) = cxt.create_dir_unique()?;
                dir
            };

            let mut dir = dir.write().await;

            let re = dir.create_dir(REAL.to_string())?;
            let im = dir.create_dir(IMAG.to_string())?;

            (re, im)
        };

        let ((re_dir, re_canon, re_versions), (im_dir, im_canon, im_versions)) =
            try_join!(dir_init(re), dir_init(im))?;

        let re_canon = SparseFile::create(re_canon, shape.clone())?;
        let im_canon = SparseFile::create(im_canon, shape.clone())?;

        Ok(Self {
            re: (re_dir, re_canon, re_versions),
            im: (im_dir, im_canon, im_versions),
            txn,
        })
    }

    pub async fn end(self) -> TCResult<(SparseBase<Txn, FE, T>, SparseBase<Txn, FE, T>)> {
        let re = SparseBase::new(self.re.0, self.re.1, self.re.2);
        let im = SparseBase::new(self.im.0, self.im.1, self.im.2);
        Ok((re, im))
    }
}

#[async_trait]
impl<Txn, FE, T> de::FromStream for SparseComplexBaseVisitor<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
    T: CDatatype + DType + de::FromStream<Context = ()> + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Context = (Txn, Shape);

    async fn from_stream<D: de::Decoder>(
        cxt: (Txn, Shape),
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let (txn, shape) = cxt;
        let visitor = Self::new(txn, shape).map_err(de::Error::custom).await?;
        decoder.decode_seq(visitor).await
    }
}

#[async_trait]
impl<Txn, FE, T> de::Visitor for SparseComplexBaseVisitor<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + de::FromStream<Context = ()> + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Value = Self;

    fn expecting() -> &'static str {
        "a complex sparse tensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let (mut guard_re, mut guard_im) = join!(self.re.1.write(), self.im.1.write());

        let txn_id = *self.txn.id();
        while let Some((coord, (r, i))) = seq.next_element::<(Coord, (T, T))>(()).await? {
            try_join!(
                guard_re.write_value(txn_id, coord.to_vec(), r),
                guard_im.write_value(txn_id, coord, i)
            )
            .map_err(de::Error::custom)?;
        }

        Ok(self)
    }
}

impl<Txn, FE, T: CDatatype> From<SparseBase<Txn, FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(base: SparseBase<Txn, FE, T>) -> Self {
        Self::Base(base)
    }
}

impl<Txn, FE, T> fmt::Debug for SparseBase<Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional sparse tensor")
    }
}

#[inline]
async fn fs_init<FE>(store: fs::Dir<FE>) -> TCResult<(DirLock<FE>, DirLock<FE>, DirLock<FE>)>
where
    FE: ThreadSafe + Clone,
{
    let dir = store.into_inner();
    dir_init(dir).await
}

#[inline]
async fn dir_init<FE>(dir: DirLock<FE>) -> TCResult<(DirLock<FE>, DirLock<FE>, DirLock<FE>)>
where
    FE: ThreadSafe + Clone,
{
    let (canon, versions) = {
        let mut dir = dir.write().await;
        let versions = dir.create_dir(fs::VERSIONS.to_string())?;
        let canon = dir.create_dir(CANON.to_string())?;
        (canon, versions)
    };

    Ok((dir, canon, versions))
}
