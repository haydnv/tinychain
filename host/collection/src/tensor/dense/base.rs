use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use collate::Collator;
use destream::de;
use ds_ext::{OrdHashMap, OrdHashSet};
use freqfs::{DirLock, FileWriteGuardOwned};
use futures::{join, try_join, TryFutureExt};
use ha_ndarray::{Array, ArrayBase, Buffer, CDatatype};
use log::debug;
use rayon::prelude::*;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::{fs, Transact, Transaction, TxnId};
use tc_value::{DType, Number, NumberClass, NumberType};
use tcgeneric::{label, Instance, Label, ThreadSafe};

use crate::tensor::sparse::Node;
use crate::tensor::{
    Coord, Range, Semaphore, Shape, TensorInstance, TensorPermitRead, TensorPermitWrite,
    TensorType, IMAG, REAL,
};

use super::access::{DenseAccess, DenseCow, DenseVersion};
use super::file::DenseFile;
use super::{
    ideal_block_size_for, BlockStream, DenseCacheFile, DenseInstance, DenseWrite, DenseWriteGuard,
    DenseWriteLock,
};

const CANON: Label = label("canon");

struct State<Txn, FE, T> {
    commits: OrdHashSet<TxnId>,
    deltas: OrdHashMap<TxnId, DirLock<FE>>,
    pending: OrdHashMap<TxnId, DirLock<FE>>,
    versions: DirLock<FE>,
    finalized: Option<TxnId>,
    dtype: PhantomData<(Txn, T)>,
}

impl<Txn, FE, T> State<Txn, FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>> + 'static,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    #[inline]
    fn latest_version(
        &self,
        txn_id: TxnId,
        canon: DenseAccess<Txn, FE, T>,
    ) -> TCResult<DenseAccess<Txn, FE, T>> {
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
        canon: DenseAccess<Txn, FE, T>,
    ) -> TCResult<DenseCow<FE, DenseAccess<Txn, FE, T>>> {
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
    state: Arc<RwLock<State<Txn, FE, T>>>,
}

impl<Txn, FE, T> Clone for DenseBase<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            canon: self.canon.clone(),
            state: self.state.clone(),
        }
    }
}

impl<Txn, FE, T> DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>>,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    fn access(&self, txn_id: TxnId) -> TCResult<DenseAccess<Txn, FE, T>> {
        let state = self.state.read().expect("dense state");
        state.latest_version(txn_id, self.canon.clone().into())
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
impl<Txn, FE, T> TensorPermitRead for DenseBase<Txn, FE, T>
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
impl<Txn, FE, T> TensorPermitWrite for DenseBase<Txn, FE, T>
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
impl<Txn, FE, T> DenseInstance for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.canon.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let version = self.access(txn_id)?;
        version.read_block(txn_id, block_id).await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let version = self.access(txn_id)?;
        version.read_blocks(txn_id).await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let version = self.access(txn_id)?;
        version.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<Txn, FE, T> DenseWrite for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    type BlockWrite = ArrayBase<FileWriteGuardOwned<FE, Buffer<T>>>;

    async fn write_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::BlockWrite> {
        let version = {
            let mut state = self.state.write().expect("dense state");
            state.pending_version(txn_id, self.canon.clone().into())?
        };

        version.write_block(txn_id, block_id).await
    }

    async fn write_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::BlockWrite>> {
        let version = {
            let mut state = self.state.write().expect("dense state");
            state.pending_version(txn_id, self.canon.clone().into())?
        };

        version.write_blocks(txn_id).await
    }
}

#[async_trait]
impl<'a, Txn, FE, T> DenseWriteLock<'a> for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    type WriteGuard = DenseBaseWriteGuard<'a, Txn, FE, T>;

    async fn write(&'a self) -> Self::WriteGuard {
        DenseBaseWriteGuard { base: self }
    }
}

pub struct DenseBaseWriteGuard<'a, Txn, FE, T> {
    base: &'a DenseBase<Txn, FE, T>,
}

#[async_trait]
impl<'a, Txn, FE, T> DenseWriteGuard<T> for DenseBaseWriteGuard<'a, Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    async fn overwrite<O>(&self, txn_id: TxnId, other: O) -> TCResult<()>
    where
        O: DenseInstance<DType = T> + TensorPermitRead,
    {
        let version = {
            let mut state = self.base.state.write().expect("dense state");
            state.pending_version(txn_id, self.base.canon.clone().into())?
        };

        let guard = version.write().await;
        guard.overwrite(txn_id, other).await
    }

    async fn overwrite_value(&self, txn_id: TxnId, value: T) -> TCResult<()> {
        let version = {
            let mut state = self.base.state.write().expect("dense state");
            let canon = state.latest_version(txn_id, self.base.canon.clone().into())?;
            state.pending_version(txn_id, canon)?
        };

        let guard = version.write().await;
        guard.overwrite_value(txn_id, value).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: T) -> TCResult<()> {
        let version = {
            let mut state = self.base.state.write().expect("dense state");
            let canon = state.latest_version(txn_id, self.base.canon.clone().into())?;
            state.pending_version(txn_id, canon)?
        };

        let guard = version.write().await;
        guard.write_value(txn_id, coord, value).await
    }
}

#[async_trait]
impl<Txn, FE, T> Transact for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Buffer<T>> + ThreadSafe,
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

#[async_trait]
impl<Txn, FE, T> fs::Persist<FE> for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + Clone,
    T: CDatatype + DType + de::FromStream<Context = ()>,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Txn = Txn;
    type Schema = Shape;

    async fn create(_txn_id: TxnId, shape: Shape, store: fs::Dir<FE>) -> TCResult<Self> {
        let (dir, canon, versions) = fs_init(store).await?;
        let canon = DenseFile::constant(canon, shape, T::zero()).await?;
        Ok(Self::new(dir, canon, versions))
    }

    async fn load(_txn_id: TxnId, shape: Shape, store: fs::Dir<FE>) -> TCResult<Self> {
        let (dir, canon, versions) = fs_init(store).await?;
        let canon = DenseFile::load(canon, shape).await?;
        Ok(Self::new(dir, canon, versions))
    }

    fn dir(&self) -> fs::Inner<FE> {
        self.dir.clone()
    }
}

#[async_trait]
impl<Txn, FE, T, O> fs::CopyFrom<FE, O> for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + Clone,
    T: CDatatype + DType + de::FromStream<Context = ()>,
    O: DenseInstance<DType = T>,
    Buffer<T>: de::FromStream<Context = ()>,
{
    async fn copy_from(txn: &Txn, store: fs::Dir<FE>, other: O) -> TCResult<Self> {
        let (dir, canon, versions) = fs_init(store).await?;
        let canon = DenseFile::copy_from(canon, *txn.id(), other).await?;
        Ok(Self::new(dir, canon, versions))
    }
}

#[async_trait]
impl<Txn, FE, T> fs::Restore<FE> for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node> + Clone,
    T: CDatatype + DType + de::FromStream<Context = ()> + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        // always acquire these permits in-order to avoid the risk of a deadlock
        let _write_permit = self.write_permit(txn_id, Range::default()).await?;
        let _read_permit = backup.read_permit(txn_id, Range::default()).await?;

        let version = {
            let mut state = self.state.write().expect("dense state");
            state.pending_version(txn_id, self.canon.clone().into())?
        };

        let guard = version.write().await;
        guard.overwrite(txn_id, backup.clone()).await
    }
}

#[async_trait]
impl<Txn, FE, T> de::FromStream for DenseBase<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Buffer<T>> + ThreadSafe + Clone,
    T: CDatatype + DType,
    DenseFile<FE, T>: de::FromStream<Context = (DirLock<FE>, Shape)>,
{
    type Context = (Txn, Shape);

    async fn from_stream<D: de::Decoder>(
        cxt: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        let (txn, shape) = cxt;

        let (_dir_name, dir) = {
            let mut cxt = txn.context().write().await;
            cxt.create_dir_unique().map_err(de::Error::custom)?
        };

        let (dir, canon, versions) = dir_init(dir).map_err(de::Error::custom).await?;
        let canon = DenseFile::from_stream((canon, shape), decoder).await?;
        Ok(Self::new(dir, canon, versions))
    }
}

impl<Txn, FE, T: CDatatype> From<DenseBase<Txn, FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(base: DenseBase<Txn, FE, T>) -> Self {
        Self::Base(base)
    }
}

impl<Txn, FE, T> fmt::Debug for DenseBase<Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional dense tensor")
    }
}

pub(super) struct DenseComplexBaseVisitor<Txn, FE, T> {
    re: (DirLock<FE>, DirLock<FE>, DirLock<FE>),
    im: (DirLock<FE>, DirLock<FE>, DirLock<FE>),
    shape: Shape,
    phantom: PhantomData<(Txn, T)>,
}

impl<Txn, FE, T> DenseComplexBaseVisitor<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + Clone,
    T: CDatatype + DType + de::FromStream<Context = ()>,
    Buffer<T>: de::FromStream<Context = ()>,
{
    pub async fn end(self) -> TCResult<(DenseBase<Txn, FE, T>, DenseBase<Txn, FE, T>)> {
        let re = DenseFile::load(self.re.1, self.shape.clone());
        let im = DenseFile::load(self.im.1, self.shape);
        let (re, im) = try_join!(re, im)?;

        let re = DenseBase::new(self.re.0, re, self.re.2);
        let im = DenseBase::new(self.im.0, im, self.im.2);

        Ok((re, im))
    }
}

macro_rules! impl_from_stream_complex {
    ($t:ty, $decode:ident, $visit:ident) => {
        #[async_trait]
        impl<Txn, FE> de::FromStream for DenseComplexBaseVisitor<Txn, FE, $t>
        where
            Txn: Transaction<FE>,
            FE: DenseCacheFile + Clone,
        {
            type Context = (Txn, Shape);

            async fn from_stream<D: de::Decoder>(
                cxt: (Txn, Shape),
                decoder: &mut D,
            ) -> Result<Self, D::Error> {
                let (txn, shape) = cxt;

                let (re, im) = {
                    let dir = {
                        let mut cxt = txn.context().write().await;
                        let (_dir_name, dir) =
                            cxt.create_dir_unique().map_err(de::Error::custom)?;
                        dir
                    };

                    let mut dir = dir.write().await;

                    let re = dir
                        .create_dir(REAL.to_string())
                        .map_err(de::Error::custom)?;

                    let im = dir
                        .create_dir(IMAG.to_string())
                        .map_err(de::Error::custom)?;

                    (re, im)
                };

                let (re, im) = try_join!(dir_init(re), dir_init(im)).map_err(de::Error::custom)?;

                let visitor = Self {
                    re,
                    im,
                    shape,
                    phantom: PhantomData,
                };

                decoder.$decode(visitor).await
            }
        }

        #[async_trait]
        impl<Txn, FE> de::Visitor for DenseComplexBaseVisitor<Txn, FE, $t>
        where
            Txn: Transaction<FE>,
            FE: DenseCacheFile,
        {
            type Value = Self;

            fn expecting() -> &'static str {
                "complex dense tensor data"
            }

            async fn $visit<A: de::ArrayAccess<$t>>(
                self,
                mut array: A,
            ) -> Result<Self::Value, A::Error> {
                let (block_size, num_blocks) = ideal_block_size_for(&self.shape);

                let (mut contents_r, mut contents_i) = join!(self.re.1.write(), self.im.1.write());

                let type_size = <$t>::dtype().size();
                let mut buffer = Vec::<$t>::with_capacity(block_size);
                for block_id in 0..num_blocks {
                    let size = array.buffer(&mut buffer).await?;
                    if size % 2 != 0 {
                        return Err(de::Error::invalid_length(
                            size,
                            format!("{} complex numbers", size / 2),
                        ));
                    }

                    let re = buffer
                        .par_iter()
                        .take(size)
                        .step_by(2)
                        .copied()
                        .collect::<Vec<$t>>();

                    let im = buffer
                        .par_iter()
                        .take(size)
                        .skip(1)
                        .step_by(2)
                        .copied()
                        .collect::<Vec<$t>>();

                    let block_size = size * type_size;

                    contents_r
                        .create_file(block_id.to_string(), Buffer::from(re), block_size)
                        .map_err(de::Error::custom)?;

                    contents_i
                        .create_file(block_id.to_string(), Buffer::from(im), block_size)
                        .map_err(de::Error::custom)?;
                }

                std::mem::drop(contents_r);
                std::mem::drop(contents_i);

                Ok(self)
            }
        }
    };
}

impl_from_stream_complex!(f32, decode_array_f32, visit_array_f32);
impl_from_stream_complex!(f64, decode_array_f64, visit_array_f64);

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
        let mut guard = dir.write().await;
        let versions = guard.create_dir(fs::VERSIONS.to_string())?;
        let canon = guard.create_dir(CANON.to_string())?;
        (canon, versions)
    };

    Ok((dir, canon, versions))
}
