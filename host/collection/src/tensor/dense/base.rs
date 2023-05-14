use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use collate::Collator;
use destream::de;
use ds_ext::{OrdHashMap, OrdHashSet};
use fensor::{Buffer, CDatatype, DenseFile};
use freqfs::{DirLock, FileLoad};
use safecast::AsType;
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::{Dir, Inner, Persist, VERSIONS};
use tc_transact::{Transaction, TxnId};
use tc_value::{DType, NumberInstance, NumberType};
use tcgeneric::{label, Instance, Label, ThreadSafe};

use crate::tensor::{Range, Shape, TensorInstance, TensorType};

const CANON: Label = label("canon");

type Version<FE, T> = DenseFile<FE, T>;

type Semaphore = tc_transact::lock::Semaphore<Collator<u64>, Range>;

#[derive(Clone)]
struct Delta<FE, T> {
    original: Version<FE, T>,
    modified: Version<FE, T>,
}

struct State<FE, T> {
    commits: OrdHashSet<TxnId>,
    deltas: OrdHashMap<TxnId, Delta<FE, T>>,
    pending: OrdHashMap<TxnId, Delta<FE, T>>,
    versions: DirLock<FE>,
    finalized: Option<TxnId>,
}

pub struct DenseTensor<Txn, FE, T> {
    dir: DirLock<FE>,
    canon: Version<FE, T>,
    state: Arc<RwLock<State<FE, T>>>,
    semaphore: Semaphore,
    phantom: PhantomData<Txn>,
}

impl<Txn, FE, T> Clone for DenseTensor<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            canon: self.canon.clone(),
            state: self.state.clone(),
            semaphore: self.semaphore.clone(),
            phantom: self.phantom,
        }
    }
}

impl<Txn, FE, T> DenseTensor<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: AsType<Buffer<T>> + ThreadSafe,
    T: CDatatype,
{
    fn new(dir: DirLock<FE>, canon: Version<FE, T>, versions: DirLock<FE>) -> Self {
        let semaphore = Semaphore::new(Arc::new(Collator::default()));

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

impl<Txn, FE, T> Instance for DenseTensor<Txn, FE, T>
where
    Self: Send + Sync,
{
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Sparse
    }
}

impl<Txn, FE, T> fensor::TensorInstance for DenseTensor<Txn, FE, T>
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

impl<Txn, FE, T> TensorInstance for DenseTensor<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
}

#[async_trait]
impl<Txn, FE, T> Persist<FE> for DenseTensor<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: FileLoad + AsType<Buffer<T>> + ThreadSafe,
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

        let canon = Version::constant(canon, shape, T::zero()).await?;

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

        let canon = Version::load(canon, shape).await?;

        Ok(Self::new(dir, canon, versions))
    }

    fn dir(&self) -> Inner<FE> {
        self.dir.clone()
    }
}
