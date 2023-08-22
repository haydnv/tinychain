use std::fmt;
use std::marker::PhantomData;

use async_hash::{Output, Sha256};
use async_trait::async_trait;
use destream::de;
use futures::TryFutureExt;
use log::{debug, info};
use safecast::{AsType, TryCastFrom};

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Persist, Restore};
use tc_transact::{fs, AsyncHash, IntoView, Transact, Transaction, TxnId};
use tcgeneric::{Instance, NativeClass, TCPathBuf, ThreadSafe};

use super::btree::{BTree, BTreeFile, BTreeInstance};
use super::table::{Table, TableFile, TableInstance};
use super::tensor::{
    Dense, DenseBase, DenseCacheFile, Sparse, SparseBase, Tensor, TensorBase, TensorInstance,
    TensorType,
};
use super::{BTreeNode, Collection, CollectionType, CollectionView, Schema, TensorNode};

#[derive(Clone)]
pub enum CollectionBase<Txn, FE> {
    BTree(BTreeFile<Txn, FE>),
    Table(TableFile<Txn, FE>),
    Tensor(TensorBase<Txn, FE>),
}

impl<Txn, FE> CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<BTreeNode> + ThreadSafe,
{
    /// Get the [`Schema`] of this [`Collection`]
    pub fn schema(&self) -> Schema {
        match self {
            Self::BTree(btree) => btree.schema().clone().into(),
            Self::Table(table) => table.schema().clone().into(),
            Self::Tensor(tensor) => match tensor {
                TensorBase::Dense(dense) => Schema::Dense(dense.schema()),
                TensorBase::Sparse(sparse) => Schema::Sparse(sparse.schema()),
            },
        }
    }
}

impl<Txn, FE> Instance for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: ThreadSafe,
{
    type Class = CollectionType;

    fn class(&self) -> CollectionType {
        match self {
            Self::BTree(btree) => btree.class().into(),
            Self::Table(table) => table.class().into(),
            Self::Tensor(tensor) => tensor.class().into(),
        }
    }
}

#[async_trait]
impl<Txn, FE> Transact for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + for<'en> fs::FileSave<'en>,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::Tensor(tensor) => tensor.finalize(txn_id).await,
        }
    }
}

#[async_trait]
impl<T, FE> AsyncHash for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        Collection::from(self).hash(txn_id).await
    }
}

#[async_trait]
impl<Txn, FE> Persist<FE> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
    BTreeNode: freqfs::FileLoad,
{
    type Txn = Txn;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        info!("create persistent mutable collection at {store:?}");

        match schema {
            Schema::BTree(schema) => {
                BTreeFile::create(txn_id, schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
            Schema::Table(schema) => {
                TableFile::create(txn_id, schema, store)
                    .map_ok(Self::Table)
                    .await
            }
            Schema::Dense(schema) => {
                DenseBase::create(txn_id, schema, store)
                    .map_ok(TensorBase::Dense)
                    .map_ok(Self::Tensor)
                    .await
            }
            Schema::Sparse(schema) => {
                SparseBase::create(txn_id, schema.into(), store)
                    .map_ok(TensorBase::Sparse)
                    .map_ok(Self::Tensor)
                    .await
            }
        }
    }

    async fn load(txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        info!("load persistent mutable collection at {store:?}");

        match schema {
            Schema::BTree(schema) => {
                BTreeFile::load(txn_id, schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
            Schema::Table(schema) => {
                TableFile::load(txn_id, schema, store)
                    .map_ok(Self::Table)
                    .await
            }
            Schema::Dense(schema) => {
                DenseBase::load(txn_id, schema, store)
                    .map_ok(TensorBase::Dense)
                    .map_ok(Self::Tensor)
                    .await
            }
            Schema::Sparse(schema) => {
                SparseBase::load(txn_id, schema.into(), store)
                    .map_ok(TensorBase::Sparse)
                    .map_ok(Self::Tensor)
                    .await
            }
        }
    }

    fn dir(&self) -> tc_transact::fs::Inner<FE> {
        match self {
            Self::BTree(btree) => btree.dir(),
            Self::Table(table) => table.dir(),
            Self::Tensor(tensor) => tensor.dir(),
        }
    }
}

#[async_trait]
impl<Txn, FE> CopyFrom<FE, Collection<Txn, FE>> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
    BTreeNode: freqfs::FileLoad,
{
    async fn copy_from(txn: &Txn, store: Dir<FE>, instance: Collection<Txn, FE>) -> TCResult<Self> {
        match instance {
            Collection::BTree(instance) => {
                BTreeFile::copy_from(txn, store, instance)
                    .map_ok(Self::BTree)
                    .await
            }
            Collection::Table(instance) => {
                TableFile::copy_from(txn, store, instance)
                    .map_ok(Self::Table)
                    .await
            }
            Collection::Tensor(instance) => {
                TensorBase::copy_from(txn, store, instance.into())
                    .map_ok(Self::Tensor)
                    .await
            }
        }
    }
}

#[async_trait]
impl<Txn, FE> Restore<FE> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
    BTreeNode: freqfs::FileLoad,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        match (self, backup) {
            (Self::BTree(this), Self::BTree(backup)) => this.restore(txn_id, backup).await,
            (Self::Table(this), Self::Table(backup)) => this.restore(txn_id, backup).await,
            (Self::Tensor(this), Self::Tensor(backup)) => this.restore(txn_id, backup).await,
            (this, that) => Err(bad_request!("cannot restore {:?} from {:?}", this, that)),
        }
    }
}

impl<T, FE> TryCastFrom<Collection<T, FE>> for CollectionBase<T, FE> {
    fn can_cast_from(collection: &Collection<T, FE>) -> bool {
        match collection {
            Collection::BTree(BTree::File(_)) => true,
            Collection::Table(Table::Table(_)) => true,
            Collection::Tensor(Tensor::Dense(Dense::Base(_))) => true,
            Collection::Tensor(Tensor::Sparse(Sparse::Base(_))) => true,
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection<T, FE>) -> Option<Self> {
        match collection {
            Collection::BTree(BTree::File(btree)) => Some(Self::BTree(btree)),
            Collection::Table(Table::Table(table)) => Some(Self::Table(table)),
            Collection::Tensor(Tensor::Dense(Dense::Base(dense))) => {
                Some(Self::Tensor(TensorBase::Dense(dense)))
            }
            Collection::Tensor(Tensor::Sparse(Sparse::Base(sparse))) => {
                Some(Self::Tensor(TensorBase::Sparse(sparse)))
            }
            _ => None,
        }
    }
}

#[async_trait]
impl<'en, T, FE> IntoView<'en, FE> for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
    Self: 'en,
{
    type Txn = T;
    type View = CollectionView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        Collection::from(self).into_view(txn).await
    }
}

impl<T, FE> fmt::Debug for CollectionBase<T, FE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a Collection")
    }
}

/// A [`de::Visitor`] used to deserialize a [`Collection`].
pub struct CollectionVisitor<Txn, FE> {
    txn: Txn,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> CollectionVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
{
    pub fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }

    pub async fn visit_map_value<A: de::MapAccess>(
        self,
        class: CollectionType,
        access: &mut A,
    ) -> Result<CollectionBase<Txn, FE>, A::Error> {
        debug!("CollectionVisitor::visit_map_value with class {:?}", class);

        match class {
            CollectionType::BTree(_) => {
                access
                    .next_value(self.txn)
                    .map_ok(CollectionBase::BTree)
                    .await
            }

            CollectionType::Table(_) => {
                access
                    .next_value(self.txn)
                    .map_ok(CollectionBase::Table)
                    .await
            }

            CollectionType::Tensor(tt) => match tt {
                TensorType::Dense => {
                    access
                        .next_value(self.txn)
                        .map_ok(TensorBase::Dense)
                        .map_ok(CollectionBase::Tensor)
                        .await
                }
                TensorType::Sparse => {
                    access
                        .next_value(self.txn)
                        .map_ok(TensorBase::Sparse)
                        .map_ok(CollectionBase::Tensor)
                        .await
                }
            },
        }
    }
}

#[async_trait]
impl<T, FE> de::Visitor for CollectionVisitor<T, FE>
where
    T: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
{
    type Value = CollectionBase<T, FE>;

    fn expecting() -> &'static str {
        "a Collection"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let classpath: TCPathBuf = map
            .next_key(())
            .await?
            .ok_or_else(|| de::Error::custom("expected a Collection type"))?;

        let class = CollectionType::from_path(&classpath)
            .ok_or_else(|| de::Error::invalid_value(classpath, "a Collection type"))?;

        self.visit_map_value(class, &mut map).await
    }
}

#[async_trait]
impl<T, FE> de::FromStream for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
{
    type Context = T;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionVisitor::new(txn)).await
    }
}
