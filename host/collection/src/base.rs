use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::de;
use futures::TryFutureExt;
use log::{debug, info};
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Persist, Restore};
use tc_transact::hash::{AsyncHash, Output, Sha256};
use tc_transact::{fs, IntoView, Transact, Transaction, TxnId};
use tcgeneric::{Instance, NativeClass, TCPathBuf, ThreadSafe};

#[cfg(feature = "btree")]
use crate::btree::{BTree, BTreeFile, BTreeInstance};
#[cfg(feature = "table")]
use crate::table::{Table, TableFile, TableInstance};
#[cfg(feature = "tensor")]
use crate::tensor::{
    Dense, DenseBase, Sparse, SparseBase, Tensor, TensorBase, TensorInstance, TensorType,
};
use crate::{Collection, CollectionBlock, CollectionType, CollectionView, Schema};

/// The base type of a mutable transactional collection of data.
pub enum CollectionBase<Txn, FE> {
    Null(Dir<FE>, PhantomData<Txn>),
    #[cfg(feature = "btree")]
    BTree(BTreeFile<Txn, FE>),
    #[cfg(feature = "table")]
    Table(TableFile<Txn, FE>),
    #[cfg(feature = "tensor")]
    Tensor(TensorBase<Txn, FE>),
}

impl<Txn, FE> Clone for CollectionBase<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Null(dir, txn) => Self::Null(dir.clone(), *txn),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => Self::BTree(btree.clone()),
            #[cfg(feature = "table")]
            Self::Table(table) => Self::Table(table.clone()),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
        }
    }
}

impl<Txn, FE> CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: CollectionBlock,
{
    /// Return the [`Schema`] of this [`Collection`]
    pub fn schema(&self) -> Schema {
        match self {
            Self::Null(_, _) => Schema::Null,
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.schema().clone().into(),
            #[cfg(feature = "table")]
            Self::Table(table) => table.schema().clone().into(),
            #[cfg(feature = "tensor")]
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
            Self::Null(_, _) => CollectionType::Null,
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.class().into(),
            #[cfg(feature = "table")]
            Self::Table(table) => table.class().into(),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.class().into(),
        }
    }
}

#[async_trait]
impl<Txn, FE> Transact for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: CollectionBlock,
{
    type Commit = ();

    #[allow(unused_variables)]
    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        match self {
            Self::Null(_, _) => (),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.commit(txn_id).await,
            #[cfg(feature = "table")]
            Self::Table(table) => table.commit(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    #[allow(unused_variables)]
    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Null(_, _) => (),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.rollback(txn_id).await,
            #[cfg(feature = "table")]
            Self::Table(table) => table.rollback(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }

    #[allow(unused_variables)]
    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Null(_, _) => (),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.finalize(txn_id).await,
            #[cfg(feature = "table")]
            Self::Table(table) => table.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.finalize(txn_id).await,
        }
    }
}

#[async_trait]
impl<T, FE> AsyncHash for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: CollectionBlock,
{
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        Collection::from(self.clone()).hash(txn_id).await
    }
}

#[async_trait]
impl<Txn, FE> Persist<FE> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: CollectionBlock,
{
    type Txn = Txn;
    type Schema = Schema;

    #[allow(unused_variables)]
    async fn create(txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        debug!("create persistent mutable collection at {store:?}");

        match schema {
            Schema::Null => Ok(Self::Null(store, PhantomData)),
            #[cfg(feature = "btree")]
            Schema::BTree(schema) => {
                BTreeFile::create(txn_id, schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
            #[cfg(feature = "table")]
            Schema::Table(schema) => {
                TableFile::create(txn_id, schema, store)
                    .map_ok(Self::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            Schema::Dense(schema) => {
                DenseBase::create(txn_id, schema, store)
                    .map_ok(TensorBase::Dense)
                    .map_ok(Self::Tensor)
                    .await
            }
            #[cfg(feature = "tensor")]
            Schema::Sparse(schema) => {
                SparseBase::create(txn_id, schema.into(), store)
                    .map_ok(TensorBase::Sparse)
                    .map_ok(Self::Tensor)
                    .await
            }
        }
    }

    #[allow(unused_variables)]
    async fn load(txn_id: TxnId, schema: Schema, store: Dir<FE>) -> TCResult<Self> {
        info!("load persistent mutable collection at {store:?}");

        match schema {
            Schema::Null => Ok(Self::Null(store, PhantomData)),
            #[cfg(feature = "btree")]
            Schema::BTree(schema) => {
                BTreeFile::load(txn_id, schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
            #[cfg(feature = "table")]
            Schema::Table(schema) => {
                TableFile::load(txn_id, schema, store)
                    .map_ok(Self::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            Schema::Dense(schema) => {
                DenseBase::load(txn_id, schema, store)
                    .map_ok(TensorBase::Dense)
                    .map_ok(Self::Tensor)
                    .await
            }
            #[cfg(feature = "tensor")]
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
            Self::Null(store, _) => store.clone().into_inner(),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.dir(),
            #[cfg(feature = "table")]
            Self::Table(table) => table.dir(),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.dir(),
        }
    }
}

#[async_trait]
impl<Txn, FE> CopyFrom<FE, Collection<Txn, FE>> for CollectionBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: CollectionBlock,
{
    #[allow(unused_variables)]
    async fn copy_from(txn: &Txn, store: Dir<FE>, instance: Collection<Txn, FE>) -> TCResult<Self> {
        match instance {
            Collection::Null(_, _) => Ok(Self::Null(store, PhantomData)),
            #[cfg(feature = "btree")]
            Collection::BTree(instance) => {
                BTreeFile::copy_from(txn, store, instance)
                    .map_ok(Self::BTree)
                    .await
            }
            #[cfg(feature = "table")]
            Collection::Table(instance) => {
                TableFile::copy_from(txn, store, instance)
                    .map_ok(Self::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
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
    FE: CollectionBlock,
{
    #[allow(unused_variables)]
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        match (self, backup) {
            (Self::Null(_, _), Self::Null(_, _)) => Ok(()),
            #[cfg(feature = "btree")]
            (Self::BTree(this), Self::BTree(backup)) => this.restore(txn_id, backup).await,
            #[cfg(feature = "table")]
            (Self::Table(this), Self::Table(backup)) => this.restore(txn_id, backup).await,
            #[cfg(feature = "tensor")]
            (Self::Tensor(this), Self::Tensor(backup)) => this.restore(txn_id, backup).await,
            #[cfg(any(feature = "btree", feature = "table", feature = "tensor"))]
            (this, that) => Err(bad_request!("cannot restore {:?} from {:?}", this, that)),
        }
    }
}

impl<T, FE> TryCastFrom<Collection<T, FE>> for CollectionBase<T, FE> {
    fn can_cast_from(collection: &Collection<T, FE>) -> bool {
        match collection {
            Collection::Null(_, _) => true,
            #[cfg(feature = "btree")]
            Collection::BTree(BTree::File(_)) => true,
            #[cfg(feature = "table")]
            Collection::Table(Table::Table(_)) => true,
            #[cfg(feature = "tensor")]
            Collection::Tensor(Tensor::Dense(Dense::Base(_))) => true,
            #[cfg(feature = "tensor")]
            Collection::Tensor(Tensor::Sparse(Sparse::Base(_))) => true,
            #[cfg(any(feature = "btree", feature = "table", feature = "tensor"))]
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection<T, FE>) -> Option<Self> {
        match collection {
            Collection::Null(dir, data) => Some(Self::Null(dir, data)),
            #[cfg(feature = "btree")]
            Collection::BTree(BTree::File(btree)) => Some(Self::BTree(btree)),
            #[cfg(feature = "table")]
            Collection::Table(Table::Table(table)) => Some(Self::Table(table)),
            #[cfg(feature = "tensor")]
            Collection::Tensor(Tensor::Dense(Dense::Base(dense))) => {
                Some(Self::Tensor(TensorBase::Dense(dense)))
            }
            #[cfg(feature = "tensor")]
            Collection::Tensor(Tensor::Sparse(Sparse::Base(sparse))) => {
                Some(Self::Tensor(TensorBase::Sparse(sparse)))
            }
            #[cfg(any(feature = "btree", feature = "table", feature = "tensor"))]
            _ => None,
        }
    }
}

#[async_trait]
impl<'en, T, FE> IntoView<'en, FE> for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: CollectionBlock,
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
    FE: CollectionBlock,
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
            CollectionType::Null => {
                let _: () = access.next_value(()).await?;
                let dir = self
                    .txn
                    .context()
                    .and_then(|dir| fs::Dir::load(*self.txn.id(), dir))
                    .map_err(de::Error::custom)
                    .await?;

                Ok(CollectionBase::Null(dir, PhantomData))
            }

            #[cfg(feature = "btree")]
            CollectionType::BTree(_) => {
                access
                    .next_value(self.txn)
                    .map_ok(CollectionBase::BTree)
                    .await
            }

            #[cfg(feature = "table")]
            CollectionType::Table(_) => {
                access
                    .next_value(self.txn)
                    .map_ok(CollectionBase::Table)
                    .await
            }

            #[cfg(feature = "tensor")]
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
    FE: CollectionBlock,
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
    FE: CollectionBlock,
{
    type Context = T;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionVisitor::new(txn)).await
    }
}
