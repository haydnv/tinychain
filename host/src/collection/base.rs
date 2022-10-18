use std::convert::TryInto;
use std::fmt;

use async_trait::async_trait;
use destream::de;
use futures::future::{FutureExt, TryFutureExt};
use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
#[cfg(feature = "tensor")]
use tc_tensor::TensorPersist;
use tc_transact::fs::{CopyFrom, Dir, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction};
use tcgeneric::{Instance, NativeClass, TCPathBuf};

use crate::fs;
use crate::transact::TxnId;
use crate::txn::Txn;

use super::schema::{CollectionSchema, CollectionType};
use super::{BTree, BTreeFile, Collection, CollectionView, Table, TableIndex};
#[cfg(feature = "tensor")]
use super::{DenseTensor, DenseTensorFile, SparseTable, SparseTensor, Tensor, TensorType};

/// The base type of a [`Collection`] which supports [`CopyFrom`], [`Persist`], and [`Transact`]
#[derive(Clone)]
pub enum CollectionBase {
    BTree(BTreeFile),
    Table(TableIndex),
    #[cfg(feature = "tensor")]
    Dense(DenseTensor<DenseTensorFile>),
    #[cfg(feature = "tensor")]
    Sparse(SparseTensor<SparseTable>),
}

impl Instance for CollectionBase {
    type Class = CollectionType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => btree.class().into(),
            Self::Table(table) => table.class().into(),
            #[cfg(feature = "tensor")]
            Self::Dense(_dense) => TensorType::Dense.into(),
            #[cfg(feature = "tensor")]
            Self::Sparse(_sparse) => TensorType::Sparse.into(),
        }
    }
}

#[async_trait]
impl Persist<fs::Dir> for CollectionBase {
    type Schema = CollectionSchema;
    type Store = fs::Store;
    type Txn = Txn;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        match schema {
            CollectionSchema::BTree(btree_schema) => {
                let store = store.try_into()?;
                BTreeFile::create(txn, btree_schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
            CollectionSchema::Table(table_schema) => {
                let store = store.try_into()?;
                TableIndex::create(txn, table_schema, store)
                    .map_ok(Self::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(tensor_schema) => {
                let store = store.try_into()?;
                DenseTensor::create(txn, tensor_schema, store)
                    .map_ok(Self::Dense)
                    .await
            }
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(tensor_schema) => {
                let store = store.try_into()?;
                SparseTensor::create(txn, tensor_schema, store)
                    .map_ok(Self::Sparse)
                    .await
            }
        }
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        match schema {
            CollectionSchema::BTree(btree_schema) => {
                let store = store.try_into()?;
                BTreeFile::load(txn, btree_schema, store)
                    .map_ok(Self::BTree)
                    .await
            }
            CollectionSchema::Table(table_schema) => {
                let store = store.try_into()?;
                TableIndex::load(txn, table_schema, store)
                    .map_ok(Self::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            CollectionSchema::Dense(tensor_schema) => {
                let store = store.try_into()?;
                DenseTensor::load(txn, tensor_schema, store)
                    .map_ok(Self::Dense)
                    .await
            }
            #[cfg(feature = "tensor")]
            CollectionSchema::Sparse(tensor_schema) => {
                let store = store.try_into()?;
                SparseTensor::load(txn, tensor_schema, store)
                    .map_ok(Self::Sparse)
                    .await
            }
        }
    }
}

#[async_trait]
impl CopyFrom<fs::Dir, Collection> for CollectionBase {
    async fn copy_from(instance: Collection, store: Self::Store, txn: &Txn) -> TCResult<Self> {
        match instance {
            Collection::BTree(btree) => {
                let store = store.try_into()?;
                BTreeFile::copy_from(btree, store, txn)
                    .map_ok(Self::BTree)
                    .await
            }
            Collection::Table(table) => {
                let store = store.try_into()?;
                TableIndex::copy_from(table, store, txn)
                    .map_ok(Self::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            Collection::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => {
                    let store = store.try_into()?;
                    DenseTensor::copy_from(dense, store, txn)
                        .map_ok(Self::Dense)
                        .await
                }
                Tensor::Sparse(sparse) => {
                    let store = store.try_into()?;
                    SparseTensor::copy_from(sparse, store, txn)
                        .map_ok(Self::Sparse)
                        .await
                }
            },
        }
    }
}

#[async_trait]
impl Restore<fs::Dir> for CollectionBase {
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        match (self, backup) {
            (Self::BTree(btree), Self::BTree(backup)) => btree.restore(backup, txn_id).await,
            (Self::Table(table), Self::Table(backup)) => table.restore(backup, txn_id).await,
            #[cfg(feature = "tensor")]
            (Self::Dense(tensor), Self::Dense(backup)) => tensor.restore(backup, txn_id).await,
            #[cfg(feature = "tensor")]
            (Self::Sparse(tensor), Self::Sparse(backup)) => tensor.restore(backup, txn_id).await,
            (collection, backup) => Err(TCError::unsupported(format!(
                "cannot restore {} from {}",
                collection, backup
            ))),
        }
    }
}

pub enum CollectionBaseCommitGuard {
    BTree(<BTreeFile as Transact>::Commit),
    Table(<TableIndex as Transact>::Commit),
    #[cfg(feature = "tensor")]
    Dense(<DenseTensorFile as Transact>::Commit),
    #[cfg(feature = "tensor")]
    Sparse(<SparseTable as Transact>::Commit),
}

#[async_trait]
impl Transact for CollectionBase {
    type Commit = CollectionBaseCommitGuard;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        match self {
            Self::BTree(btree) => {
                btree
                    .commit(txn_id)
                    .map(CollectionBaseCommitGuard::BTree)
                    .await
            }
            Self::Table(table) => {
                table
                    .commit(txn_id)
                    .map(CollectionBaseCommitGuard::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => {
                dense
                    .commit(txn_id)
                    .map(CollectionBaseCommitGuard::Dense)
                    .await
            }
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => {
                sparse
                    .commit(txn_id)
                    .map(CollectionBaseCommitGuard::Sparse)
                    .await
            }
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => dense.finalize(txn_id).await,
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => sparse.finalize(txn_id).await,
        }
    }
}

/// A [`de::Visitor`] used to deserialize a [`Collection`].
pub struct CollectionVisitor {
    txn: Txn,
}

impl CollectionVisitor {
    pub fn new(txn: Txn) -> Self {
        Self { txn }
    }

    pub async fn visit_map_value<A: de::MapAccess>(
        self,
        class: CollectionType,
        access: &mut A,
    ) -> Result<CollectionBase, A::Error> {
        debug!("deserialize Collection");

        match class {
            CollectionType::BTree(_) => {
                let file = self
                    .txn
                    .context()
                    .create_file_unique(*self.txn.id())
                    .map_err(de::Error::custom)
                    .await?;

                access
                    .next_value((self.txn.clone(), file))
                    .map_ok(CollectionBase::BTree)
                    .await
            }

            CollectionType::Table(_) => {
                access
                    .next_value(self.txn)
                    .map_ok(CollectionBase::Table)
                    .await
            }

            #[cfg(feature = "tensor")]
            CollectionType::Tensor(tt) => match tt {
                TensorType::Dense => {
                    let tensor = access.next_value(self.txn).await?;
                    Ok(CollectionBase::Dense(tensor))
                }
                TensorType::Sparse => {
                    let tensor = access.next_value(self.txn).await?;
                    Ok(CollectionBase::Sparse(tensor))
                }
            },
        }
    }
}

#[async_trait]
impl de::Visitor for CollectionVisitor {
    type Value = CollectionBase;

    fn expecting() -> &'static str {
        "a Collection"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let classpath = map
            .next_key::<TCPathBuf>(())
            .await?
            .ok_or_else(|| de::Error::custom("expected a Collection type"))?;

        let class = CollectionType::from_path(&classpath)
            .ok_or_else(|| de::Error::invalid_value(classpath, "a Collection type"))?;

        self.visit_map_value(class, &mut map).await
    }
}

#[async_trait]
impl de::FromStream for CollectionBase {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionVisitor { txn }).await
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for CollectionBase {
    type Txn = Txn;
    type View = CollectionView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        Collection::from(self).into_view(txn).await
    }
}

impl fmt::Display for CollectionBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Table(table) => fmt::Display::fmt(table, f),
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => fmt::Display::fmt(dense, f),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => fmt::Display::fmt(sparse, f),
        }
    }
}

impl TryCastFrom<Collection> for CollectionBase {
    fn can_cast_from(collection: &Collection) -> bool {
        match collection {
            Collection::BTree(BTree::File(_)) => true,
            Collection::Table(Table::Table(_)) => true,
            #[cfg(feature = "tensor")]
            Collection::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => dense.is_persistent(),
                Tensor::Sparse(sparse) => sparse.is_persistent(),
            },
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection) -> Option<Self> {
        match collection {
            Collection::BTree(BTree::File(btree)) => Some(Self::BTree(btree)),
            Collection::Table(Table::Table(table)) => Some(Self::Table(table)),
            #[cfg(feature = "tensor")]
            Collection::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => dense.as_persistent().map(Self::Dense),
                Tensor::Sparse(sparse) => sparse.as_persistent().map(Self::Sparse),
            },
            _ => None,
        }
    }
}

impl From<CollectionBase> for Collection {
    fn from(collection: CollectionBase) -> Self {
        match collection {
            CollectionBase::BTree(btree) => Self::BTree(btree.into()),
            CollectionBase::Table(table) => Self::Table(table.into()),
            #[cfg(feature = "tensor")]
            CollectionBase::Dense(dense) => Self::Tensor(dense.into()),
            #[cfg(feature = "tensor")]
            CollectionBase::Sparse(sparse) => Self::Tensor(sparse.into()),
        }
    }
}
