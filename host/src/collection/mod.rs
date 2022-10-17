//! A [`Collection`] such as a [`BTree`] or [`Table`].

use std::fmt;

use async_hash::hash_try_stream;
use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use sha2::digest::Output;
use sha2::Sha256;

use tc_btree::{BTreeInstance, BTreeView, Node, NodeId};
use tc_error::*;
use tc_table::{TableStream, TableView};
#[cfg(feature = "tensor")]
use tc_tensor::{Array, TensorView};
use tc_transact::fs::{CopyFrom, Dir, DirCreate, DirCreateFile};
use tc_transact::{IntoView, Transaction};
use tcgeneric::{path_label, Id, Instance, NativeClass, PathLabel};

use crate::fs;
use crate::txn::Txn;

pub use tc_btree::BTreeType;
pub use tc_table::TableType;

pub use base::{CollectionBase, CollectionBaseCommitGuard, CollectionVisitor};
pub use schema::{CollectionSchema, CollectionType};

mod base;
mod schema;

#[cfg(feature = "tensor")]
pub use tc_tensor::{DenseAccess, SparseAccess, TensorType};

pub type BTree = tc_btree::BTree<fs::File<NodeId, Node>, fs::Dir, Txn>;
pub type BTreeFile = tc_btree::BTreeFile<fs::File<NodeId, Node>, fs::Dir, Txn>;

pub type Table = tc_table::Table<fs::File<NodeId, Node>, fs::Dir, Txn>;
pub type TableIndex = tc_table::TableIndex<fs::File<NodeId, Node>, fs::Dir, Txn>;

#[cfg(feature = "tensor")]
pub type Tensor = tc_tensor::Tensor<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn>;
#[cfg(feature = "tensor")]
pub type DenseAccessor =
    tc_tensor::DenseAccessor<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn>;
#[cfg(feature = "tensor")]
pub type DenseTensor<B> =
    tc_tensor::DenseTensor<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn, B>;
#[cfg(feature = "tensor")]
pub type DenseTensorFile =
    tc_tensor::BlockListFile<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn>;
#[cfg(feature = "tensor")]
pub type SparseAccessor =
    tc_tensor::SparseAccessor<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn>;
#[cfg(feature = "tensor")]
pub type SparseTensor<A> =
    tc_tensor::SparseTensor<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn, A>;
#[cfg(feature = "tensor")]
pub type SparseTable =
    tc_tensor::SparseTable<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn>;

pub(crate) const PREFIX: PathLabel = path_label(&["state", "collection"]);

/// A stateful, transaction-aware [`Collection`], such as a [`BTree`] or [`Table`].
#[derive(Clone)]
pub enum Collection {
    BTree(BTree),
    Table(Table),
    #[cfg(feature = "tensor")]
    Tensor(Tensor),
}

impl Instance for Collection {
    type Class = CollectionType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => CollectionType::BTree(btree.class()),
            Self::Table(table) => CollectionType::Table(table.class()),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => CollectionType::Tensor(tensor.class()),
        }
    }
}

impl Collection {
    pub async fn copy_from(
        txn: &Txn,
        container: &fs::Dir,
        name: Id,
        source: Collection,
    ) -> TCResult<Collection> {
        let txn_id = *txn.id();
        let mut lock = container.write(txn_id).await?;

        match source {
            Collection::BTree(source) => {
                let file = lock.create_file(name)?;
                BTreeFile::copy_from(source, file, txn)
                    .map_ok(BTree::from)
                    .map_ok(Collection::BTree)
                    .await
            }
            Collection::Table(source) => {
                let dir = lock.create_dir(name)?;
                TableIndex::copy_from(source, dir, txn)
                    .map_ok(Table::from)
                    .map_ok(Collection::Table)
                    .await
            }
            #[cfg(feature = "tensor")]
            Collection::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => {
                    let file = lock.create_file(name)?;
                    DenseTensor::copy_from(dense, file, txn)
                        .map_ok(Tensor::from)
                        .map_ok(Collection::Tensor)
                        .await
                }
                Tensor::Sparse(sparse) => {
                    let dir = lock.create_dir(name)?;
                    SparseTensor::copy_from(sparse, dir, txn)
                        .map_ok(Tensor::from)
                        .map_ok(Collection::Tensor)
                        .await
                }
            },
        }
    }

    pub async fn hash(self, txn: Txn) -> TCResult<Output<Sha256>> {
        match self {
            Self::BTree(btree) => {
                let keys = btree.keys(*txn.id()).await?;
                hash_try_stream::<Sha256, _, _, _>(keys).await
            }
            Self::Table(table) => {
                let rows = table.rows(*txn.id()).await?;
                hash_try_stream::<Sha256, _, _, _>(rows).await
            }
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => match tensor {
                tc_tensor::Tensor::Dense(dense) => {
                    let elements = dense.into_inner().value_stream(txn).await?;
                    hash_try_stream::<Sha256, _, _, _>(elements).await
                }
                tc_tensor::Tensor::Sparse(sparse) => {
                    let filled = sparse.into_inner().filled(txn).await?;
                    hash_try_stream::<Sha256, _, _, _>(filled).await
                }
            },
        }
    }
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Self {
        Self::BTree(btree)
    }
}

impl From<BTreeFile> for Collection {
    fn from(btree: BTreeFile) -> Self {
        Self::BTree(btree.into())
    }
}

impl From<Table> for Collection {
    fn from(table: Table) -> Self {
        Self::Table(table)
    }
}

impl From<TableIndex> for Collection {
    fn from(table: TableIndex) -> Self {
        Self::Table(table.into())
    }
}

#[cfg(feature = "tensor")]
impl From<Tensor> for Collection {
    fn from(tensor: Tensor) -> Self {
        Self::Tensor(tensor)
    }
}

#[cfg(feature = "tensor")]
impl<B: DenseAccess<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn>>
    From<DenseTensor<B>> for Collection
{
    fn from(tensor: DenseTensor<B>) -> Self {
        Self::Tensor(tensor.into())
    }
}

#[cfg(feature = "tensor")]
impl<A: SparseAccess<fs::File<u64, Array>, fs::File<NodeId, Node>, fs::Dir, Txn>>
    From<SparseTensor<A>> for Collection
{
    fn from(tensor: SparseTensor<A>) -> Self {
        Self::Tensor(tensor.into())
    }
}

#[cfg(feature = "tensor")]
impl safecast::TryCastFrom<Collection> for Tensor {
    fn can_cast_from(collection: &Collection) -> bool {
        match collection {
            Collection::Tensor(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection) -> Option<Self> {
        match collection {
            Collection::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }
}

#[async_trait]
impl de::FromStream for Collection {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        CollectionBase::from_stream(txn, decoder)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Collection {
    type Txn = Txn;
    type View = CollectionView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::BTree(btree) => btree.into_view(txn).map_ok(CollectionView::BTree).await,
            Self::Table(table) => table.into_view(txn).map_ok(CollectionView::Table).await,
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.into_view(txn).map_ok(CollectionView::Tensor).await,
        }
    }
}

impl fmt::Debug for Collection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Debug::fmt(btree, f),
            Self::Table(table) => fmt::Debug::fmt(table, f),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => fmt::Debug::fmt(tensor, f),
        }
    }
}

impl fmt::Display for Collection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
            Self::Table(table) => fmt::Display::fmt(table, f),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => fmt::Display::fmt(tensor, f),
        }
    }
}

/// A view of a [`Collection`] within a single `Transaction`, used for serialization.
pub enum CollectionView<'en> {
    BTree(BTreeView<'en>),
    Table(TableView<'en>),
    #[cfg(feature = "tensor")]
    Tensor(TensorView<'en>),
}

impl<'en> en::IntoStream<'en> for CollectionView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use destream::en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;

        match self {
            Self::BTree(btree) => map.encode_entry(BTreeType::default().path(), btree),
            Self::Table(table) => map.encode_entry(TableType::default().path(), table),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => match tensor {
                TensorView::Dense(dense) => map.encode_entry(TensorType::Dense.path(), dense),
                TensorView::Sparse(sparse) => map.encode_entry(TensorType::Sparse.path(), sparse),
            },
        }?;

        map.end()
    }
}
