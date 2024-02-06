use std::fmt;

use async_hash::{Digest, Hash, Output, Sha256};
use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use safecast::{as_type, AsType, TryCastFrom};

use tc_error::*;
use tc_transact::{AsyncHash, IntoView, Transaction, TxnId};
use tcgeneric::{
    path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf, ThreadSafe,
};

use btree::{BTreeInstance, BTreeType};
use table::{TableInstance, TableStream, TableType};
use tensor::TensorType;

pub use base::{CollectionBase, CollectionVisitor};
pub use btree::{BTree, BTreeFile, Node as BTreeNode};
pub use schema::Schema;
pub use table::Table;
pub use tensor::{
    Dense, DenseBase, DenseCacheFile, DenseView, Node as TensorNode, Sparse, SparseBase,
    SparseView, Tensor, TensorBase, TensorInstance, TensorView,
};

mod base;
mod schema;

pub mod btree;
pub mod public;
pub mod table;
pub mod tensor;

/// The prefix of the absolute path to [`Collection`] data types
pub const PREFIX: PathLabel = path_label(&["state", "collection"]);

/// The [`Class`] of a `Collection`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollectionType {
    BTree(BTreeType),
    Table(TableType),
    Tensor(TensorType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() > 2 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                "btree" => BTreeType::from_path(path).map(Self::BTree),
                "table" => TableType::from_path(path).map(Self::Table),
                "tensor" => TensorType::from_path(path).map(Self::Tensor),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            Self::BTree(btt) => btt.path(),
            Self::Table(tt) => tt.path(),
            Self::Tensor(tt) => tt.path(),
        }
    }
}

as_type!(CollectionType, BTree, BTreeType);
as_type!(CollectionType, Table, TableType);
as_type!(CollectionType, Tensor, TensorType);

impl fmt::Debug for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btt) => fmt::Debug::fmt(btt, f),
            Self::Table(tt) => fmt::Debug::fmt(tt, f),
            Self::Tensor(tt) => fmt::Debug::fmt(tt, f),
        }
    }
}

/// A mutable transactional collection of data.
pub enum Collection<Txn, FE> {
    BTree(BTree<Txn, FE>),
    Table(Table<Txn, FE>),
    Tensor(Tensor<Txn, FE>),
}

impl<Txn, FE> Clone for Collection<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::BTree(btree) => Self::BTree(btree.clone()),
            Self::Table(table) => Self::Table(table.clone()),
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
        }
    }
}

as_type!(Collection<Txn, FE>, BTree, BTree<Txn, FE>);
as_type!(Collection<Txn, FE>, Table, Table<Txn, FE>);
as_type!(Collection<Txn, FE>, Tensor, Tensor<Txn, FE>);

impl<Txn, FE> Collection<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<BTreeNode> + ThreadSafe,
{
    /// Return the [`Schema`] of this [`Collection`].
    pub fn schema(&self) -> Schema {
        match self {
            Self::BTree(btree) => btree.schema().clone().into(),
            Self::Table(table) => table.schema().clone().into(),
            Self::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => Schema::Dense(dense.schema()),
                Tensor::Sparse(sparse) => Schema::Sparse(sparse.schema()),
            },
        }
    }
}

impl<T, FE> Instance for Collection<T, FE>
where
    T: Transaction<FE>,
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
impl<Txn, FE> AsyncHash for Collection<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<btree::Node> + AsType<tensor::Node> + Clone,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let schema_hash = Hash::<Sha256>::hash(self.schema());

        let contents_hash = match self {
            Self::BTree(btree) => {
                let keys = btree.keys(txn_id).await?;
                async_hash::hash_try_stream::<Sha256, _, _, _>(keys).await?
            }
            Self::Table(table) => {
                let rows = table.rows(txn_id).await?;
                async_hash::hash_try_stream::<Sha256, _, _, _>(rows).await?
            }
            Self::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => {
                    let elements = DenseView::from(dense).into_elements(txn_id).await?;
                    async_hash::hash_try_stream::<Sha256, _, _, _>(elements).await?
                }
                Tensor::Sparse(sparse) => {
                    let elements = SparseView::from(sparse).into_elements(txn_id).await?;
                    async_hash::hash_try_stream::<Sha256, _, _, _>(elements).await?
                }
            },
        };

        let mut hasher = Sha256::new();
        hasher.update(schema_hash);
        hasher.update(contents_hash);
        Ok(hasher.finalize())
    }
}

impl<Txn, FE> From<CollectionBase<Txn, FE>> for Collection<Txn, FE> {
    fn from(base: CollectionBase<Txn, FE>) -> Self {
        match base {
            CollectionBase::BTree(btree) => Self::BTree(btree.into()),
            CollectionBase::Table(table) => Self::Table(table.into()),
            CollectionBase::Tensor(tensor) => Self::Tensor(tensor.into()),
        }
    }
}

impl<Txn, FE> From<BTreeFile<Txn, FE>> for Collection<Txn, FE> {
    fn from(btree: BTreeFile<Txn, FE>) -> Self {
        Self::BTree(btree.into())
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for Collection<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<btree::Node> + AsType<tensor::Node> + Clone,
{
    type Txn = Txn;
    type View = CollectionView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::BTree(btree) => btree.into_view(txn).map_ok(CollectionView::BTree).await,
            Self::Table(table) => table.into_view(txn).map_ok(CollectionView::Table).await,
            Self::Tensor(tensor) => tensor.into_view(txn).map_ok(CollectionView::Tensor).await,
        }
    }
}

#[async_trait]
impl<T, FE> de::FromStream for Collection<T, FE>
where
    T: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
{
    type Context = T;

    async fn from_stream<D: de::Decoder>(
        txn: Self::Context,
        decoder: &mut D,
    ) -> Result<Self, D::Error> {
        decoder
            .decode_map(CollectionVisitor::new(txn))
            .map_ok(Self::from)
            .await
    }
}

impl<Txn, FE> TryCastFrom<Collection<Txn, FE>> for BTree<Txn, FE> {
    fn can_cast_from(collection: &Collection<Txn, FE>) -> bool {
        match collection {
            Collection::BTree(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection<Txn, FE>) -> Option<Self> {
        match collection {
            Collection::BTree(btree) => Some(btree),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<Collection<Txn, FE>> for Table<Txn, FE> {
    fn can_cast_from(collection: &Collection<Txn, FE>) -> bool {
        match collection {
            Collection::Table(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection<Txn, FE>) -> Option<Self> {
        match collection {
            Collection::Table(table) => Some(table),
            _ => None,
        }
    }
}

impl<Txn, FE> TryCastFrom<Collection<Txn, FE>> for Tensor<Txn, FE> {
    fn can_cast_from(collection: &Collection<Txn, FE>) -> bool {
        match collection {
            Collection::Tensor(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection<Txn, FE>) -> Option<Self> {
        match collection {
            Collection::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }
}

impl<Txn, FE> fmt::Debug for Collection<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => btree.fmt(f),
            Self::Table(table) => table.fmt(f),
            Self::Tensor(tensor) => tensor.fmt(f),
        }
    }
}

/// A view of a [`Collection`] within a single `Transaction`, used for serialization.
pub enum CollectionView<'en> {
    BTree(btree::BTreeView<'en>),
    Table(table::TableView<'en>),
    Tensor(tensor::view::TensorView),
}

impl<'en> en::IntoStream<'en> for CollectionView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;

        match self {
            Self::BTree(btree) => {
                let classpath = BTreeType::default().path();
                map.encode_entry(classpath.to_string(), btree)?;
            }
            Self::Table(table) => {
                let classpath = TableType::default().path();
                map.encode_entry(classpath.to_string(), table)?;
            }
            Self::Tensor(tensor) => {
                let classpath = match tensor {
                    tensor::view::TensorView::Dense(_) => TensorType::Dense,
                    tensor::view::TensorView::Sparse(_) => TensorType::Sparse,
                }
                .path();

                map.encode_entry(classpath.to_string(), tensor)?;
            }
        }

        map.end()
    }
}

async fn finalize_dir<FE: Send + Sync>(dir: &freqfs::DirLock<FE>, txn_id: &TxnId) {
    let dir = dir.read().await;

    let versions = dir
        .get_dir(tc_transact::fs::VERSIONS)
        .expect("transactional versions directory");

    let mut versions = versions.write().await;
    versions.delete(txn_id);
}
