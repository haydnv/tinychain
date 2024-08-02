use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, en};
use freqfs::FileSave;
use futures::TryFutureExt;
#[cfg(feature = "btree")]
use safecast::{as_type, AsType, TryCastFrom};

use tc_error::*;
use tc_transact::fs;
#[cfg(feature = "btree")]
use tc_transact::hash::hash_try_stream;
use tc_transact::hash::{AsyncHash, Digest, Hash, Output, Sha256};
use tc_transact::IntoView;
use tc_transact::{Transaction, TxnId};
use tcgeneric::{
    label, path_label, Class, Instance, Label, NativeClass, PathLabel, PathSegment, TCPathBuf,
};

#[cfg(feature = "btree")]
use btree::{BTreeInstance, BTreeType};
#[cfg(feature = "table")]
use table::{TableInstance, TableStream, TableType};
#[cfg(feature = "tensor")]
use tensor::TensorType;

pub use base::{CollectionBase, CollectionVisitor};
#[cfg(feature = "btree")]
pub use btree::{BTree, BTreeFile, Node as BTreeNode};
pub use schema::Schema;
#[cfg(feature = "table")]
pub use table::{Table, TableFile};
#[cfg(feature = "tensor")]
pub use tensor::{
    Dense, DenseBase, DenseCacheFile, DenseView, Node as TensorNode, Sparse, SparseBase,
    SparseView, Tensor, TensorBase, TensorInstance, TensorView,
};

mod base;
mod schema;

#[cfg(feature = "btree")]
pub mod btree;
#[cfg(feature = "table")]
pub mod table;
#[cfg(feature = "tensor")]
pub mod tensor;

pub mod public;

/// The prefix of the absolute path to [`Collection`] data types
pub const PREFIX: PathLabel = path_label(&["state", "collection"]);

const NULL: Label = label("null");

/// A block in a [`Collection`]

#[cfg(all(feature = "btree", not(feature = "tensor")))]
pub trait CollectionBlock:
    AsType<BTreeNode> + tcgeneric::ThreadSafe + Clone + for<'a> FileSave<'a>
{
}

#[cfg(all(feature = "btree", not(feature = "tensor")))]
impl<T> CollectionBlock for T where
    T: AsType<BTreeNode> + tcgeneric::ThreadSafe + Clone + for<'a> FileSave<'a>
{
}

#[cfg(feature = "tensor")]
pub trait CollectionBlock:
    DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone + for<'a> FileSave<'a>
{
}

#[cfg(feature = "tensor")]
impl<T> CollectionBlock for T where
    T: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone + for<'a> FileSave<'a>
{
}

#[cfg(not(feature = "btree"))]
pub trait CollectionBlock: tcgeneric::ThreadSafe + Clone + for<'a> FileSave<'a> {}

#[cfg(not(feature = "btree"))]
impl<T> CollectionBlock for T where T: tcgeneric::ThreadSafe + Clone + for<'a> FileSave<'a> {}

/// The [`Class`] of a `Collection`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollectionType {
    Null,
    #[cfg(feature = "btree")]
    BTree(BTreeType),
    #[cfg(feature = "table")]
    Table(TableType),
    #[cfg(feature = "tensor")]
    Tensor(TensorType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() > 2 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                #[cfg(feature = "btree")]
                "btree" => BTreeType::from_path(path).map(Self::BTree),
                #[cfg(feature = "table")]
                "table" => TableType::from_path(path).map(Self::Table),
                #[cfg(feature = "tensor")]
                "tensor" => TensorType::from_path(path).map(Self::Tensor),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            Self::Null => TCPathBuf::from(NULL),
            #[cfg(feature = "btree")]
            Self::BTree(btt) => btt.path(),
            #[cfg(feature = "table")]
            Self::Table(tt) => tt.path(),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => tt.path(),
        }
    }
}

#[cfg(feature = "btree")]
as_type!(CollectionType, BTree, BTreeType);
#[cfg(feature = "table")]
as_type!(CollectionType, Table, TableType);
#[cfg(feature = "tensor")]
as_type!(CollectionType, Tensor, TensorType);

impl fmt::Debug for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Null => f.write_str("null collection"),
            #[cfg(feature = "btree")]
            Self::BTree(btt) => fmt::Debug::fmt(btt, f),
            #[cfg(feature = "table")]
            Self::Table(tt) => fmt::Debug::fmt(tt, f),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => fmt::Debug::fmt(tt, f),
        }
    }
}

/// A mutable transactional collection of data.
pub enum Collection<Txn, FE> {
    Null(fs::Dir<FE>, PhantomData<Txn>),
    #[cfg(feature = "btree")]
    BTree(BTree<Txn, FE>),
    #[cfg(feature = "table")]
    Table(Table<Txn, FE>),
    #[cfg(feature = "tensor")]
    Tensor(Tensor<Txn, FE>),
}

impl<Txn, FE> Clone for Collection<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Null(dir, data) => Self::Null(dir.clone(), *data),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => Self::BTree(btree.clone()),
            #[cfg(feature = "table")]
            Self::Table(table) => Self::Table(table.clone()),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
        }
    }
}

#[cfg(feature = "btree")]
as_type!(Collection<Txn, FE>, BTree, BTree<Txn, FE>);
#[cfg(feature = "table")]
as_type!(Collection<Txn, FE>, Table, Table<Txn, FE>);
#[cfg(feature = "tensor")]
as_type!(Collection<Txn, FE>, Tensor, Tensor<Txn, FE>);

impl<Txn, FE> Collection<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: CollectionBlock,
{
    /// Return the [`Schema`] of this [`Collection`].
    pub fn schema(&self) -> Schema {
        match self {
            Self::Null(_, _) => Schema::Null,
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.schema().clone().into(),
            #[cfg(feature = "table")]
            Self::Table(table) => table.schema().clone().into(),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => Schema::Dense(dense.schema()),
                Tensor::Sparse(sparse) => Schema::Sparse(sparse.schema()),
            },
        }
    }
}

impl<Txn, FE> Instance for Collection<Txn, FE>
where
    Txn: Send + Sync,
    FE: Send + Sync,
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
impl<Txn, FE> AsyncHash for Collection<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: CollectionBlock + Clone,
{
    #[allow(unused_variables)]
    async fn hash(&self, txn_id: TxnId) -> TCResult<Output<Sha256>> {
        let schema_hash = Hash::<Sha256>::hash(self.schema());

        let contents_hash = match self {
            Self::Null(_, _) => tc_transact::hash::default_hash::<Sha256>(),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => {
                let keys = btree.clone().keys(txn_id).await?;
                hash_try_stream::<Sha256, _, _, _>(keys).await?
            }
            #[cfg(feature = "table")]
            Self::Table(table) => {
                let rows = table.clone().rows(txn_id).await?;
                hash_try_stream::<Sha256, _, _, _>(rows).await?
            }
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => match tensor {
                Tensor::Dense(dense) => {
                    let elements = DenseView::from(dense.clone()).into_elements(txn_id).await?;
                    hash_try_stream::<Sha256, _, _, _>(elements).await?
                }
                Tensor::Sparse(sparse) => {
                    let elements = SparseView::from(sparse.clone())
                        .into_elements(txn_id)
                        .await?;

                    hash_try_stream::<Sha256, _, _, _>(elements).await?
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
            CollectionBase::Null(dir, data) => Self::Null(dir, data),
            #[cfg(feature = "btree")]
            CollectionBase::BTree(btree) => Self::BTree(btree.into()),
            #[cfg(feature = "table")]
            CollectionBase::Table(table) => Self::Table(table.into()),
            #[cfg(feature = "tensor")]
            CollectionBase::Tensor(tensor) => Self::Tensor(tensor.into()),
        }
    }
}

#[cfg(feature = "btree")]
impl<Txn, FE> From<BTreeFile<Txn, FE>> for Collection<Txn, FE> {
    fn from(btree: BTreeFile<Txn, FE>) -> Self {
        Self::BTree(btree.into())
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for Collection<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: CollectionBlock,
{
    type Txn = Txn;
    type View = CollectionView<'en>;

    #[allow(unused_variables)]
    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::Null(_dir, data) => Ok(CollectionView::Null(PhantomData)),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.into_view(txn).map_ok(CollectionView::BTree).await,
            #[cfg(feature = "table")]
            Self::Table(table) => table.into_view(txn).map_ok(CollectionView::Table).await,
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.into_view(txn).map_ok(CollectionView::Tensor).await,
        }
    }
}

#[async_trait]
impl<T, FE> de::FromStream for Collection<T, FE>
where
    T: Transaction<FE>,
    FE: CollectionBlock,
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
            #[cfg(feature = "btree")]
            Collection::BTree(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(collection: Collection<Txn, FE>) -> Option<Self> {
        match collection {
            #[cfg(feature = "btree")]
            Collection::BTree(btree) => Some(btree),
            _ => None,
        }
    }
}

#[cfg(feature = "table")]
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

#[cfg(feature = "tensor")]
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
            Self::Null(_, _) => f.write_str("null collection"),
            #[cfg(feature = "btree")]
            Self::BTree(btree) => btree.fmt(f),
            #[cfg(feature = "table")]
            Self::Table(table) => table.fmt(f),
            #[cfg(feature = "tensor")]
            Self::Tensor(tensor) => tensor.fmt(f),
        }
    }
}

/// A view of a [`Collection`] within a single `Transaction`, used for serialization.
pub enum CollectionView<'en> {
    Null(PhantomData<&'en ()>),
    #[cfg(feature = "btree")]
    BTree(btree::BTreeView<'en>),
    #[cfg(feature = "table")]
    Table(table::TableView<'en>),
    #[cfg(feature = "tensor")]
    Tensor(tensor::view::TensorView),
}

impl<'en> en::IntoStream<'en> for CollectionView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;

        let mut map = encoder.encode_map(Some(1))?;

        match self {
            Self::Null(_) => map.encode_entry(CollectionType::Null.path(), ())?,
            #[cfg(feature = "btree")]
            Self::BTree(btree) => {
                let classpath = BTreeType::default().path();
                map.encode_entry(classpath.to_string(), btree)?;
            }
            #[cfg(feature = "table")]
            Self::Table(table) => {
                let classpath = TableType::default().path();
                map.encode_entry(classpath.to_string(), table)?;
            }
            #[cfg(feature = "tensor")]
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

#[cfg(feature = "btree")]
async fn finalize_dir<FE: Send + Sync>(dir: &freqfs::DirLock<FE>, txn_id: &TxnId) {
    let dir = dir.read().await;

    let versions = dir
        .get_dir(tc_transact::fs::VERSIONS)
        .expect("transactional versions directory");

    let mut versions = versions.write().await;
    versions.delete(txn_id).await;
}
