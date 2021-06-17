//! A [`Collection`] such as a [`BTree`] or [`Table`].

/// The `Collection` enum used in `State::Collection`.
use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use log::debug;

use tc_btree::BTreeView;
use tc_error::*;
use tc_table::TableView;
#[cfg(feature = "tensor")]
use tc_tensor::TensorView;
use tc_transact::fs::Dir;
use tc_transact::{IntoView, Transaction};
use tcgeneric::{
    path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPath, TCPathBuf,
};

use crate::fs;
use crate::txn::Txn;

pub use tc_btree::BTreeType;
pub use tc_table::TableType;

#[cfg(feature = "tensor")]
pub use tc_tensor::{DenseAccess, TensorType};

pub type BTree = tc_btree::BTree<fs::File<tc_btree::Node>, fs::Dir, Txn>;
pub type BTreeFile = tc_btree::BTreeFile<fs::File<tc_btree::Node>, fs::Dir, Txn>;

pub type Table = tc_table::Table<fs::File<tc_btree::Node>, fs::Dir, Txn>;
pub type TableIndex = tc_table::TableIndex<fs::File<tc_btree::Node>, fs::Dir, Txn>;

#[cfg(feature = "tensor")]
pub type Tensor =
    tc_tensor::Tensor<fs::File<afarray::Array>, fs::File<tc_btree::Node>, fs::Dir, Txn>;
#[cfg(feature = "tensor")]
pub type DenseTensor<B> =
    tc_tensor::DenseTensor<fs::File<afarray::Array>, fs::File<tc_btree::Node>, fs::Dir, Txn, B>;
#[cfg(feature = "tensor")]
pub type DenseTensorFile =
    tc_tensor::BlockListFile<fs::File<afarray::Array>, fs::File<tc_btree::Node>, fs::Dir, Txn>;
#[cfg(feature = "tensor")]
pub type SparseTensor<A> =
    tc_tensor::SparseTensor<fs::File<afarray::Array>, fs::File<tc_btree::Node>, fs::Dir, Txn, A>;
#[cfg(feature = "tensor")]
pub type SparseTable =
    tc_tensor::SparseTable<fs::File<afarray::Array>, fs::File<tc_btree::Node>, fs::Dir, Txn>;

pub const PREFIX: PathLabel = path_label(&["state", "collection"]);

/// The [`Class`] of a [`Collection`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollectionType {
    BTree(BTreeType),
    Table(TableType),
    #[cfg(feature = "tensor")]
    Tensor(TensorType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        debug!("CollectionType::from_path {}", TCPath::from(path));

        if path.len() > 2 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                "btree" => BTreeType::from_path(path).map(Self::BTree),
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
            Self::BTree(btt) => btt.path(),
            Self::Table(tt) => tt.path(),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => tt.path(),
        }
    }
}

impl From<BTreeType> for CollectionType {
    fn from(btt: BTreeType) -> Self {
        Self::BTree(btt)
    }
}

impl From<TableType> for CollectionType {
    fn from(tt: TableType) -> Self {
        Self::Table(tt)
    }
}

#[cfg(feature = "tensor")]
impl From<TensorType> for CollectionType {
    fn from(tt: TensorType) -> Self {
        Self::Tensor(tt)
    }
}

impl fmt::Display for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btt) => fmt::Display::fmt(btt, f),
            Self::Table(tt) => fmt::Display::fmt(tt, f),
            #[cfg(feature = "tensor")]
            Self::Tensor(tt) => fmt::Display::fmt(tt, f),
        }
    }
}

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
impl<B: DenseAccess<fs::File<afarray::Array>, fs::File<tc_btree::Node>, fs::Dir, Txn>>
    From<DenseTensor<B>> for Collection
{
    fn from(tensor: DenseTensor<B>) -> Self {
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
    ) -> Result<Collection, A::Error> {
        debug!("deserialize Collection");

        match class {
            CollectionType::BTree(_) => {
                let file = self
                    .txn
                    .context()
                    .create_file_tmp(*self.txn.id(), BTreeType::default())
                    .map_err(de::Error::custom)
                    .await?;

                access
                    .next_value((self.txn.clone(), file))
                    .map_ok(Collection::BTree)
                    .await
            }

            CollectionType::Table(_) => access.next_value(self.txn).map_ok(Collection::Table).await,

            #[cfg(feature = "tensor")]
            CollectionType::Tensor(tt) => match tt {
                TensorType::Dense => {
                    let tensor: DenseTensor<DenseTensorFile> = access.next_value(self.txn).await?;
                    Ok(Collection::Tensor(tensor.into()))
                }
                TensorType::Sparse => {
                    let tensor: SparseTensor<SparseTable> = access.next_value(self.txn).await?;
                    Ok(Collection::Tensor(tensor.into()))
                }
            },
        }
    }
}

#[async_trait]
impl de::Visitor for CollectionVisitor {
    type Value = Collection;

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
impl de::FromStream for Collection {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(CollectionVisitor { txn }).await
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
