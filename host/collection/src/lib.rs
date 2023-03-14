use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;
use safecast::{as_type, AsType};

use tc_error::*;
use tc_transact::{AsyncHash, IntoView, Sha256, Transaction};
use tcgeneric::{
    path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf, ThreadSafe,
};

use btree::{BTreeInstance, BTreeType};

pub use btree::{BTree, Node};
use table::TableType;
use tensor::TensorType;

mod base;
mod schema;

pub mod btree;
pub mod table;
pub mod tensor;

use crate::btree::BTreeFile;
pub use base::{CollectionBase, CollectionVisitor};
pub use schema::Schema;

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

#[derive(Clone)]
pub enum Collection<Txn, FE> {
    BTree(BTree<Txn, FE>),
}

as_type!(Collection<Txn, FE>, BTree, BTree<Txn, FE>);

impl<Txn, FE> Collection<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    fn schema(&self) -> Schema {
        match self {
            Self::BTree(btree) => btree.schema().clone().into(),
        }
    }
}

impl<T, FE> Instance for Collection<T, FE>
where
    T: Transaction<FE>,
    FE: Send + Sync,
{
    type Class = CollectionType;

    fn class(&self) -> CollectionType {
        match self {
            Self::BTree(btree) => btree.class().into(),
        }
    }
}

#[async_trait]
impl<T, FE> AsyncHash<FE> for Collection<T, FE>
where
    T: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Txn = T;

    async fn hash(self, txn: &Self::Txn) -> TCResult<Output<Sha256>> {
        let schema_hash = Hash::<Sha256>::hash(self.schema());

        let contents_hash = match self {
            Self::BTree(btree) => {
                let keys = btree.keys(*txn.id()).await?;
                async_hash::hash_try_stream::<Sha256, _, _, _>(keys).await?
            }
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
        }
    }
}

impl<Txn, FE> From<BTreeFile<Txn, FE>> for Collection<Txn, FE> {
    fn from(btree: BTreeFile<Txn, FE>) -> Self {
        Self::BTree(btree.into())
    }
}

#[async_trait]
impl<'en, T, FE> IntoView<'en, FE> for Collection<T, FE>
where
    T: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    Self: 'en,
{
    type Txn = T;
    type View = CollectionView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::BTree(btree) => btree.into_view(txn).map_ok(CollectionView::BTree).await,
        }
    }
}

#[async_trait]
impl<T, FE> de::FromStream for Collection<T, FE>
where
    T: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
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

impl<T, FE> fmt::Debug for Collection<T, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Collection")
    }
}

/// A view of a [`Collection`] within a single `Transaction`, used for serialization.
pub enum CollectionView<'en> {
    BTree(btree::BTreeView<'en>),
}

impl<'en> en::IntoStream<'en> for CollectionView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::BTree(btree) => btree.into_stream(encoder),
        }
    }
}
