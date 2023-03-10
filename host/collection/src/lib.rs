use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;

use tc_error::*;
use tc_transact::{IntoView, Transaction};
use tcgeneric::{path_label, Class, NativeClass, PathLabel, PathSegment, TCPathBuf};

use btree::BTreeType;
use table::TableType;
use tensor::TensorType;

mod base;
mod schema;

pub mod btree;
pub mod table;
pub mod tensor;

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

impl From<TensorType> for CollectionType {
    fn from(tt: TensorType) -> Self {
        Self::Tensor(tt)
    }
}

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
pub struct Collection<T, FE> {
    phantom: PhantomData<(T, FE)>,
}

impl<T, FE> From<CollectionBase<T, FE>> for Collection<T, FE> {
    fn from(_base: CollectionBase<T, FE>) -> Self {
        todo!()
    }
}

#[async_trait]
impl<'en, T, FE> IntoView<'en, FE> for Collection<T, FE>
where
    T: Transaction<FE>,
    FE: Send + Sync,
    Self: 'en,
{
    type Txn = T;
    type View = CollectionView<'en, T, FE>;

    async fn into_view(self, _txn: Self::Txn) -> TCResult<Self::View> {
        todo!()
    }
}

#[async_trait]
impl<T, FE> de::FromStream for Collection<T, FE>
where
    T: Transaction<FE>,
    FE: Send + Sync,
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
pub struct CollectionView<'en, T, FE> {
    phantom: PhantomData<&'en (T, FE)>,
}

impl<'en, T, FE> en::IntoStream<'en> for CollectionView<'en, T, FE>
where
    Self: 'en,
{
    fn into_stream<E: en::Encoder<'en>>(self, _encoder: E) -> Result<E::Ok, E::Error> {
        todo!()
    }
}
