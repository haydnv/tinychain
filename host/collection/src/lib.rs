use std::fmt;
use std::marker::PhantomData;

use tcgeneric::{path_label, Class, NativeClass, PathLabel, PathSegment, TCPathBuf};

use btree::BTreeType;
use table::TableType;
use tensor::TensorType;

mod base;
mod schema;

pub mod btree;
pub mod table;
pub mod tensor;

pub use base::CollectionBase;
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
