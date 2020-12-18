use std::fmt;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::class::{Class, NativeClass, TCType};
use crate::error;
use crate::general::{CastInto, TCResult, TCStreamOld, TryCastFrom};
use crate::scalar::{label, Link, PathSegment, Scalar, TCPathBuf, Value};
use crate::transaction::Txn;

use super::btree::BTreeType;
use super::table::TableType;
use super::tensor::TensorType;
use super::Collection;

#[async_trait]
pub trait CollectionClass: Class + Into<CollectionType> + Send {
    type Instance;

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<<Self as CollectionClass>::Instance>;
}

#[async_trait]
pub trait CollectionInstance {
    type Item: CastInto<Scalar> + TryCastFrom<Scalar>;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool>;

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStreamOld<Scalar>>;
}

#[derive(Clone, Eq, PartialEq)]
pub enum CollectionType {
    BTree(BTreeType),
    Table(TableType),
    Tensor(TensorType),
}

impl Class for CollectionType {
    type Instance = Collection;
}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::unsupported("You must specify a type of Collection"))
        } else {
            use CollectionType::*;
            match suffix[0].as_str() {
                "btree" if suffix.len() == 1 => Ok(Self::BTree(BTreeType::Tree)),
                "table" => TableType::from_path(path).map(Table),
                "tensor" => TensorType::from_path(path).map(Tensor),
                _ => Err(error::path_not_found(suffix)),
            }
        }
    }

    fn prefix() -> TCPathBuf {
        TCType::prefix().append(label("collection"))
    }
}

#[async_trait]
impl CollectionClass for CollectionType {
    type Instance = Collection;

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<<Self as CollectionClass>::Instance> {
        match self {
            Self::BTree(btt) if btt == &BTreeType::Tree => {
                btt.get(txn, schema).map_ok(Collection::BTree).await
            }
            Self::Table(tt) if tt == &TableType::Table => {
                tt.get(txn, schema).map_ok(Collection::Table).await
            }
            Self::Tensor(tt) => tt.get(txn, schema).map_ok(Collection::Tensor).await,
            other => Err(error::bad_request(
                "Cannot instantiate a Collection view directly",
                other,
            )),
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

impl From<CollectionType> for Link {
    fn from(ct: CollectionType) -> Link {
        match ct {
            CollectionType::BTree(btt) => btt.into(),
            CollectionType::Table(tt) => tt.into(),
            CollectionType::Tensor(tt) => tt.into(),
        }
    }
}

impl fmt::Display for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btt) => fmt::Display::fmt(btt, f),
            Self::Table(tt) => fmt::Display::fmt(tt, f),
            Self::Tensor(tt) => fmt::Display::fmt(tt, f),
        }
    }
}
