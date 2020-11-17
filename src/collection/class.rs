use std::fmt;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::class::{Class, Instance, NativeClass, State, TCResult, TCStream, TCType};
use crate::error;
use crate::request::Request;
use crate::scalar::{
    label, CastInto, Link, Object, PathSegment, Scalar, TCPathBuf, TryCastFrom, TryCastInto, Value,
};
use crate::transaction::{Transact, Txn};

use super::btree::{BTreeFile, BTreeType};
use super::null::{Null, NullType};
use super::table::{TableBaseType, TableType};
use super::tensor::{TensorBaseType, TensorType};
use super::{Collection, CollectionBase, CollectionView};

#[async_trait]
pub trait CollectionClass: Class + Into<CollectionType> + Send {
    type Instance: CollectionInstance;

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<<Self as CollectionClass>::Instance>;
}

#[async_trait]
pub trait CollectionInstance: Instance + Into<Collection> + Transact + Send {
    type Item: CastInto<Scalar> + TryCastFrom<Scalar>;
    type Slice;

    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State>;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool>;

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State>;

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()>;

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>>;
}

#[derive(Clone, Eq, PartialEq)]
pub enum CollectionType {
    Base(CollectionBaseType),
    View(CollectionViewType),
}

impl Class for CollectionType {
    type Instance = Collection;
}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        CollectionBaseType::from_path(path).map(CollectionType::Base)
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
            Self::Base(cbt) => cbt.get(txn, schema).map_ok(Collection::Base).await,
            Self::View(_) => Err(error::unsupported(
                "Cannot instantiate a CollectionView directly",
            )),
        }
    }
}

impl From<CollectionBaseType> for CollectionType {
    fn from(cbt: CollectionBaseType) -> CollectionType {
        CollectionType::Base(cbt)
    }
}

impl From<CollectionViewType> for CollectionType {
    fn from(cvt: CollectionViewType) -> CollectionType {
        CollectionType::View(cvt)
    }
}

impl From<CollectionType> for Link {
    fn from(ct: CollectionType) -> Link {
        use CollectionType::*;
        match ct {
            Base(base) => base.into(),
            View(view) => view.into(),
        }
    }
}

impl fmt::Display for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CollectionType::Base(base) => write!(f, "{}", base),
            CollectionType::View(view) => write!(f, "{}", view),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum CollectionBaseType {
    BTree,
    Null,
    Table(TableBaseType),
    Tensor(TensorBaseType),
}

impl Class for CollectionBaseType {
    type Instance = CollectionBase;
}

impl NativeClass for CollectionBaseType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.is_empty() {
            Err(error::unsupported("You must specify a type of Collection"))
        } else {
            use CollectionBaseType::*;
            match suffix[0].as_str() {
                "btree" if suffix.len() == 1 => Ok(BTree),
                "null" if suffix.len() == 1 => Ok(Null),
                "table" => TableBaseType::from_path(path).map(Table),
                "tensor" => TensorBaseType::from_path(path).map(Tensor),
                _ => Err(error::path_not_found(suffix)),
            }
        }
    }

    fn prefix() -> TCPathBuf {
        CollectionType::prefix()
    }
}

#[async_trait]
impl CollectionClass for CollectionBaseType {
    type Instance = CollectionBase;

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<CollectionBase> {
        match self {
            Self::BTree => {
                let schema = schema
                    .try_cast_into(|s| error::bad_request("Expected BTree schema but found", s))?;

                BTreeFile::create(txn, schema)
                    .map_ok(CollectionBase::BTree)
                    .await
            }
            Self::Null => Ok(CollectionBase::Null(Null::create())),
            Self::Table(tt) => tt.get(txn, schema).map_ok(CollectionBase::Table).await,
            Self::Tensor(tt) => tt.get(txn, schema).map_ok(CollectionBase::Tensor).await,
        }
    }
}

impl From<CollectionBaseType> for Link {
    fn from(ct: CollectionBaseType) -> Link {
        use CollectionBaseType::*;
        match ct {
            BTree => BTreeType::Tree.into(),
            Null => CollectionBaseType::prefix().append(label("null")).into(),
            Table(tbt) => tbt.into(),
            Tensor(tbt) => tbt.into(),
        }
    }
}

impl fmt::Display for CollectionBaseType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CollectionBaseType::*;
        match self {
            BTree => write!(f, "{}", BTreeType::Tree),
            Null => write!(f, "{}", NullType),
            Table(tbt) => write!(f, "{}", tbt),
            Tensor(tbt) => write!(f, "{}", tbt),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum CollectionViewType {
    BTree(BTreeType),
    Null(NullType),
    Table(TableType),
    Tensor(TensorType),
}

impl Class for CollectionViewType {
    type Instance = CollectionView;
}

impl NativeClass for CollectionViewType {
    fn from_path(_path: &[PathSegment]) -> TCResult<Self> {
        Err(error::internal(crate::class::ERR_PROTECTED))
    }

    fn prefix() -> TCPathBuf {
        CollectionType::prefix()
    }
}

impl From<BTreeType> for CollectionViewType {
    fn from(btt: BTreeType) -> CollectionViewType {
        Self::BTree(btt)
    }
}

impl From<NullType> for CollectionViewType {
    fn from(nt: NullType) -> CollectionViewType {
        Self::Null(nt)
    }
}

impl From<TableType> for CollectionViewType {
    fn from(tt: TableType) -> CollectionViewType {
        Self::Table(tt)
    }
}

impl From<TensorType> for CollectionViewType {
    fn from(tt: TensorType) -> CollectionViewType {
        Self::Tensor(tt)
    }
}

impl From<CollectionViewType> for Link {
    fn from(cvt: CollectionViewType) -> Link {
        use CollectionViewType::*;
        match cvt {
            BTree(btt) => btt.into(),
            Null(nt) => nt.into(),
            Table(tt) => tt.into(),
            Tensor(tt) => tt.into(),
        }
    }
}

impl fmt::Display for CollectionViewType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CollectionViewType::*;
        match self {
            BTree(btree_type) => write!(f, "{}", btree_type),
            Null(null_type) => write!(f, "{}", null_type),
            Table(table_type) => write!(f, "{}", table_type),
            Tensor(tensor_type) => write!(f, "{}", tensor_type),
        }
    }
}
