use std::convert::TryInto;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::class::{Class, Instance, State, TCResult, TCStream, TCType};
use crate::error;
use crate::transaction::{Transact, Txn};
use crate::value::link::{Link, TCPath};
use crate::value::{label, Value};

use super::btree::{BTreeFile, BTreeType};
use super::graph::{Graph, GraphType};
use super::null::{Null, NullType};
use super::table::{TableBaseType, TableType};
use super::tensor::{TensorBaseType, TensorType};
use super::{Collection, CollectionBase, CollectionView};

pub enum CollectionItem<I: Into<Value>, S: CollectionInstance> {
    Value(I),
    Slice(S),
}

impl<I: Into<Value>, S: CollectionInstance> From<CollectionItem<I, S>> for State {
    fn from(ci: CollectionItem<I, S>) -> State {
        match ci {
            CollectionItem::Value(v) => State::Value(v.into()),
            CollectionItem::Slice(s) => State::Collection(s.into()),
        }
    }
}

#[async_trait]
pub trait CollectionClass: Class + Into<CollectionType> + Send + Sync {
    type Instance: CollectionInstance;

    async fn get(
        txn: Arc<Txn>,
        path: &TCPath,
        schema: Value,
    ) -> TCResult<<Self as CollectionClass>::Instance>;
}

#[async_trait]
pub trait CollectionInstance: Instance + Into<Collection> + Transact + Send + Sync {
    type Item: Into<Value>;
    type Slice: CollectionInstance;

    async fn get_item(
        &self,
        txn: Arc<Txn>,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>>;

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool>;

    async fn put_item(
        &self,
        txn: Arc<Txn>,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()>;

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Value>>;
}

#[derive(Clone, Eq, PartialEq)]
pub enum CollectionType {
    Base(CollectionBaseType),
    View(CollectionViewType),
}

impl Class for CollectionType {
    type Instance = Collection;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        CollectionBaseType::from_path(path).map(CollectionType::Base)
    }

    fn prefix() -> TCPath {
        TCType::prefix().join(label("collection").into())
    }
}

#[async_trait]
impl CollectionClass for CollectionType {
    type Instance = Collection;

    async fn get(
        txn: Arc<Txn>,
        path: &TCPath,
        schema: Value,
    ) -> TCResult<<Self as CollectionClass>::Instance> {
        CollectionBaseType::get(txn, path, schema)
            .await
            .map(Collection::Base)
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
    Graph,
    Null,
    Table(TableBaseType),
    Tensor(TensorBaseType),
}

impl Class for CollectionBaseType {
    type Instance = CollectionBase;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            Err(error::unsupported("You must specify a type of Collection"))
        } else {
            use CollectionBaseType::*;
            match suffix[0].as_str() {
                "btree" if suffix.len() == 1 => Ok(BTree),
                "graph" if suffix.len() == 1 => Ok(Graph),
                "null" if suffix.len() == 1 => Ok(Null),
                "table" => TableBaseType::from_path(path).map(Table),
                "tensor" => TensorBaseType::from_path(path).map(Tensor),
                other => Err(error::not_found(other)),
            }
        }
    }

    fn prefix() -> TCPath {
        CollectionType::prefix()
    }
}

#[async_trait]
impl CollectionClass for CollectionBaseType {
    type Instance = CollectionBase;

    async fn get(txn: Arc<Txn>, path: &TCPath, schema: Value) -> TCResult<CollectionBase> {
        let suffix = path.from_path(&Self::prefix())?;

        if suffix.is_empty() {
            return Err(error::unsupported("You must specify a type of Collection"));
        }

        match suffix[0].as_str() {
            "btree" if suffix.len() == 1 => {
                BTreeFile::create(txn, schema.try_into()?)
                    .map_ok(CollectionBase::BTree)
                    .await
            }
            "graph" if suffix.len() == 1 => {
                Graph::create(txn, schema.try_into()?)
                    .map_ok(CollectionBase::Graph)
                    .await
            }
            "null" if suffix.len() == 1 => {
                if schema != Value::None {
                    Err(error::bad_request("Null Collection has no schema, found", schema))
                } else {
                    Ok(CollectionBase::Null(Null::create()))
                }
            }
            "table" => {
                TableBaseType::get(txn, path, schema)
                    .map_ok(CollectionBase::Table)
                    .await
            }
            "tensor" => {
                TensorBaseType::get(txn, path, schema)
                    .map_ok(CollectionBase::Tensor)
                    .await
            }
            other => Err(error::not_found(other)),
        }
    }
}

impl From<CollectionBaseType> for Link {
    fn from(ct: CollectionBaseType) -> Link {
        let prefix = CollectionBaseType::prefix();

        use CollectionBaseType::*;
        match ct {
            BTree => BTreeType::Tree.into(),
            Graph => prefix.join(label("graph").into()).into(),
            Null => prefix.join(label("null").into()).into(),
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
            Graph => write!(f, "{}", GraphType::Graph),
            Null => write!(f, "{}", NullType),
            Table(tbt) => write!(f, "{}", tbt),
            Tensor(tbt) => write!(f, "{}", tbt),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum CollectionViewType {
    BTree(BTreeType),
    Graph(GraphType),
    Null(NullType),
    Table(TableType),
    Tensor(TensorType),
}

impl Class for CollectionViewType {
    type Instance = CollectionView;

    fn from_path(_path: &TCPath) -> TCResult<Self> {
        Err(error::internal(crate::class::ERR_PROTECTED))
    }

    fn prefix() -> TCPath {
        CollectionType::prefix()
    }
}

impl From<BTreeType> for CollectionViewType {
    fn from(btt: BTreeType) -> CollectionViewType {
        Self::BTree(btt)
    }
}

impl From<GraphType> for CollectionViewType {
    fn from(gt: GraphType) -> CollectionViewType {
        Self::Graph(gt)
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
            Graph(gt) => gt.into(),
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
            Graph(graph_type) => write!(f, "{}", graph_type),
            Null(null_type) => write!(f, "{}", null_type),
            Table(table_type) => write!(f, "{}", table_type),
            Tensor(tensor_type) => write!(f, "{}", tensor_type),
        }
    }
}
