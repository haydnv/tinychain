use std::convert::TryFrom;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Class, Instance, TCResult, TCType};
use crate::error;
use crate::transaction::{Transact, Txn};
use crate::value::link::{Link, TCPath};
use crate::value::{label, Value};

use super::btree::BTreeType;
use super::{Collection, CollectionBase, CollectionView};

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
pub trait CollectionBaseClass: CollectionClass + Into<CollectionBaseType> + Send + Sync {
    type Instance: CollectionBaseInstance;

    fn get(
        txn: Arc<Txn>,
        path: &TCPath,
        schema: Value,
    ) -> TCResult<<Self as CollectionClass>::Instance>;
}

pub trait CollectionViewClass: CollectionClass + Into<CollectionViewType> + Send + Sync {
    type Instance: CollectionViewInstance;
}

#[async_trait]
pub trait CollectionInstance: Instance + Into<Collection> + Transact + Send + Sync {
    type Selector: Clone + TryFrom<Value, Error = error::TCError> + Send + Sync + 'static;

    type Item: Clone + Into<Value> + TryFrom<Value, Error = error::TCError> + Send + Sync + 'static;

    type Slice: CollectionViewInstance;

    async fn get(&self, txn: Arc<Txn>, selector: Self::Selector) -> TCResult<Self::Slice>;

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        value: Self::Item,
    ) -> TCResult<()>;
}

#[async_trait]
pub trait CollectionBaseInstance: CollectionInstance {
    type Schema: TryFrom<Value, Error = error::TCError>;

    async fn create(txn: Arc<Txn>, schema: Self::Schema) -> TCResult<Self>;
}

pub trait CollectionViewInstance: CollectionInstance + Into<CollectionView> {}

#[derive(Clone, Eq, PartialEq)]
pub enum CollectionType {
    Base(CollectionBaseType),
    View(CollectionViewType),
}

impl Class for CollectionType {
    type Instance = Collection;

    fn prefix() -> TCPath {
        TCType::prefix().join(label("collection").into())
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
    Table,
    Tensor,
}

impl Class for CollectionBaseType {
    type Instance = CollectionBase;

    fn prefix() -> TCPath {
        CollectionType::prefix()
    }
}

impl From<CollectionBaseType> for Link {
    fn from(ct: CollectionBaseType) -> Link {
        let prefix = CollectionBaseType::prefix();

        use CollectionBaseType::*;
        match ct {
            BTree => BTreeType::Tree.into(),
            Graph => prefix.join(label("graph").into()).into(), // TODO
            Table => prefix.join(label("table").into()).into(), // TODO
            Tensor => prefix.join(label("tensor").into()).into(), // TODO
        }
    }
}

impl fmt::Display for CollectionBaseType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CollectionBaseType::*;
        match self {
            BTree => write!(f, "class BTree"),
            Graph => write!(f, "class Graph"),
            Table => write!(f, "class Table"),
            Tensor => write!(f, "class Tensor"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum CollectionViewType {
    BTree(BTreeType),
    Graph,
    Table,
    Tensor,
}

impl Class for CollectionViewType {
    type Instance = CollectionView;

    fn prefix() -> TCPath {
        CollectionType::prefix()
    }
}

impl From<BTreeType> for CollectionViewType {
    fn from(btt: BTreeType) -> CollectionViewType {
        CollectionViewType::BTree(btt)
    }
}

impl From<CollectionViewType> for Link {
    fn from(cvt: CollectionViewType) -> Link {
        let prefix = CollectionViewType::prefix();

        use CollectionViewType::*;
        match cvt {
            BTree(btt) => btt.into(),
            Graph => prefix.join(label("graph").into()).into(), // TODO
            Table => prefix.join(label("table").into()).into(), // TODO
            Tensor => prefix.join(label("tensor").into()).into(), // TODO
        }
    }
}

impl fmt::Display for CollectionViewType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use CollectionViewType::*;
        match self {
            BTree(btree_type) => write!(f, "{}", btree_type),
            Graph => write!(f, "class Graph"),       // TODO
            Table => write!(f, "class TableView"),   // TODO
            Tensor => write!(f, "class TensorView"), // TODO
        }
    }
}
