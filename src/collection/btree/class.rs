use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Class, Instance, TCResult, TCType};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionView};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::{Link, TCPath};
use crate::value::{label, Value};

use super::{BTreeFile, BTreeSlice};

#[derive(Clone, Eq, PartialEq)]
pub enum BTreeType {
    Tree,
    Slice,
}

impl Class for BTreeType {
    type Instance = BTree;

    fn from_path(path: &TCPath) -> TCResult<TCType> {
        if path.is_empty() {
            Ok(TCType::Collection(CollectionType::Base(
                CollectionBaseType::BTree,
            )))
        } else {
            Err(error::not_found(path))
        }
    }

    fn prefix() -> TCPath {
        CollectionType::prefix().join(label("btree").into())
    }
}

#[async_trait]
impl CollectionClass for BTreeType {
    type Instance = BTree;

    async fn get(_txn: Arc<Txn>, path: &TCPath, _schema: Value) -> TCResult<BTree> {
        if path.is_empty() {
            Err(error::not_implemented())
        } else {
            Err(error::not_found(path))
        }
    }
}

impl From<BTreeType> for CollectionType {
    fn from(btree_type: BTreeType) -> CollectionType {
        CollectionType::View(CollectionViewType::BTree(btree_type))
    }
}

impl From<BTreeType> for Link {
    fn from(btt: BTreeType) -> Link {
        let prefix = BTreeType::prefix();

        use BTreeType::*;
        match btt {
            Tree => prefix.into(),
            Slice => prefix.join(label("slice").into()).into(),
        }
    }
}

impl fmt::Display for BTreeType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Tree => write!(f, "class BTree"),
            Self::Slice => write!(f, "class BTreeSlice"),
        }
    }
}

#[derive(Clone)]
pub enum BTree {
    Tree(Arc<BTreeFile>),
    Slice(BTreeSlice),
}

impl Instance for BTree {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        match self {
            Self::Tree(tree) => tree.class(),
            Self::Slice(slice) => slice.class(),
        }
    }
}

#[async_trait]
impl CollectionInstance for BTree {
    type Slice = BTreeSlice;

    async fn get(&self, txn: Arc<Txn>, selector: Value) -> TCResult<BTreeSlice> {
        match self {
            Self::Tree(tree) => tree.get(txn, selector).await,
            Self::Slice(slice) => slice.get(txn, selector).await,
        }
    }

    async fn put(&self, txn: Arc<Txn>, selector: Value, value: Value) -> TCResult<()> {
        match self {
            Self::Tree(tree) => tree.put(txn, selector, value).await,
            Self::Slice(slice) => slice.put(txn, selector, value).await,
        }
    }
}

#[async_trait]
impl Transact for BTree {
    async fn commit(&self, txn_id: &TxnId) {
        let no_op = ();

        match self {
            Self::Tree(tree) => tree.commit(txn_id).await,
            Self::Slice(_) => no_op,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let no_op = ();

        match self {
            Self::Tree(tree) => tree.rollback(txn_id).await,
            Self::Slice(_) => no_op,
        }
    }
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Collection {
        Collection::View(CollectionView::BTree(btree))
    }
}
