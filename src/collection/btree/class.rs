use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Class, Instance, TCResult};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionView};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::TCPath;
use crate::value::Value;

use super::{BTreeFile, BTreeSlice, Key, Selector};

#[derive(Clone, Eq, PartialEq)]
pub enum BTreeType {
    Tree,
    Slice,
}

impl Class for BTreeType {
    type Instance = BTree;
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
    type Selector = Selector;
    type Item = Key;
    type Slice = BTreeSlice;

    async fn get(&self, txn: Arc<Txn>, selector: Selector) -> TCResult<BTreeSlice> {
        match self {
            Self::Tree(tree) => tree.get(txn, selector).await,
            Self::Slice(slice) => slice.get(txn, selector).await,
        }
    }

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        value: Self::Item,
    ) -> TCResult<()> {
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
