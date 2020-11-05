use std::fmt;

use async_trait::async_trait;

use crate::class::{Class, Instance, NativeClass, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionView};
use crate::error;
use crate::scalar::{label, Link, Scalar, TCPath, Value};
use crate::transaction::{Transact, Txn, TxnId};

use super::{BTreeFile, BTreeSlice, Key, Selector};

#[derive(Clone, Eq, PartialEq)]
pub enum BTreeType {
    Tree,
    Slice,
}

impl Class for BTreeType {
    type Instance = BTree;
}

impl NativeClass for BTreeType {
    fn from_path(path: &TCPath) -> TCResult<Self> {
        let path = path.from_path(&Self::prefix())?;

        if path.is_empty() {
            Ok(BTreeType::Tree)
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

    async fn get(&self, _txn: &Txn, _schema: Value) -> TCResult<BTree> {
        Err(error::not_implemented("BTreeType::get"))
    }
}

impl From<BTreeType> for CollectionType {
    fn from(btree_type: BTreeType) -> CollectionType {
        CollectionType::View(CollectionViewType::BTree(btree_type))
    }
}

impl From<BTreeType> for Link {
    fn from(_btt: BTreeType) -> Link {
        BTreeType::prefix().into()
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
    Tree(BTreeFile),
    View(BTreeSlice),
}

impl Instance for BTree {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        match self {
            Self::Tree(tree) => tree.class(),
            Self::View(view) => view.class(),
        }
    }
}

#[async_trait]
impl CollectionInstance for BTree {
    type Item = Key;
    type Slice = BTreeSlice;

    async fn get(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        match self {
            Self::Tree(tree) => tree.get(txn, path, selector).await,
            Self::View(view) => view.get(txn, path, selector).await,
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::Tree(tree) => tree.is_empty(txn).await,
            Self::View(view) => view.is_empty(txn).await,
        }
    }

    async fn put(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        match self {
            Self::Tree(tree) => tree.put(txn, path, selector, value).await,
            Self::View(view) => view.put(txn, path, selector, value).await,
        }
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        match self {
            Self::Tree(tree) => tree.to_stream(txn).await,
            Self::View(view) => view.to_stream(txn).await,
        }
    }
}

#[async_trait]
impl Transact for BTree {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.commit(txn_id).await,
            Self::View(_) => (), // no-op
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.rollback(txn_id).await,
            Self::View(_) => (), // no-op
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Tree(tree) => tree.finalize(txn_id).await,
            Self::View(_) => (), // no-op
        }
    }
}

impl From<BTreeFile> for BTree {
    fn from(btree: BTreeFile) -> BTree {
        BTree::Tree(btree)
    }
}

impl From<BTreeSlice> for BTree {
    fn from(slice: BTreeSlice) -> BTree {
        BTree::View(slice)
    }
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Collection {
        Collection::View(CollectionView::BTree(btree))
    }
}

impl From<BTree> for BTreeSlice {
    fn from(btree: BTree) -> BTreeSlice {
        match btree {
            BTree::View(slice) => slice,
            BTree::Tree(btree) => BTreeSlice {
                source: btree,
                bounds: Selector::all(),
            },
        }
    }
}
