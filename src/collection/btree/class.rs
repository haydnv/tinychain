use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;

use crate::class::{Class, Instance, TCBoxFuture, TCBoxTryFuture, TCResult, TCStream};
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

    async fn get(&self, _txn: Arc<Txn>, _schema: Value) -> TCResult<BTree> {
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

impl CollectionInstance for BTree {
    type Item = Key;
    type Slice = BTreeSlice;

    fn get<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        selector: Value,
    ) -> TCBoxTryFuture<CollectionItem<Self::Item, Self::Slice>> {
        match self {
            Self::Tree(tree) => tree.get(txn, path, selector),
            Self::View(view) => view.get(txn, path, selector),
        }
    }

    fn is_empty<'a>(&'a self, txn: Arc<Txn>) -> TCBoxTryFuture<'a, bool> {
        match self {
            Self::Tree(tree) => tree.is_empty(txn),
            Self::View(view) => view.is_empty(txn),
        }
    }

    fn put<'a>(
        &'a self,
        txn: Arc<Txn>,
        path: TCPath,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Tree(tree) => tree.put(txn, path, selector, value),
            Self::View(view) => view.put(txn, path, selector, value),
        }
    }

    fn to_stream<'a>(&'a self, txn: Arc<Txn>) -> TCBoxTryFuture<'a, TCStream<Scalar>> {
        match self {
            Self::Tree(tree) => tree.to_stream(txn),
            Self::View(view) => view.to_stream(txn),
        }
    }
}

impl Transact for BTree {
    fn commit<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        match self {
            Self::Tree(tree) => tree.commit(txn_id),
            Self::View(_) => Box::pin(future::ready(())),
        }
    }

    fn rollback<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()> {
        match self {
            Self::Tree(tree) => tree.rollback(txn_id),
            Self::View(_) => Box::pin(future::ready(())),
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
