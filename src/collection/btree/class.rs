use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::class::{Class, Instance, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionView};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::{Link, TCPath};
use crate::value::{label, Value};

use super::{BTreeFile, BTreeSlice, Key};

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

    async fn get(_txn: Arc<Txn>, path: &TCPath, _schema: Value) -> TCResult<BTree> {
        if path.is_empty() {
            Err(error::not_implemented("BTreeType::get"))
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
    type Error = error::TCError;
    type Item = Key;
    type Slice = BTreeSlice;

    async fn get(&self, txn: Arc<Txn>, selector: Value) -> TCResult<BTreeSlice> {
        match self {
            Self::Tree(tree) => tree.get(txn, selector).await,
            Self::Slice(slice) => slice.get(txn, selector).await,
        }
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        match self {
            Self::Tree(tree) => tree.is_empty(txn).await,
            Self::Slice(slice) => slice.is_empty(txn).await,
        }
    }

    async fn put(&self, txn: Arc<Txn>, selector: Value, value: Key) -> TCResult<()> {
        match self {
            Self::Tree(tree) => tree.put(txn, selector, value).await,
            Self::Slice(slice) => slice.put(txn, selector, value).await,
        }
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Self::Item>> {
        match self {
            Self::Tree(tree) => tree.to_stream(txn).await,
            Self::Slice(slice) => slice.to_stream(txn).await,
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
