use std::fmt;

use async_trait::async_trait;
use futures::Stream;

use crate::class::{Class, NativeClass, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::schema::Column;
use crate::error;
use crate::scalar::{label, Link, PathSegment, TCPathBuf, Value};
use crate::transaction::{Txn, TxnId};

use super::{BTree, BTreeRange, Key};

#[async_trait]
pub trait BTreeInstance {
    async fn delete(&self, txn_id: &TxnId, range: BTreeRange) -> TCResult<()>;

    async fn insert(&self, txn_id: &TxnId, key: Key) -> TCResult<()>;

    async fn insert_from<S: Stream<Item = Key> + Send>(
        &self,
        txn_id: &TxnId,
        source: S,
    ) -> TCResult<()>;

    async fn try_insert_from<S: Stream<Item = TCResult<Key>> + Send>(
        &self,
        txn_id: &TxnId,
        source: S,
    ) -> TCResult<()>;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool>;

    async fn len(&self, txn_id: TxnId, range: BTreeRange) -> TCResult<u64>;

    fn schema(&'_ self) -> &'_ [Column];

    async fn stream(
        &self,
        txn_id: TxnId,
        range: BTreeRange,
        reverse: bool,
    ) -> TCResult<TCStream<Key>>;
}

#[derive(Clone, Eq, PartialEq)]
pub enum BTreeType {
    Tree,
    View,
}

impl Class for BTreeType {
    type Instance = BTree;
}

impl NativeClass for BTreeType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let path = Self::prefix().try_suffix(path)?;

        if path.is_empty() {
            Ok(BTreeType::Tree)
        } else {
            Err(error::path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        CollectionType::prefix().append(label("btree"))
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
            Self::View => write!(f, "class BTreeView"),
        }
    }
}
