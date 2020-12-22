use std::fmt;

use async_trait::async_trait;
use futures::{Stream, TryFutureExt};

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::collection::class::*;
use crate::collection::schema::Column;
use crate::error;
use crate::general::{TCResult, TCTryStream, TryCastInto};
use crate::scalar::{label, Link, PathSegment, TCPathBuf, Value};
use crate::transaction::{Transact, Txn, TxnId};

use super::{BTree, BTreeRange, Key};
use crate::collection::{BTreeFile, BTreeImpl};

#[async_trait]
pub trait BTreeInstance: Instance<Class = BTreeType> + Transact {
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

    async fn len(&self, txn_id: &TxnId, range: BTreeRange) -> TCResult<u64>;

    fn schema(&'_ self) -> &'_ [Column];

    async fn stream<'a>(
        &'a self,
        txn_id: &'a TxnId,
        range: BTreeRange,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Key>>;
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

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<BTree> {
        match self {
            Self::Tree => {
                let schema =
                    schema.try_cast_into(|v| error::bad_request("Invalid B-Tree schema", v))?;
                BTreeFile::create(txn, schema)
                    .map_ok(BTreeImpl::from)
                    .map_ok(BTree::Tree)
                    .await
            }
            Self::View => Err(error::unsupported(
                "Cannot instantiate a B-Tree view directly",
            )),
        }
    }
}

impl From<BTreeType> for TCType {
    fn from(btt: BTreeType) -> TCType {
        TCType::Collection(btt.into())
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
