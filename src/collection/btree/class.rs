use std::fmt;

use async_trait::async_trait;

use crate::class::{Class, NativeClass, TCResult};
use crate::collection::class::*;
use crate::error;
use crate::scalar::{label, Link, PathSegment, TCPathBuf, Value};
use crate::transaction::Txn;

use super::BTree;

pub trait BTreeInstance {}

#[derive(Clone, Eq, PartialEq)]
pub enum BTreeType {
    Tree,
    Slice,
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
            Self::Slice => write!(f, "class BTreeSlice"),
        }
    }
}
