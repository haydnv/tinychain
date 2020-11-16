use std::ops::Deref;

use async_trait::async_trait;

use crate::class::{Instance, State, TCStream};
use crate::collection::class::CollectionInstance;
use crate::collection::{Collection, CollectionView};
use crate::error::{self, TCResult};
use crate::request::Request;
use crate::scalar::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::schema::Column;

mod bounds;
mod class;
mod collator;
mod file;
mod slice;

pub use bounds::*;
pub use class::*;
pub use file::*;
pub use slice::*;

pub type Key = Vec<Value>;

fn format_schema(schema: &[Column]) -> String {
    let schema: Vec<String> = schema.iter().map(|c| c.to_string()).collect();
    format!("[{}]", schema.join(", "))
}

fn validate_key(key: Key, schema: &[Column]) -> TCResult<Key> {
    if key.len() != schema.len() {
        return Err(error::bad_request(
            &format!("Invalid key {} for schema", Value::Tuple(key.to_vec())),
            format_schema(schema),
        ));
    }

    validate_prefix(key, schema)
}

fn validate_prefix(prefix: Key, schema: &[Column]) -> TCResult<Key> {
    if prefix.len() > schema.len() {
        return Err(error::bad_request(
            &format!(
                "Invalid selector {} for schema",
                Value::Tuple(prefix.to_vec())
            ),
            format_schema(schema),
        ));
    }

    prefix
        .into_iter()
        .zip(schema)
        .map(|(value, column)| {
            let value = column.dtype().try_cast(value)?;

            let key_size = bincode::serialized_size(&value)?;
            if let Some(size) = column.max_len() {
                if key_size as usize > *size {
                    return Err(error::bad_request(
                        "Column value exceeds the maximum lendth",
                        column.name(),
                    ));
                }
            }

            Ok(value)
        })
        .collect()
}

#[derive(Clone)]
pub struct BTreeImpl<T: BTreeInstance> {
    inner: T,
}

impl<T: BTreeInstance> BTreeImpl<T> {
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: BTreeInstance> Deref for BTreeImpl<T> {
    type Target = T;

    fn deref(&'_ self) -> &'_ T {
        &self.inner
    }
}

impl<T: BTreeInstance> From<T> for BTreeImpl<T> {
    fn from(inner: T) -> BTreeImpl<T> {
        BTreeImpl { inner }
    }
}

#[derive(Clone)]
pub enum BTree {
    Tree(BTreeImpl<BTreeFile>),
    View(BTreeImpl<BTreeSlice>),
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
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State> {
        match self {
            Self::Tree(tree) => tree.get(request, txn, path, selector).await,
            Self::View(view) => view.get(request, txn, path, selector).await,
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
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()> {
        match self {
            Self::Tree(tree) => tree.put(request, txn, path, selector, value).await,
            Self::View(view) => view.put(request, txn, path, selector, value).await,
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
        BTree::Tree(btree.into())
    }
}

impl From<BTreeSlice> for BTree {
    fn from(slice: BTreeSlice) -> BTree {
        BTree::View(slice.into())
    }
}

impl From<BTree> for Collection {
    fn from(btree: BTree) -> Collection {
        Collection::View(CollectionView::BTree(btree))
    }
}
