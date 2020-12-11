use std::ops::Deref;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt};

use crate::class::{Instance, State, TCStream};
use crate::collection::class::CollectionInstance;
use crate::collection::{Collection, CollectionView};
use crate::error::{self, TCResult};
use crate::handler::Public;
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
pub use collator::*;
pub use file::*;
pub use slice::*;

pub type Key = Vec<Value>;

const ERR_INVALID_RANGE: &str = "Invalid BTree range";

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

    async fn route(
        &self,
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        range: BTreeRange,
    ) -> TCResult<State> {
        if path.is_empty() {
            if range == BTreeRange::default() {
                Ok(State::Collection(self.inner.clone().into()))
            } else if range.is_key(self.schema()) {
                let mut rows = self.stream(txn.id().clone(), range, false).await?;
                let row = rows.next().await;
                row.ok_or_else(|| error::not_found("(btree key)"))
                    .map(|key| State::Scalar(Scalar::Value(Value::Tuple(key))))
            } else {
                let slice = BTreeSlice::new(self.inner.clone().into(), range, false)?;
                Ok(State::Collection(slice.into()))
            }
        } else if path.len() == 1 {
            match path[0].as_str() {
                "count" => {
                    let len = self.len(txn.id().clone(), range).await?;
                    Ok(State::Scalar(Number::from(len).into()))
                }
                "reverse" => {
                    let slice = BTreeSlice::new(self.inner.clone().into(), range, true)?;
                    Ok(State::Collection(slice.into()))
                }
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(path))
        }
    }
}

#[async_trait]
impl<T: BTreeInstance> CollectionInstance for BTreeImpl<T> {
    type Item = Key;
    type Slice = BTreeSlice;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        BTreeInstance::is_empty(self.deref(), txn).await
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        let stream = self
            .stream(txn.id().clone(), BTreeRange::default(), false)
            .await?
            .map(Value::Tuple)
            .map(Scalar::Value);

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl<T: BTreeInstance> Public for BTreeImpl<T> {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        range: Value,
    ) -> TCResult<State> {
        let range = BTreeRange::try_cast_from(range, |v| error::bad_request(ERR_INVALID_RANGE, v))?;
        let range = validate_range(range, self.schema())?;
        self.route(request, txn, path, range).await
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        mut params: Object,
    ) -> TCResult<State> {
        let range = params
            .remove(&label("where").into())
            .unwrap_or_else(|| Scalar::from(()));

        let range = BTreeRange::try_cast_from(range, |s| error::bad_request(ERR_INVALID_RANGE, s))?;
        let range = validate_range(range, self.schema())?;

        self.route(request, txn, path, range).await
    }

    async fn put(
        &self,
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        range: Value,
        key: State,
    ) -> TCResult<()> {
        let range: BTreeRange =
            range.try_cast_into(|v| error::bad_request("Invalid BTree selector", v))?;
        let range = validate_range(range, self.schema())?;

        if path.len() == 1 && &path[0] == "delete" {
            return BTreeInstance::delete(self.deref(), txn.id(), range).await;
        } else if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        if range == BTreeRange::default() {
            match key {
                State::Collection(collection) => {
                    let keys = collection.to_stream(txn.clone()).await?;
                    let keys = keys
                        .map(|s| s.try_cast_into(|k| error::bad_request("Invalid BTree key", k)));

                    self.try_insert_from(txn.id(), keys).await?;
                }
                State::Scalar(scalar) if scalar.matches::<Vec<Key>>() => {
                    let keys: Vec<Key> = scalar.opt_cast_into().unwrap();
                    self.insert_from(txn.id(), stream::iter(keys.into_iter()))
                        .await?;
                }
                State::Scalar(scalar) if scalar.matches::<Key>() => {
                    let key: Key = scalar.opt_cast_into().unwrap();
                    let key = validate_key(key, self.schema())?;
                    self.insert(txn.id(), key).await?;
                }
                other => {
                    return Err(error::bad_request("Invalid key for BTree", other));
                }
            }
        } else {
            return Err(error::not_implemented("BTree::update"));
        }

        Ok(())
    }

    async fn delete(
        &self,
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        range: Value,
    ) -> TCResult<()> {
        if path.is_empty() {
            let range =
                BTreeRange::try_cast_from(range, |v| error::bad_request(ERR_INVALID_RANGE, v))?;
            let range = validate_range(range, self.schema())?;

            BTreeInstance::delete(self.deref(), txn.id(), range).await
        } else {
            Err(error::path_not_found(path))
        }
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

#[async_trait]
impl CollectionInstance for BTree {
    type Item = Scalar;
    type Slice = CollectionView;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::Tree(tree) => CollectionInstance::is_empty(tree, txn).await,
            Self::View(view) => CollectionInstance::is_empty(view, txn).await,
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
impl Public for BTree {
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

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        match self {
            Self::Tree(tree) => tree.post(request, txn, path, params).await,
            Self::View(view) => view.post(request, txn, path, params).await,
        }
    }

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<()> {
        match self {
            Self::Tree(tree) => tree.delete(request, txn, path, selector).await,
            Self::View(view) => view.delete(request, txn, path, selector).await,
        }
    }
}

impl Instance for BTree {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        match self {
            Self::Tree(_) => BTreeType::Tree,
            Self::View(_) => BTreeType::View,
        }
    }
}

#[async_trait]
impl BTreeInstance for BTree {
    async fn delete(&self, txn_id: &TxnId, range: BTreeRange) -> TCResult<()> {
        match self {
            Self::Tree(tree) => BTreeInstance::delete(tree.deref(), txn_id, range).await,
            Self::View(view) => BTreeInstance::delete(view.deref(), txn_id, range).await,
        }
    }

    async fn insert(&self, txn_id: &TxnId, key: Key) -> TCResult<()> {
        match self {
            Self::Tree(tree) => tree.insert(txn_id, key).await,
            Self::View(view) => view.insert(txn_id, key).await,
        }
    }

    async fn insert_from<S: Stream<Item = Key> + Send>(
        &self,
        txn_id: &TxnId,
        source: S,
    ) -> TCResult<()> {
        match self {
            Self::Tree(tree) => tree.insert_from(txn_id, source).await,
            Self::View(view) => view.insert_from(txn_id, source).await,
        }
    }

    async fn try_insert_from<S: Stream<Item = TCResult<Key>> + Send>(
        &self,
        txn_id: &TxnId,
        source: S,
    ) -> TCResult<()> {
        match self {
            Self::Tree(tree) => tree.try_insert_from(txn_id, source).await,
            Self::View(view) => view.try_insert_from(txn_id, source).await,
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::Tree(tree) => BTreeInstance::is_empty(tree.deref(), txn).await,
            Self::View(view) => BTreeInstance::is_empty(view.deref(), txn).await,
        }
    }

    async fn len(&self, txn_id: TxnId, range: BTreeRange) -> TCResult<u64> {
        match self {
            Self::Tree(tree) => tree.len(txn_id, range).await,
            Self::View(view) => view.len(txn_id, range).await,
        }
    }

    fn schema(&'_ self) -> &'_ [Column] {
        match self {
            Self::Tree(tree) => tree.schema(),
            Self::View(view) => view.schema(),
        }
    }

    async fn stream(
        &self,
        txn_id: TxnId,
        range: BTreeRange,
        reverse: bool,
    ) -> TCResult<TCStream<Key>> {
        match self {
            Self::Tree(tree) => tree.stream(txn_id, range, reverse).await,
            Self::View(view) => view.stream(txn_id, range, reverse).await,
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
