use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};

use crate::auth::{Scope, SCOPE_READ, SCOPE_WRITE};
use crate::class::{Instance, State, TCType};
use crate::collection::class::CollectionInstance;
use crate::collection::Collection;
use crate::error;
use crate::general::{Map, TCResult, TCStream, TCTryStream, TryCastInto};
use crate::handler::*;
use crate::request::Request;
use crate::scalar::{label, MethodType, PathSegment, Scalar, ScalarClass, ScalarInstance, Value};
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

fn format_schema(schema: &[Column]) -> String {
    let schema: Vec<String> = schema.iter().map(|c| c.to_string()).collect();
    format!("[{}]", schema.join(", "))
}

fn validate_key(key: Key, schema: &[Column]) -> TCResult<Key> {
    if key.len() != schema.len() {
        return Err(error::bad_request(
            &format!(
                "Invalid key {} for schema",
                Value::Tuple(key.to_vec().into())
            ),
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
                Value::Tuple(prefix.to_vec().into())
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
                        "Column value exceeds the maximum length",
                        column.name(),
                    ));
                }
            }

            Ok(value)
        })
        .collect()
}

struct CountHandler<'a, T: BTreeInstance> {
    btree: &'a T,
}

#[async_trait]
impl<'a, T: BTreeInstance> Handler for CountHandler<'a, T> {
    fn subject(&self) -> TCType {
        Instance::class(self.btree).into()
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_READ.into())
    }

    async fn handle_get(&self, txn: &Txn, range: Value) -> TCResult<State> {
        let range = validate_range(range, self.btree.schema())?;
        let count = self.btree.len(txn.id(), range).await?;
        Ok(State::Scalar(Scalar::Value(Value::Number(count.into()))))
    }
}

struct DeleteHandler<'a, T: BTreeInstance> {
    btree: &'a T,
}

#[async_trait]
impl<'a, T: BTreeInstance> Handler for DeleteHandler<'a, T> {
    fn subject(&self) -> TCType {
        Instance::class(self.btree).into()
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_WRITE.into())
    }

    async fn handle_delete(&self, txn: &Txn, range: Value) -> TCResult<()> {
        let range = validate_range(range, self.btree.schema())?;
        BTreeInstance::delete(self.btree, txn.id(), range).await
    }
}

struct SliceHandler<'a, T: BTreeInstance> {
    btree: &'a T,
}

impl<'a, T: BTreeInstance> SliceHandler<'a, T>
where
    Collection: From<T>,
    BTree: From<T>,
{
    async fn slice(&self, txn: &Txn, range: BTreeRange) -> TCResult<State> {
        if range == BTreeRange::default() {
            Ok(State::Collection(self.btree.clone().into()))
        } else if range.is_key(self.btree.schema()) {
            let mut rows = self.btree.stream(txn.id(), range, false).await?;
            let row = rows.try_next().await?;
            row.ok_or_else(|| error::not_found("(btree key)"))
                .map(|key| State::Scalar(Scalar::Value(Value::Tuple(key.to_vec().into()))))
        } else {
            let slice = BTreeSlice::new(self.btree.clone().into(), range, false)?;
            Ok(State::Collection(slice.into()))
        }
    }
}

#[async_trait]
impl<'a, T: BTreeInstance> Handler for SliceHandler<'a, T>
where
    Collection: From<T>,
    BTree: From<T>,
{
    fn subject(&self) -> TCType {
        Instance::class(self.btree).into()
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_READ.into())
    }

    async fn handle_get(&self, txn: &Txn, range: Value) -> TCResult<State> {
        let range = validate_range(range, self.btree.schema())?;
        self.slice(txn, range).await
    }

    async fn handle_post(
        &self,
        _request: &Request,
        txn: &Txn,
        mut params: Map<Scalar>,
    ) -> TCResult<State> {
        let range = params
            .remove(&label("where").into())
            .unwrap_or_else(|| Scalar::from(()));

        let range = validate_range(range, self.btree.schema())?;
        self.slice(txn, range).await
    }
}

struct ReverseHandler<'a, T: BTreeInstance> {
    btree: &'a T,
}

impl<'a, T: BTreeInstance> ReverseHandler<'a, T>
where
    BTree: From<T>,
{
    fn reverse(&self, range: BTreeRange) -> TCResult<State> {
        let slice = BTreeSlice::new(self.btree.clone().into(), range, true)?;
        Ok(State::Collection(Collection::BTree(slice.into())))
    }
}

#[async_trait]
impl<'a, T: BTreeInstance> Handler for ReverseHandler<'a, T>
where
    BTree: From<T>,
{
    fn subject(&self) -> TCType {
        Instance::class(self.btree).into()
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_READ.into())
    }

    async fn handle_get(&self, _txn: &Txn, range: Value) -> TCResult<State> {
        let range = validate_range(range, self.btree.schema())?;
        self.reverse(range)
    }

    async fn handle_post(
        &self,
        _request: &Request,
        _txn: &Txn,
        mut params: Map<Scalar>,
    ) -> TCResult<State> {
        let range = params
            .remove(&label("where").into())
            .unwrap_or_else(|| Scalar::from(()));

        let range = validate_range(range, self.btree.schema())?;
        self.reverse(range)
    }
}

struct WriteHandler<'a, T: BTreeInstance> {
    btree: &'a T,
}

#[async_trait]
impl<'a, T: BTreeInstance> Handler for WriteHandler<'a, T> {
    fn subject(&self) -> TCType {
        Instance::class(self.btree).into()
    }

    fn scope(&self) -> Option<Scope> {
        Some(SCOPE_WRITE.into())
    }

    async fn handle_put(
        &self,
        _request: &Request,
        txn: &Txn,
        range: Value,
        data: State,
    ) -> TCResult<()> {
        let range = validate_range(range, self.btree.schema())?;

        if range == BTreeRange::default() {
            match data {
                State::Collection(collection) => {
                    let keys = collection.to_stream(txn).await?;
                    let keys = keys
                        .map(|s| s.try_cast_into(|k| error::bad_request("Invalid BTree key", k)));

                    self.btree.try_insert_from(txn.id(), keys).await?;
                }
                State::Scalar(scalar) if scalar.matches::<Vec<Key>>() => {
                    let keys: Vec<Key> = scalar.opt_cast_into().unwrap();
                    self.btree
                        .insert_from(txn.id(), stream::iter(keys.into_iter()))
                        .await?;
                }
                State::Scalar(scalar) if scalar.matches::<Key>() => {
                    let key: Key = scalar.opt_cast_into().unwrap();
                    let key = validate_key(key, self.btree.schema())?;
                    self.btree.insert(txn.id(), key).await?;
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
}

#[derive(Clone)]
pub struct BTreeImpl<T: BTreeInstance> {
    inner: T,
}

impl<T: BTreeInstance> BTreeImpl<T> {
    fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: BTreeInstance> Instance for BTreeImpl<T> {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        self.inner.class()
    }
}

#[async_trait]
impl<T: BTreeInstance> CollectionInstance for BTreeImpl<T> {
    type Item = Key;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.inner.is_empty(txn).await
    }

    async fn to_stream<'a>(&'a self, txn: &'a Txn) -> TCResult<TCStream<'a, Scalar>> {
        let stream = self.stream(txn.id(), BTreeRange::default(), false).await?;
        Ok(Box::pin(stream.map(|row| row.unwrap()).map(Scalar::from)))
    }
}

impl<T: BTreeInstance> Route for BTreeImpl<T>
where
    Collection: From<T>,
    BTree: From<T>,
{
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        let btree = &self.inner;

        if path.is_empty() {
            match method {
                MethodType::Get | MethodType::Post => Some(Box::new(SliceHandler { btree })),
                MethodType::Put => Some(Box::new(WriteHandler { btree })),
                MethodType::Delete => Some(Box::new(DeleteHandler { btree })),
            }
        } else if path.len() == 1 {
            match path[0].as_str() {
                "count" => Some(Box::new(CountHandler { btree })),
                "reverse" => Some(Box::new(ReverseHandler { btree })),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl<T: BTreeInstance> Deref for BTreeImpl<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: BTreeInstance> From<T> for BTreeImpl<T> {
    fn from(inner: T) -> Self {
        Self { inner }
    }
}

#[derive(Clone)]
pub enum BTree {
    Tree(BTreeImpl<BTreeFile>),
    View(BTreeImpl<BTreeSlice>),
}

impl Instance for BTree {
    type Class = BTreeType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Tree(tree) => tree.class(),
            Self::View(view) => view.class(),
        }
    }
}

#[async_trait]
impl CollectionInstance for BTree {
    type Item = Key;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::Tree(tree) => tree.is_empty(txn).await,
            Self::View(view) => view.is_empty(txn).await,
        }
    }

    async fn to_stream<'a>(&'a self, txn: &'a Txn) -> TCResult<TCStream<'a, Scalar>> {
        match self {
            Self::Tree(tree) => tree.to_stream(txn).await,
            Self::View(view) => view.to_stream(txn).await,
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

    async fn len(&self, txn_id: &TxnId, range: BTreeRange) -> TCResult<u64> {
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

    async fn stream<'a>(
        &'a self,
        txn_id: &'a TxnId,
        range: BTreeRange,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Key>> {
        match self {
            Self::Tree(tree) => tree.stream(txn_id, range, reverse).await,
            Self::View(view) => view.stream(txn_id, range, reverse).await,
        }
    }
}

impl Route for BTree {
    fn route(
        &'_ self,
        method: MethodType,
        path: &'_ [PathSegment],
    ) -> Option<Box<dyn Handler + '_>> {
        match self {
            Self::Tree(tree) => tree.route(method, path),
            Self::View(view) => view.route(method, path),
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

impl fmt::Display for BTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Tree(_) => write!(f, "(b-tree)"),
            Self::View(_) => write!(f, "(b-tree slice)"),
        }
    }
}
