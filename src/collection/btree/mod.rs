use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{self, join, try_join, try_join_all, Future, TryFutureExt};
use futures::stream::{self, FuturesOrdered, Stream, StreamExt, TryStreamExt};
use log::debug;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::block::File;
use crate::block::{Block, BlockData, BlockId, BlockMut, BlockOwned};
use crate::class::{Instance, TCBoxTryFuture, TCResult, TCStream};
use crate::error;
use crate::scalar::{
    CastFrom, CastInto, PathSegment, Scalar, TryCastFrom, TryCastInto, Value, ValueClass, ValueType,
};
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};

use super::class::*;
use super::schema::{Column, RowSchema};
use super::{Collection, CollectionBase, CollectionView};

mod class;
mod collator;

pub type BTree = class::BTree;
pub type BTreeType = class::BTreeType;

const DEFAULT_BLOCK_SIZE: usize = 4_000;
const BLOCK_ID_SIZE: usize = 128; // UUIDs are 128-bit

type NodeId = BlockId;

#[derive(Clone, Deserialize, Serialize)]
struct NodeKey {
    value: Vec<Value>,
    deleted: bool,
}

impl From<&[Value]> for NodeKey {
    fn from(values: &[Value]) -> NodeKey {
        values.to_vec().into()
    }
}

impl From<Vec<Value>> for NodeKey {
    fn from(value: Vec<Value>) -> NodeKey {
        NodeKey {
            value,
            deleted: false,
        }
    }
}

impl fmt::Display for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BTree node key: {}", Value::Tuple(self.value.to_vec()))
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct Node {
    leaf: bool,
    keys: Vec<NodeKey>,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    rebalance: bool, // TODO: implement rebalancing to clear deleted values
}

impl Node {
    fn new(leaf: bool, parent: Option<NodeId>) -> Node {
        Node {
            leaf,
            keys: vec![],
            parent,
            children: vec![],
            rebalance: false,
        }
    }

    fn values(&self) -> Vec<&[Value]> {
        self.keys.iter().map(|k| &k.value[..]).collect()
    }
}

impl TryFrom<Bytes> for Node {
    type Error = error::TCError;

    fn try_from(serialized: Bytes) -> TCResult<Node> {
        bincode::deserialize(&serialized).map_err(|e| e.into())
    }
}

impl From<Node> for Bytes {
    fn from(node: Node) -> Bytes {
        bincode::serialize(&node).unwrap().into()
    }
}

impl BlockData for Node {
    fn size(&self) -> usize {
        self.keys.len() + self.children.len()
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(BTree node with {} keys)", self.keys.len())
    }
}

pub type Key = Vec<Value>;

fn format_schema(schema: &[Column]) -> String {
    let schema: Vec<String> = schema.iter().map(|c| c.to_string()).collect();
    format!("[{}]", schema.join(", "))
}

fn validate_key(key: &[Value], schema: &[Column]) -> TCResult<()> {
    if key.len() != schema.len() {
        return Err(error::bad_request(
            &format!("Invalid key {} for schema", Value::Tuple(key.to_vec())),
            format_schema(schema),
        ));
    }

    validate_prefix(key, schema)
}

fn validate_selector(selector: &Selector, schema: &[Column]) -> TCResult<()> {
    match selector {
        Selector::Key(key) => validate_prefix(key, schema),
        Selector::Range(range, _) => validate_range(range, schema),
    }
}

fn validate_prefix(prefix: &[Value], schema: &[Column]) -> TCResult<()> {
    if prefix.len() > schema.len() {
        return Err(error::bad_request(
            &format!(
                "Invalid selector {} for schema",
                Value::Tuple(prefix.to_vec())
            ),
            format_schema(schema),
        ));
    }

    for (val, col) in prefix.iter().zip(&schema[0..prefix.len()]) {
        if !val.is_a(*col.dtype()) {
            return Err(error::bad_request(
                &format!("Expected {} for {}, found", col.dtype(), col.name()),
                val,
            ));
        }

        let key_size = bincode::serialized_size(&val)?;
        if let Some(size) = col.max_len() {
            if key_size as usize > *size {
                return Err(error::bad_request(
                    "Column value exceeds the maximum length",
                    col.name(),
                ));
            }
        }
    }

    Ok(())
}

fn validate_range(range: &BTreeRange, schema: &[Column]) -> TCResult<()> {
    use Bound::*;

    let expect = |column: &Column, value: &Value| {
        value.expect(
            *column.dtype(),
            format!("for column {} in BTreeRange", column.name()),
        )
    };

    for (i, column) in schema.iter().enumerate() {
        if i < range.0.len() {
            match &range.0[i] {
                Unbounded => {}
                Included(value) => expect(column, value)?,
                Excluded(value) => expect(column, value)?,
            }
        }

        if i < range.1.len() {
            match &range.1[i] {
                Unbounded => {}
                Included(value) => expect(column, value)?,
                Excluded(value) => expect(column, value)?,
            }
        }
    }

    Ok(())
}

type Selection = FuturesOrdered<Pin<Box<dyn Future<Output = TCStream<Key>> + Send + Unpin>>>;

#[derive(Clone)]
pub struct BTreeSlice {
    source: BTreeFile,
    bounds: Selector,
}

impl BTreeSlice {
    fn new(source: BTreeFile, bounds: Selector) -> BTreeSlice {
        BTreeSlice { source, bounds }
    }
}

impl Instance for BTreeSlice {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        BTreeType::Slice
    }
}

#[async_trait]
impl CollectionInstance for BTreeSlice {
    type Item = Key;
    type Slice = BTreeSlice;

    async fn get(
        &self,
        txn: Txn,
        path: &[PathSegment],
        bounds: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let bounds: Selector =
            bounds.try_cast_into(|v| error::bad_request("Invalid BTree selector", v))?;
        validate_selector(&bounds, self.source.schema())?;

        let schema: Vec<ValueType> = self.source.schema().iter().map(|c| *c.dtype()).collect();
        match (&self.bounds, &bounds) {
            (Selector::Key(this), Selector::Key(that)) if this == that => {
                Ok(CollectionItem::Slice(self.clone()))
            }
            (Selector::Range(container, _), Selector::Range(contained, _))
                if container.contains(contained, &schema)? =>
            {
                self.source.get(txn, path, bounds.cast_into()).await
            }
            _ => Err(error::bad_request(
                &format!("BTreeSlice[{}] does not contain", &self.bounds),
                &bounds,
            )),
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        let count = self
            .source
            .clone()
            .len(txn.id().clone(), self.bounds.clone())
            .await?;

        Ok(count == 0)
    }

    async fn put(
        &self,
        _txn: Txn,
        _path: &[PathSegment],
        _selector: Value,
        _value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        Err(error::unsupported(
            "BTreeSlice is immutable; write to the source BTree instead",
        ))
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        let rows = self
            .source
            .clone()
            .slice(txn.id().clone(), self.bounds.clone())
            .await?;

        Ok(Box::pin(rows.map(Value::Tuple).map(Scalar::Value)))
    }
}

#[async_trait]
impl Transact for BTreeSlice {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}

impl From<BTreeSlice> for Collection {
    fn from(btree_slice: BTreeSlice) -> Collection {
        Collection::View(btree_slice.into())
    }
}

impl From<BTreeSlice> for CollectionView {
    fn from(btree_slice: BTreeSlice) -> CollectionView {
        CollectionView::BTree(class::BTree::View(btree_slice))
    }
}

#[derive(Clone)]
pub struct BTreeFile {
    file: Arc<File<Node>>,
    schema: RowSchema,
    order: usize,
    collator: collator::Collator,
    root: TxnLock<Mutable<NodeId>>,
}

impl BTreeFile {
    pub async fn create(txn: &Txn, schema: RowSchema) -> TCResult<Self> {
        let file = txn.context().await?;

        if !file.is_empty(txn.id()).await? {
            return Err(error::internal(
                "Tried to create a new BTree without a new File",
            ));
        }

        let mut key_size = 0;
        for col in &schema {
            if let Some(size) = col.dtype().size() {
                key_size += size;
                if col.max_len().is_some() {
                    return Err(error::bad_request(
                        "Found maximum length specified for a scalar type",
                        col.dtype(),
                    ));
                }
            } else if let Some(size) = col.max_len() {
                key_size += size + 8; // add 8 bytes for bincode to encode the length
            } else {
                return Err(error::bad_request(
                    "Type requires a maximum length",
                    col.dtype(),
                ));
            }
        }
        // the "leaf" and "deleted" booleans each add one byte to a key as-stored
        key_size += 2;

        let order = if DEFAULT_BLOCK_SIZE > (key_size * 2) + (BLOCK_ID_SIZE * 3) {
            // let m := order
            // maximum block size = (m * key_size) + ((m + 1) * block_id_size)
            // therefore block_size = (m * (key_size + block_id_size)) + block_id_size
            // therefore block_size - block_id_size = m * (key_size + block_id_size)
            // therefore m = floor((block_size - block_id_size) / (key_size + block_id_size))
            (DEFAULT_BLOCK_SIZE - BLOCK_ID_SIZE) / (key_size + BLOCK_ID_SIZE)
        } else {
            2
        };

        let root: BlockId = Uuid::new_v4().into();
        file.clone()
            .create_block(txn.id().clone(), root.clone(), Node::new(true, None))
            .await?;

        let collator = collator::Collator::new(schema.iter().map(|c| *c.dtype()).collect())?;
        Ok(BTreeFile {
            file,
            schema,
            order,
            collator,
            root: TxnLock::new("BTree root".to_string(), root.into()),
        })
    }

    pub fn collator(&'_ self) -> &'_ collator::Collator {
        &self.collator
    }

    pub fn schema(&'_ self) -> &'_ [Column] {
        &self.schema
    }

    async fn get_root<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<Block<'a, Node>> {
        self.root
            .read(txn_id)
            .and_then(|root_id| self.file.get_block(txn_id, (*root_id).clone()))
            .await
    }

    pub fn is_empty<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, bool> {
        Box::pin(async move {
            let root = self.get_root(txn.id()).await?;
            Ok(root.keys.is_empty())
        })
    }

    pub fn len<'a>(self, txn_id: TxnId, selector: Selector) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            let slice = self.slice(txn_id, selector).await?;
            Ok(slice.fold(0u64, |len, _| future::ready(len + 1)).await)
        })
    }

    pub async fn slice(self, txn_id: TxnId, selector: Selector) -> TCResult<TCStream<Key>> {
        validate_selector(&selector, self.schema())?;

        let root_id = self.root.read(&txn_id).await?;
        let root = self
            .file
            .clone()
            .get_block_owned(txn_id.clone(), (*root_id).clone())
            .await?;

        Ok(match selector {
            Selector::Key(key) => self._slice(txn_id, root, key.into()),
            Selector::Range(range, false) => self._slice(txn_id, root, range),
            Selector::Range(range, true) => self._slice_reverse(txn_id, root, range),
        })
    }

    fn _slice(self, txn_id: TxnId, node: BlockOwned<Node>, range: BTreeRange) -> TCStream<Key> {
        let keys = node.values();
        let (l, r) = range.bisect(&keys, &self.collator);

        if node.leaf {
            debug!(
                "_slice BTree node with {} keys from {} to {}",
                keys.len(),
                l,
                r
            );
            if l == r && l < node.keys.len() {
                if node.keys[l].deleted {
                    Box::pin(stream::empty())
                } else {
                    Box::pin(stream::once(future::ready(keys[l].to_vec())))
                }
            } else {
                let keys: Vec<Key> = node.keys[l..r]
                    .iter()
                    .filter(|k| !k.deleted)
                    .map(|k| k.value.to_vec())
                    .collect();
                debug!("_slice BTree node with {} keys not deleted", keys.len());
                Box::pin(stream::iter(keys))
            }
        } else {
            let mut selected: Selection = FuturesOrdered::new();
            for i in l..r {
                let this = self.clone();
                let txn_id = txn_id.clone();
                let child_id = node.children[i].clone();
                let range = range.clone();
                let selection = Box::pin(async move {
                    let node = this
                        .file
                        .clone()
                        .get_block_owned(txn_id.clone(), child_id)
                        .await
                        .unwrap();
                    this._slice(txn_id, node, range)
                });
                selected.push(Box::pin(selection));

                if !node.keys[i].deleted {
                    let key_at_i = node.keys[i].value.to_vec();
                    let key_at_i: TCStream<Key> = Box::pin(stream::once(future::ready(key_at_i)));
                    selected.push(Box::pin(future::ready(key_at_i)));
                }
            }

            let last_child_id = node.children[r].clone();
            let selection = Box::pin(async move {
                let node = self
                    .file
                    .clone()
                    .get_block_owned(txn_id.clone(), last_child_id)
                    .await
                    .unwrap();
                self._slice(txn_id, node, range)
            });
            selected.push(Box::pin(selection));

            Box::pin(selected.flatten())
        }
    }

    fn _slice_reverse(
        self,
        txn_id: TxnId,
        node: BlockOwned<Node>,
        range: BTreeRange,
    ) -> TCStream<Key> {
        let keys = node.values();
        let (l, r) = range.bisect(&keys, &self.collator);

        if node.leaf {
            let keys: Vec<Key> = node.keys[l..r]
                .iter()
                .filter(|k| !k.deleted)
                .map(|k| k.value.to_vec())
                .rev()
                .collect();

            Box::pin(stream::iter(keys))
        } else {
            let mut selected: Selection = FuturesOrdered::new();

            let this = self.clone();
            let txn_id_clone = txn_id.clone();
            let range_clone = range.clone();
            let last_child = node.children[r].clone();
            let selection = Box::pin(async move {
                let node = this
                    .file
                    .clone()
                    .get_block_owned(txn_id_clone.clone(), last_child)
                    .await
                    .unwrap();
                this._slice_reverse(txn_id_clone, node, range_clone)
            });
            selected.push(Box::pin(selection));

            let children = node.children.to_vec();
            for i in (l..r).rev() {
                let this = self.clone();
                let txn_id = txn_id.clone();
                let child_id = children[i].clone();
                let range = range.clone();
                let selection = Box::pin(async move {
                    let node = this
                        .file
                        .clone()
                        .get_block_owned(txn_id.clone(), child_id)
                        .await
                        .unwrap();
                    this._slice_reverse(txn_id, node, range)
                });
                selected.push(Box::pin(selection));

                if !node.keys[i].deleted {
                    let key_at_i = node.keys[i].value.to_vec();
                    let key_at_i: TCStream<Key> = Box::pin(stream::once(future::ready(key_at_i)));
                    selected.push(Box::pin(future::ready(key_at_i)));
                }
            }

            Box::pin(selected.flatten())
        }
    }

    pub fn update<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Selector,
        value: &'a [Value],
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let root_id = self.root.read(txn_id).await?;
            self._update(txn_id, (*root_id).clone(), bounds, value)
                .await
        })
    }

    fn _update<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: NodeId,
        bounds: &'a Selector,
        value: &'a [Value],
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let node = self.file.get_block(txn_id, node_id).await?;
            let keys = node.values();
            let (l, r) = bounds.bisect(&keys, &self.collator);

            if node.leaf {
                if l == r {
                    return Ok(());
                }

                let mut node = node.upgrade().await?;
                for i in l..r {
                    node.keys[i] = value.into();
                }

                Ok(())
            } else {
                let children = node.children.to_vec();

                if r > l {
                    let mut node = node.upgrade().await?;
                    let mut updates = Vec::with_capacity(r - l);
                    for (i, child_id) in children.iter().enumerate().take(r).skip(l) {
                        node.keys[i] = value.into();
                        updates.push(self._update(txn_id, child_id.clone(), bounds, value));
                    }

                    let last_update = self._update(txn_id, children[r].clone(), bounds, value);
                    try_join(try_join_all(updates), last_update).await?;
                    Ok(())
                } else {
                    self._update(txn_id, children[r].clone(), bounds, value)
                        .await
                }
            }
        })
    }

    pub async fn try_insert_from<S: Stream<Item = TCResult<Key>>>(
        &self,
        txn_id: &TxnId,
        source: S,
    ) -> TCResult<()> {
        source
            .and_then(|k| future::ready(validate_key(&k, self.schema()).map(|()| k)))
            .map_ok(|key| self.insert(txn_id, key))
            .try_buffer_unordered(2 * self.order)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    pub async fn insert_from<S: Stream<Item = Key>>(
        &self,
        txn_id: &TxnId,
        source: S,
    ) -> TCResult<()> {
        source
            .map(|k| validate_key(&k, self.schema()).map(|()| k))
            .map_ok(|key| self.insert(txn_id, key))
            .try_buffer_unordered(2 * self.order)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    pub fn insert<'a>(&'a self, txn_id: &'a TxnId, key: Key) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let root_id = self.root.read(txn_id).await?;
            let root_id_clone = (*root_id).clone();
            let root = self.file.get_block(txn_id, root_id_clone).await?;

            if root.keys.len() == (2 * self.order) - 1 {
                let mut root_id = root_id.upgrade().await?;
                let old_root_id = (*root_id).clone();
                let old_root = root.upgrade().await?;

                (*root_id) = self.file.unique_id(&txn_id).await?;
                let mut new_root = Node::new(false, None);
                new_root.children.push(old_root_id.clone());
                let new_root = self
                    .file
                    .create_block(txn_id.clone(), (*root_id).clone(), new_root)
                    .await?;

                self.split_child(txn_id, old_root_id, old_root, 0).await?;
                self._insert(txn_id, new_root, key).await
            } else {
                self._insert(txn_id, root, key).await
            }
        })
    }

    fn _insert<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node: Block<'a, Node>,
        key: Key,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let keys = &node.keys;
            let i = self.collator.bisect_left(&node.values(), &key);
            if node.leaf {
                if i == node.keys.len()
                    || self.collator.compare(&keys[i].value, &key) != Ordering::Equal
                {
                    let mut node = node.upgrade().await?;
                    node.keys.insert(i, key.into());
                    debug!("BTree node now has {} keys", node.keys.len());
                } else if keys[i].value == key && keys[i].deleted {
                    let mut node = node.upgrade().await?;
                    node.keys[i].deleted = false;
                }

                Ok(())
            } else {
                let child_id = node.children[i].clone();
                let mut child = self.file.get_block(txn_id, child_id.clone()).await?;
                if child.keys.len() == (2 * self.order) - 1 {
                    let this_key = &node.keys[i].value.to_vec();
                    let node = self
                        .split_child(txn_id, child_id, node.upgrade().await?, i)
                        .await?;

                    if self.collator.compare(&key, &this_key) == Ordering::Greater {
                        let child_id = node.children[i + 1].clone();
                        child = self.file.get_block(txn_id, child_id).await?;
                    }
                }

                self._insert(txn_id, child, key).await
            }
        })
    }

    async fn split_child<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: NodeId,
        mut node: BlockMut<'a, Node>,
        i: usize,
    ) -> TCResult<Block<'a, Node>> {
        let child_id = node.children[i].clone();
        let mut child = self
            .file
            .get_block(txn_id, child_id)
            .await?
            .upgrade()
            .await?;
        let new_node_id = self.file.unique_id(&txn_id).await?;

        node.children.insert(i + 1, new_node_id.clone());
        node.keys.insert(i, child.keys.remove(self.order - 1));

        let mut new_node = Node::new(node.leaf, Some(node_id));
        new_node.keys = child.keys.drain((self.order - 1)..).collect();
        if !child.leaf {
            new_node.children = child.children.drain(self.order..).collect();
        }
        self.file
            .create_block(txn_id.clone(), new_node_id, new_node)
            .await?;
        node.downgrade(&txn_id).await
    }

    pub fn delete<'a>(&'a self, txn_id: &'a TxnId, bounds: Selector) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let root_id = self.root.read(txn_id).await?;
            self._delete(txn_id, (*root_id).clone(), &bounds).await
        })
    }

    fn _delete<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: NodeId,
        bounds: &'a Selector,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let node = self.file.get_block(txn_id, node_id).await?;
            let keys = node.values();
            let (l, r) = bounds.bisect(&keys, &self.collator);

            if node.leaf {
                if l == r {
                    return Ok(());
                }

                let mut node = node.upgrade().await?;
                for i in l..r {
                    node.keys[i].deleted = true;
                }
                node.rebalance = true;

                Ok(())
            } else {
                let children = node.children.to_vec();

                if r > l {
                    let mut node = node.upgrade().await?;
                    let mut deletes = Vec::with_capacity(r - l);
                    for i in l..r {
                        node.keys[i].deleted = true;
                        deletes.push(self._delete(txn_id, children[i].clone(), bounds));
                    }
                    node.rebalance = true;

                    let last_delete = self._delete(txn_id, children[r].clone(), bounds);
                    try_join(try_join_all(deletes), last_delete).await?;
                    Ok(())
                } else {
                    self._delete(txn_id, children[r].clone(), bounds).await
                }
            }
        })
    }
}

impl Instance for BTreeFile {
    type Class = BTreeType;

    fn class(&self) -> BTreeType {
        BTreeType::Tree
    }
}

#[async_trait]
impl CollectionInstance for BTreeFile {
    type Item = Key;
    type Slice = BTreeSlice;

    async fn get(
        &self,
        txn: Txn,
        path: &[PathSegment],
        bounds: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let bounds: Selector =
            bounds.try_cast_into(|v| error::bad_request("Invalid BTree selector", v))?;

        validate_selector(&bounds, self.schema())?;

        if let Selector::Key(key) = bounds {
            let mut slice = self
                .clone()
                .slice(txn.id().clone(), Selector::Key(key.to_vec()))
                .await?;

            if let Some(key) = slice.next().await {
                Ok(CollectionItem::Scalar(key))
            } else {
                Err(error::not_found(Value::Tuple(key)))
            }
        } else {
            Ok(CollectionItem::Slice(BTreeSlice::new(self.clone(), bounds)))
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.get_root(txn.id())
            .await
            .map(|root| root.keys.is_empty())
    }

    async fn put(
        &self,
        txn: Txn,
        path: &[PathSegment],
        selector: Value,
        key: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let key = match key {
            CollectionItem::Scalar(key) => key,
            CollectionItem::Slice(_) => {
                return Err(error::unsupported("BTree::put(<slice>) is not supported"));
            }
        };

        let selector: Selector =
            selector.try_cast_into(|v| error::bad_request("Invalid BTree selector", v))?;

        validate_selector(&selector, self.schema())?;
        validate_key(&key, self.schema())?;

        match selector {
            Selector::Key(selector)
                if self.collator.compare(&selector, &key) == Ordering::Equal =>
            {
                self.insert(txn.id(), key).await
            }
            selector => self.update(txn.id(), &selector, &key).await,
        }
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        let stream = self
            .clone()
            .slice(txn.id().clone(), Selector::all())
            .await?
            .map(Value::Tuple)
            .map(Scalar::Value);

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl Transact for BTreeFile {
    async fn commit(&self, txn_id: &TxnId) {
        join(self.file.commit(txn_id), self.root.commit(txn_id)).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        join(self.file.rollback(txn_id), self.root.rollback(txn_id)).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join(self.file.finalize(txn_id), self.root.finalize(txn_id)).await;
    }
}

impl From<BTreeFile> for Collection {
    fn from(btree_file: BTreeFile) -> Collection {
        Collection::Base(CollectionBase::BTree(btree_file))
    }
}

#[derive(Clone)]
pub struct BTreeRange(Vec<Bound<Value>>, Vec<Bound<Value>>);

impl BTreeRange {
    pub fn all() -> BTreeRange {
        BTreeRange(vec![], vec![])
    }

    fn bisect(&self, keys: &[&[Value]], collator: &collator::Collator) -> (usize, usize) {
        (
            collator.bisect_left_range(keys, &self.0),
            collator.bisect_right_range(keys, &self.1),
        )
    }

    pub fn contains(&self, other: &BTreeRange, schema: &[ValueType]) -> TCResult<bool> {
        if other.0.len() < self.0.len() {
            return Ok(false);
        }

        if other.1.len() < self.1.len() {
            return Ok(false);
        }

        use collator::compare_value;
        use Bound::*;
        use Ordering::*;

        for (dtype, (outer, inner)) in schema[0..self.0.len()]
            .iter()
            .zip(self.0.iter().zip(other.0[0..self.0.len()].iter()))
        {
            match (outer, inner) {
                (Unbounded, _) => {}
                (_, Unbounded) => return Ok(false),
                (Excluded(o), Excluded(i)) if compare_value(&o, &i, *dtype)? == Greater => {
                    return Ok(false)
                }
                (Included(o), Included(i)) if compare_value(&o, &i, *dtype)? == Greater => {
                    return Ok(false)
                }
                (Included(o), Excluded(i)) if compare_value(&o, &i, *dtype)? == Greater => {
                    return Ok(false)
                }
                (Excluded(o), Included(i)) if compare_value(&o, &i, *dtype)? != Less => {
                    return Ok(false)
                }
                _ => {}
            }
        }

        for (dtype, (outer, inner)) in schema[0..self.1.len()]
            .iter()
            .zip(self.1.iter().zip(other.1[0..self.1.len()].iter()))
        {
            match (outer, inner) {
                (Unbounded, _) => {}
                (_, Unbounded) => return Ok(false),
                (Excluded(o), Excluded(i)) if compare_value(&o, &i, *dtype)? == Less => {
                    return Ok(false)
                }
                (Included(o), Included(i)) if compare_value(&o, &i, *dtype)? == Less => {
                    return Ok(false)
                }
                (Included(o), Excluded(i)) if compare_value(&o, &i, *dtype)? == Less => {
                    return Ok(false)
                }
                (Excluded(o), Included(i)) if compare_value(&o, &i, *dtype)? != Greater => {
                    return Ok(false)
                }
                _ => {}
            }
        }

        Ok(true)
    }

    fn start(&self) -> Value {
        Value::Tuple(
            self.0
                .iter()
                .map(|b| match b {
                    Bound::Unbounded => Value::None,
                    Bound::Excluded(v) => v.clone(),
                    Bound::Included(v) => v.clone(),
                })
                .collect(),
        )
    }

    fn end(&self) -> Value {
        Value::Tuple(
            self.1
                .iter()
                .map(|b| match b {
                    Bound::Unbounded => Value::None,
                    Bound::Excluded(v) => v.clone(),
                    Bound::Included(v) => v.clone(),
                })
                .collect(),
        )
    }
}

impl From<Key> for BTreeRange {
    fn from(mut key: Key) -> BTreeRange {
        let start = key.iter().cloned().map(Bound::Included).collect();
        let end = key.drain(..).map(Bound::Included).collect();
        BTreeRange(start, end)
    }
}

impl From<(Vec<Bound<Value>>, Vec<Bound<Value>>)> for BTreeRange {
    fn from(params: (Vec<Bound<Value>>, Vec<Bound<Value>>)) -> BTreeRange {
        BTreeRange(params.0, params.1)
    }
}

#[derive(Clone)]
pub enum Selector {
    Key(Key),
    Range(BTreeRange, bool),
}

impl Selector {
    pub fn all() -> Selector {
        Selector::Range(BTreeRange::all(), false)
    }

    pub fn reverse(range: BTreeRange) -> Selector {
        Selector::Range(range, true)
    }

    fn bisect(&self, keys: &[&[Value]], collator: &collator::Collator) -> (usize, usize) {
        match self {
            Selector::Key(key) => {
                let l = collator.bisect_left(keys, &key);
                let r = collator.bisect_right(keys, &key);
                (l, r)
            }
            Selector::Range(range, _) => range.bisect(keys, collator),
        }
    }
}

impl From<BTreeRange> for Selector {
    fn from(range: BTreeRange) -> Selector {
        Selector::Range(range, false)
    }
}

impl From<Key> for Selector {
    fn from(key: Key) -> Selector {
        Selector::Key(key)
    }
}

impl TryCastFrom<Value> for Selector {
    fn can_cast_from(_value: &Value) -> bool {
        unimplemented!()
    }

    fn opt_cast_from(_value: Value) -> Option<Selector> {
        unimplemented!()
    }
}

impl CastFrom<Selector> for Value {
    fn cast_from(_s: Selector) -> Value {
        unimplemented!()
    }
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Key(key) => write!(f, "key: {}", Value::Tuple(key.to_vec())),
            Self::Range(range, reverse) if *reverse => {
                write!(f, "range: {}, {}", range.end(), range.start())
            }
            Self::Range(range, _) => write!(f, "range: {}, {}", range.start(), range.end()),
        }
    }
}
