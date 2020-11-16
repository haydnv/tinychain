use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::ops::{Bound, Deref};
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
use crate::block::{BlockData, BlockId, BlockOwned, BlockOwnedMut};
use crate::class::{Instance, State, TCBoxTryFuture, TCResult, TCStream};
use crate::error;
use crate::request::Request;
use crate::scalar::*;
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

impl Deref for NodeKey {
    type Target = [Value];

    fn deref(&self) -> &[Value] {
        &self.value
    }
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
        write!(
            f,
            "BTree node key: {}{}",
            Value::Tuple(self.value.to_vec()),
            if self.deleted { " (DELETED)" } else { "" }
        )
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
        if self.leaf {
            writeln!(f, "leaf node:")?;
        } else {
            writeln!(f, "non-leaf node:")?;
        }

        write!(
            f,
            "\tkeys: {}",
            self.keys
                .iter()
                .map(|k| k.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )?;

        write!(f, "\t {} children", self.children.len())
    }
}

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

fn validate_range(range: BTreeRange, schema: &[Column]) -> TCResult<BTreeRange> {
    use Bound::*;

    let cast = |(bound, column): (Bound<Value>, &Column)| {
        let value = match bound {
            Unbounded => Unbounded,
            Included(value) => Included(column.dtype().try_cast(value)?),
            Excluded(value) => Excluded(column.dtype().try_cast(value)?),
        };
        Ok(value)
    };

    let cast_range = |range: Vec<Bound<Value>>| {
        range
            .into_iter()
            .zip(schema)
            .map(cast)
            .collect::<TCResult<Vec<Bound<Value>>>>()
    };

    let start = cast_range(range.0)?;
    let end = cast_range(range.1)?;
    Ok(BTreeRange(start, end))
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
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        range: Value,
    ) -> TCResult<State> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let range: BTreeRange =
            range.try_cast_into(|v| error::bad_request("Invalid BTree range", v))?;
        let range = validate_range(range, self.source.schema())?;

        let schema: Vec<ValueType> = self.source.schema().iter().map(|c| *c.dtype()).collect();
        let selector = Selector::from(range); // TODO: support reverse order

        if self.bounds == selector {
            Ok(State::Collection(Collection::View(self.clone().into())))
        } else if self.bounds.range.contains(&selector.range, &schema)? {
            self.source
                .get(request, txn, path, selector.range.cast_into())
                .await
        } else {
            Err(error::bad_request(
                &format!("BTreeSlice[{}] does not contain", &self.bounds),
                &selector,
            ))
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
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _selector: Value,
        _value: State,
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
            root: TxnLock::new("BTree root", root.into()),
        })
    }

    pub fn collator(&'_ self) -> &'_ collator::Collator {
        &self.collator
    }

    pub fn schema(&'_ self) -> &'_ [Column] {
        &self.schema
    }

    pub fn len<'a>(self, txn_id: TxnId, selector: Selector) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            let slice = self.slice(txn_id, selector).await?;
            Ok(slice.fold(0u64, |len, _| future::ready(len + 1)).await)
        })
    }

    pub async fn slice(self, txn_id: TxnId, selector: Selector) -> TCResult<TCStream<Key>> {
        let range = validate_range(selector.range, self.schema())?;

        let root_id = self.root.read(&txn_id).await?;
        let root = self
            .file
            .clone()
            .get_block_owned(txn_id.clone(), (*root_id).clone())
            .await?;

        let slice = if selector.reverse {
            self._slice_reverse(txn_id, root, range)
        } else {
            self._slice(txn_id, root, range)
        };

        Ok(slice)
    }

    fn _slice(self, txn_id: TxnId, node: BlockOwned<Node>, range: BTreeRange) -> TCStream<Key> {
        let (l, r) = range.bisect(&node.keys[..], &self.collator);

        debug!("_slice {} from {} to {}", node.deref(), l, r);

        if node.leaf {
            if l == r && l < node.keys.len() {
                if node.keys[l].deleted {
                    Box::pin(stream::empty())
                } else {
                    Box::pin(stream::once(future::ready(node.keys[l].to_vec())))
                }
            } else {
                let keys: Vec<Key> = node.keys[l..r]
                    .iter()
                    .filter(|k| !k.deleted)
                    .map(|k| k.value.to_vec())
                    .collect();

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
        let (l, r) = range.bisect(&node.keys, &self.collator);

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
        range: BTreeRange,
        value: &'a [Value],
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let range = validate_range(range, self.schema())?;
            let root_id = self.root.read(txn_id).await?;
            self._update(txn_id, &root_id, &range, value).await
        })
    }

    fn _update<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: &'a NodeId,
        range: &'a BTreeRange,
        value: &'a [Value],
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let node = self.file.get_block(txn_id, node_id).await?;
            let (l, r) = range.bisect(&node.keys, &self.collator);

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
                        updates.push(self._update(txn_id, child_id, range, value));
                    }

                    let last_update = self._update(txn_id, &children[r], range, value);
                    try_join(try_join_all(updates), last_update).await?;
                    Ok(())
                } else {
                    self._update(txn_id, &children[r], range, value).await
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
            .and_then(|k| future::ready(validate_key(k, self.schema())))
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
            .map(|k| validate_key(k, self.schema()))
            .map_ok(|key| self.insert(txn_id, key))
            .try_buffer_unordered(2 * self.order)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    pub fn insert<'a>(&'a self, txn_id: &'a TxnId, key: Key) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let root_id = self.root.read(txn_id).await?;
            let root = self
                .file
                .clone()
                .get_block_owned(txn_id.clone(), (*root_id).clone())
                .await?;

            debug!(
                "insert into BTree node with {} keys and {} children (order is {})",
                root.keys.len(),
                root.children.len(),
                self.order
            );

            if root.keys.len() == (2 * self.order) - 1 {
                let mut root_id = root_id.upgrade().await?;
                let old_root_id = (*root_id).clone();

                (*root_id) = self.file.unique_id(&txn_id).await?;
                let mut new_root = Node::new(false, None);
                new_root.children.push(old_root_id.clone());
                let new_root = self
                    .file
                    .clone()
                    .create_block(txn_id.clone(), (*root_id).clone(), new_root)
                    .await?
                    .upgrade()
                    .await?;

                let new_root = self.split_child(txn_id, old_root_id, new_root, 0).await?;
                self._insert(txn_id, new_root, key).await
            } else {
                self._insert(txn_id, root, key).await
            }
        })
    }

    fn _insert<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node: BlockOwned<Node>,
        key: Key,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let i = self.collator.bisect_left(&node.keys, &key);
            if i < node.keys.len() && self.collator.compare(&node.keys[i], &key) == Ordering::Equal
            {
                if node.keys[i].deleted {
                    let mut node = node.upgrade().await?;
                    node.keys[i].deleted = false;
                }

                return Ok(());
            }

            debug!("insert at index {} into {}", i, node.deref());

            if node.leaf {
                let mut node = node.upgrade().await?;
                node.keys.insert(i, key.into());
                Ok(())
            } else {
                let mut child = self
                    .file
                    .clone()
                    .get_block_owned(txn_id.clone(), node.children[i].clone())
                    .await?;

                if child.keys.len() == (2 * self.order) - 1 {
                    let node = self
                        .split_child(txn_id, node.children[i].clone(), node.upgrade().await?, i)
                        .await?;

                    match self.collator.compare(&key, &node.keys[i]) {
                        Ordering::Less => {}
                        Ordering::Equal => {
                            if node.keys[i].deleted {
                                let mut node = node.upgrade().await?;
                                node.keys[i].deleted = false;
                            }

                            return Ok(());
                        }
                        Ordering::Greater => {
                            child = self
                                .file
                                .clone()
                                .get_block_owned(txn_id.clone(), node.children[i + 1].clone())
                                .await?;
                        }
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
        mut node: BlockOwnedMut<Node>,
        i: usize,
    ) -> TCResult<BlockOwned<Node>> {
        let child_id = node.children[i].clone(); // needed due to mutable borrow below
        let mut child = self
            .file
            .get_block(txn_id, &child_id)
            .await?
            .upgrade()
            .await?;

        debug!(
            "child to split has {} keys and {} children",
            child.keys.len(),
            child.children.len()
        );

        let new_node_id = self.file.unique_id(&txn_id).await?;

        node.children.insert(i + 1, new_node_id.clone());
        node.keys.insert(i, child.keys.remove(self.order - 1));

        let mut new_node = Node::new(child.leaf, Some(node_id));
        new_node.keys = child.keys.drain((self.order - 1)..).collect();

        if child.leaf {
            debug!("child is a leaf node");
        } else {
            new_node.children = child.children.drain(self.order..).collect();
        }

        self.file
            .clone()
            .create_block(txn_id.clone(), new_node_id, new_node)
            .await?;

        node.downgrade(&txn_id).await
    }

    pub fn delete<'a>(&'a self, txn_id: &'a TxnId, range: BTreeRange) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let range = validate_range(range, self.schema())?;
            let root_id = self.root.read(txn_id).await?;
            self._delete(txn_id, (*root_id).clone(), &range).await
        })
    }

    fn _delete<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: NodeId,
        range: &'a BTreeRange,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let node = self.file.get_block(txn_id, &node_id).await?;
            let (l, r) = range.bisect(&node.keys, &self.collator);

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
            } else if r > l {
                let mut node = node.upgrade().await?;
                let mut deletes = Vec::with_capacity(r - l);

                for i in 0..node.children.len() {
                    node.keys[i].deleted = true;
                    deletes.push(self._delete(txn_id, node.children[i].clone(), range));
                }
                node.rebalance = true;

                let last_delete = self._delete(txn_id, node.children[r].clone(), range);
                try_join(try_join_all(deletes), last_delete).await?;

                Ok(())
            } else {
                self._delete(txn_id, node.children[r].clone(), range).await
            }
        })
    }

    async fn assert_valid(&self, txn_id: &TxnId) -> TCResult<()> {
        use num::integer::div_ceil;
        use std::collections::VecDeque;

        let root_id = self.root.read(txn_id).await?;
        let root = self.file.get_block(txn_id, &root_id).await?;
        let order = self.order;

        assert!(self.collator.is_sorted(&root.keys));
        assert!(root.children.len() <= 2 * order);
        if !root.leaf {
            assert!(root.children.len() >= 2);
        }

        let mut unvisited: VecDeque<NodeId> = root.children.iter().cloned().collect();
        while let Some(node_id) = unvisited.pop_front() {
            let node = self.file.get_block(txn_id, &node_id).await?;

            assert!(!node.keys.is_empty());
            assert!(self.collator.is_sorted(&node.keys));
            assert!(node.children.len() <= 2 * order);

            if node.leaf {
                assert!(node.children.is_empty());
            } else {
                assert!(node.children.len() == node.keys.len() + 1);
                assert!(node.children.len() >= div_ceil(order, 2));

                for i in 0..node.keys.len() {
                    let child_at_i = self.file.get_block(txn_id, &node.children[i]).await?;

                    let child_after_i = self.file.get_block(txn_id, &node.children[i + 1]).await?;

                    assert!(!child_at_i.keys.is_empty());
                    assert!(!child_after_i.keys.is_empty());
                    assert!(
                        self.collator
                            .compare(child_at_i.keys.last().unwrap(), &node.keys[i])
                            == Ordering::Less
                    );
                    assert!(
                        self.collator.compare(&child_after_i.keys[0], &node.keys[i])
                            == Ordering::Greater
                    );
                }
            }

            unvisited.extend(node.children.iter().cloned());
            debug!("node {} is valid", node_id);
        }

        Ok(())
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
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        range: Value,
    ) -> TCResult<State> {
        let range: BTreeRange =
            range.try_cast_into(|v| error::bad_request("Invalid BTree range", v))?;
        let range = validate_range(range, self.schema())?;

        if path.len() == 1 && &path[0] == "count" {
            return self
                .clone()
                .len(txn.id().clone(), range.into())
                .map_ok(|len| State::Scalar(Number::from(len).into()))
                .await;
        } else if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let slice = Collection::View(BTreeSlice::new(self.clone(), range.into()).into());
        Ok(State::Collection(slice))
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        let root_id = self.root.read(txn.id()).await?;
        let root = self.file.get_block(txn.id(), &root_id).await?;
        Ok(root.keys.is_empty())
    }

    async fn put(
        &self,
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        range: Value,
        key: State,
    ) -> TCResult<()> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let range: BTreeRange =
            range.try_cast_into(|v| error::bad_request("Invalid BTree selector", v))?;
        let range = validate_range(range, self.schema())?;

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

        #[cfg(debug_assertions)]
        self.assert_valid(txn.id()).await?;

        Ok(())
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        let stream = self
            .clone()
            .slice(txn.id().clone(), Selector::default())
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

#[derive(Clone, Eq, PartialEq)]
pub struct BTreeRange(Vec<Bound<Value>>, Vec<Bound<Value>>);

impl BTreeRange {
    pub fn all() -> BTreeRange {
        BTreeRange(vec![], vec![])
    }

    fn bisect<V: Deref<Target = [Value]>>(
        &self,
        keys: &[V],
        collator: &collator::Collator,
    ) -> (usize, usize) {
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

impl Default for BTreeRange {
    fn default() -> BTreeRange {
        BTreeRange(vec![], vec![])
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

impl TryCastFrom<Value> for BTreeRange {
    fn can_cast_from(value: &Value) -> bool {
        value == &Value::None || Key::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<BTreeRange> {
        if value == Value::None {
            Some(BTreeRange::default())
        } else {
            Key::opt_cast_from(value).map(BTreeRange::from)
        }
    }
}

impl CastFrom<BTreeRange> for Value {
    fn cast_from(_s: BTreeRange) -> Value {
        unimplemented!()
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Selector {
    range: BTreeRange,
    reverse: bool,
}

impl Default for Selector {
    fn default() -> Selector {
        Selector {
            range: BTreeRange::default(),
            reverse: false,
        }
    }
}

impl From<BTreeRange> for Selector {
    fn from(range: BTreeRange) -> Selector {
        Selector {
            range,
            reverse: false,
        }
    }
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let start = self.range.start();
        let end = self.range.end();

        if self.reverse {
            write!(f, "range: {}, {}", end, start)
        } else {
            write!(f, "range: {}, {}", start, end)
        }
    }
}
