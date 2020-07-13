use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{self, join, try_join, try_join_all, BoxFuture, Future};
use futures::stream::{self, FuturesOrdered, StreamExt};
use futures::try_join;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error;
use crate::state::file::{Block, BlockId, File};
use crate::transaction::lock::{Mutate, TxnLock, TxnLockReadGuard, TxnLockWriteGuard};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::class::{Impl, ValueClass, ValueType};
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::{Collect, GetResult, State};

mod collator;

const DEFAULT_BLOCK_SIZE: usize = 4_000;
const BLOCK_ID_SIZE: usize = 128; // UUIDs are 128-bit
const ERR_CORRUPT: &str = "BTree corrupted! Please restart Tinychain and file a bug report";

type NodeId = BlockId;

#[derive(Deserialize, Serialize)]
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

#[derive(Deserialize, Serialize)]
struct NodeData {
    leaf: bool,
    keys: Vec<NodeKey>,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    rebalance: bool, // TODO: implement rebalancing to clear deleted values
}

impl NodeData {
    fn new(leaf: bool, parent: Option<NodeId>) -> NodeData {
        NodeData {
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

struct Node {
    block: TxnLockReadGuard<Block>,
    data: NodeData,
}

impl Node {
    async fn create(
        file: Arc<File>,
        txn_id: &TxnId,
        node_id: NodeId,
        leaf: bool,
        parent: Option<NodeId>,
    ) -> TCResult<Node> {
        Self::try_from(
            file.create_block(
                txn_id,
                node_id,
                bincode::serialize(&NodeData::new(leaf, parent))?.into(),
            )
            .await?,
        )
        .await
    }

    async fn try_from(block: TxnLockReadGuard<Block>) -> TCResult<Node> {
        let data = bincode::deserialize(&block.as_bytes().await)?;
        Ok(Node { block, data })
    }

    async fn upgrade(self) -> TCResult<NodeMut> {
        Ok(NodeMut {
            block: self.block.upgrade().await?,
            data: self.data,
        })
    }
}

struct NodeMut {
    block: TxnLockWriteGuard<Block>,
    data: NodeData,
}

impl NodeMut {
    async fn sync(&self) -> TCResult<()> {
        self.block
            .rewrite(bincode::serialize(&self.data)?.into())
            .await;

        Ok(())
    }

    async fn sync_and_downgrade(self, txn_id: &TxnId) -> TCResult<Node> {
        self.block
            .rewrite(bincode::serialize(&self.data)?.into())
            .await;

        Ok(Node {
            block: self.block.downgrade(txn_id).await?,
            data: self.data,
        })
    }
}

pub type Key = Vec<Value>;
type Selection = FuturesOrdered<Pin<Box<dyn Future<Output = TCStream<Key>> + Send + Sync + Unpin>>>;

#[derive(Clone)]
struct BTreeRoot(NodeId);

#[async_trait]
impl Mutate for BTreeRoot {
    fn diverge(&self, _txn_id: &TxnId) -> Self {
        self.clone()
    }

    async fn converge(&mut self, new_value: BTreeRoot) {
        self.0 = new_value.0
    }
}

pub struct Column {
    name: ValueId,
    dtype: ValueType,
    max_len: Option<usize>,
}

struct Schema(Vec<Column>);

impl Schema {
    fn columns(&'_ self) -> &'_ [Column] {
        &self.0
    }

    fn dtypes(&self) -> Vec<ValueType> {
        self.0.iter().map(|c| c.dtype).collect()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .iter()
                .map(|c| c.dtype.to_string())
                .collect::<Vec<String>>()
                .join(",")
        )
    }
}

pub struct BTree {
    file: Arc<File>,
    schema: Schema,
    order: usize,
    collator: collator::Collator,
    root: TxnLock<BTreeRoot>,
}

impl BTree {
    async fn create(txn_id: TxnId, schema: Schema, file: Arc<File>) -> TCResult<BTree> {
        if !file.is_empty(&txn_id).await? {
            return Err(error::bad_request(
                "Tried to create a new BTree without a new File",
                file,
            ));
        }

        let mut key_size = 0;
        for col in schema.columns() {
            if let Some(size) = col.dtype.size() {
                key_size += size;
                if col.max_len.is_some() {
                    return Err(error::bad_request(
                        "Found maximum length specified for a scalar type",
                        &col.dtype,
                    ));
                }
            } else if let Some(size) = col.max_len {
                key_size += size + 8; // add 8 bytes for bincode to encode the length
            } else {
                return Err(error::bad_request(
                    "Type requires a maximum length",
                    &col.dtype,
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
            .create_block(
                &txn_id,
                root.clone(),
                Bytes::from(bincode::serialize(&NodeData::new(true, None))?),
            )
            .await?;

        let collator = collator::Collator::new(schema.dtypes())?;
        Ok(BTree {
            file,
            schema,
            order,
            collator,
            root: TxnLock::new(txn_id, BTreeRoot(root)),
        })
    }

    async fn get_node(&self, txn_id: &TxnId, node_id: &NodeId) -> TCResult<Node> {
        if let Some(block) = self.file.get_block(txn_id, node_id).await? {
            Node::try_from(block).await
        } else {
            Err(error::internal(ERR_CORRUPT))
        }
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        let root_id = self.root.read(txn_id).await?;
        let root = self.get_node(txn_id, &root_id.0).await?;
        Ok(root.data.keys.is_empty())
    }

    pub async fn len(self: Arc<Self>, txn_id: TxnId, bounds: Selector) -> TCResult<u64> {
        // TODO: stream nodes directly rather than counting each key
        Ok(self
            .slice(txn_id, bounds)
            .await?
            .fold(0u64, |len, _| future::ready(len + 1))
            .await)
    }

    pub async fn slice(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Selector,
    ) -> TCResult<TCStream<Key>> {
        bounds.validate(&self.schema)?;

        let root_id = self.root.read(&txn_id).await?;
        let root = self.get_node(&txn_id, &root_id.0).await?;

        Ok(match bounds {
            Selector::Key(key) => self._slice(txn_id, root, key.into()),
            Selector::Range(range, false) => self._slice(txn_id, root, range),
            Selector::Range(range, true) => self._slice_reverse(txn_id, root, range),
        })
    }

    fn _slice(self: Arc<Self>, txn_id: TxnId, node: Node, range: BTreeRange) -> TCStream<Key> {
        let keys = node.data.values();
        let (l, r) = range.bisect(&keys, &self.collator);

        if node.data.leaf {
            let keys: Vec<Key> = node.data.keys[l..r]
                .iter()
                .filter(|k| !k.deleted)
                .map(|k| k.value.to_vec())
                .collect();
            Box::pin(stream::iter(keys))
        } else {
            let mut selected: Selection = FuturesOrdered::new();
            for i in l..r {
                let this = self.clone();
                let child = node.data.children[i].clone();
                let txn_id = txn_id.clone();
                let bounds = range.clone();

                let selection = Box::pin(async move {
                    let node = this.get_node(&txn_id, &child).await.unwrap();
                    this._slice(txn_id.clone(), node, bounds)
                });
                selected.push(Box::pin(selection));

                if !node.data.keys[i].deleted {
                    let key_at_i = node.data.keys[i].value.to_vec();
                    let key_at_i: TCStream<Key> = Box::pin(stream::once(future::ready(key_at_i)));
                    selected.push(Box::pin(future::ready(key_at_i)));
                }
            }

            let last_child = node.data.children[r].clone();
            let selection = Box::pin(async move {
                let node = self.get_node(&txn_id, &last_child).await.unwrap();
                self._slice(txn_id, node, range)
            });
            selected.push(Box::pin(selection));

            Box::pin(selected.flatten())
        }
    }

    fn _slice_reverse(
        self: Arc<Self>,
        txn_id: TxnId,
        node: Node,
        range: BTreeRange,
    ) -> TCStream<Key> {
        let keys = node.data.values();
        let (l, r) = range.bisect(&keys, &self.collator);

        if node.data.leaf {
            let keys: Vec<Key> = node.data.keys[l..r]
                .iter()
                .filter(|k| !k.deleted)
                .map(|k| k.value.to_vec())
                .rev()
                .collect();
            Box::pin(stream::iter(keys))
        } else {
            let mut selected: Selection = FuturesOrdered::new();

            let last_child = node.data.children[r].clone();
            let this = self.clone();
            let txn_id_clone = txn_id.clone();
            let range_clone = range.clone();
            let selection = Box::pin(async move {
                let node = this.get_node(&txn_id_clone, &last_child).await.unwrap();
                this._slice_reverse(txn_id_clone, node, range_clone)
            });
            selected.push(Box::pin(selection));

            for i in (r..l).rev() {
                let this = self.clone();
                let child = node.data.children[i].clone();
                let txn_id = txn_id.clone();
                let bounds = range.clone();

                let selection = Box::pin(async move {
                    let node = this.get_node(&txn_id, &child).await.unwrap();
                    this._slice_reverse(txn_id.clone(), node, bounds)
                });
                selected.push(Box::pin(selection));

                if !node.data.keys[i].deleted {
                    let key_at_i = node.data.keys[i].value.to_vec();
                    let key_at_i: TCStream<Key> = Box::pin(stream::once(future::ready(key_at_i)));
                    selected.push(Box::pin(future::ready(key_at_i)));
                }
            }

            Box::pin(selected.flatten())
        }
    }

    fn update<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: &'a NodeId,
        bounds: &'a Selector,
        value: &'a [Value],
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let node = self.get_node(txn_id, node_id).await?;
            let keys = node.data.values();
            let (l, r) = bounds.bisect(&keys, &self.collator);

            if node.data.leaf {
                if l == r {
                    return Ok(());
                }

                let mut node = node.upgrade().await?;
                for i in l..r {
                    node.data.keys[i] = value.into();
                }
                node.sync().await
            } else {
                let children = node.data.children.to_vec();

                if r > l {
                    let mut updates = Vec::with_capacity(r - l);
                    let mut node = node.upgrade().await?;
                    for (i, child) in children.iter().enumerate().take(r).skip(l) {
                        node.data.keys[i] = value.into();
                        updates.push(self.update(txn_id, &child, bounds, value));
                    }
                    node.sync().await?;

                    let last_update = self.update(txn_id, &children[r], bounds, value);
                    try_join(try_join_all(updates), last_update).await?;
                    Ok(())
                } else {
                    self.update(txn_id, &children[r], bounds, value).await
                }
            }
        })
    }

    fn insert<'a>(
        &'a self,
        txn_id: &'a TxnId,
        mut node: Node,
        key: Key,
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let keys = &node.data.keys;
            let i = self.collator.bisect_left(&node.data.values(), &key);
            if node.data.leaf {
                if i == node.data.keys.len()
                    || self.collator.compare(&keys[i].value, &key) != Ordering::Equal
                {
                    let mut node = node.upgrade().await?;
                    node.data.keys.insert(i, key.into());
                    node.sync().await
                } else if keys[i].value == key && keys[i].deleted {
                    let mut node = node.upgrade().await?;
                    node.data.keys[i].deleted = false;
                    node.sync().await
                } else {
                    Ok(())
                }
            } else {
                let child_id = &node.data.children[i];
                let mut child = self.get_node(txn_id, child_id).await?;
                if child.data.keys.len() == (2 * self.order) - 1 {
                    let this_key = &node.data.keys[i].value.to_vec();
                    node = self
                        .split_child(txn_id, child_id.clone(), node.upgrade().await?, i)
                        .await?;
                    if self.collator.compare(&key, &this_key) == Ordering::Greater {
                        child = self.get_node(txn_id, &node.data.children[i + 1]).await?;
                    }
                }

                self.insert(txn_id, child, key).await
            }
        })
    }

    async fn split_child(
        &self,
        txn_id: &TxnId,
        node_id: NodeId,
        mut node: NodeMut,
        i: usize,
    ) -> TCResult<Node> {
        let mut child = self.get_node(txn_id, &node.data.children[i]).await?;
        let new_node_id = self.file.unique_id(txn_id).await?;

        node.data.children.insert(i + 1, new_node_id.clone());
        node.data
            .keys
            .insert(i, child.data.keys.remove(self.order - 1));
        let mut new_node = Node::create(
            self.file.clone(),
            txn_id,
            new_node_id.clone(),
            node.data.leaf,
            Some(node_id),
        )
        .await?
        .upgrade()
        .await?;
        new_node.data.keys = child.data.keys.drain((self.order - 1)..).collect();

        if !child.data.leaf {
            new_node.data.children = child.data.children.drain(self.order..).collect();
        }

        let (_, node) = try_join!(new_node.sync(), node.sync_and_downgrade(txn_id))?;

        Ok(node)
    }

    fn delete<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: &'a NodeId,
        bounds: &'a Selector,
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let node = self.get_node(txn_id, node_id).await?;
            let keys = node.data.values();
            let (l, r) = bounds.bisect(&keys, &self.collator);

            if node.data.leaf {
                if l == r {
                    return Ok(());
                }

                let mut node = node.upgrade().await?;
                for i in l..r {
                    node.data.keys[i].deleted = true;
                }
                node.data.rebalance = true;
                node.sync().await
            } else {
                let children = node.data.children.to_vec();

                if r > l {
                    let mut deletes = Vec::with_capacity(r - l);
                    let mut node = node.upgrade().await?;
                    for i in l..r {
                        node.data.keys[i].deleted = true;
                        deletes.push(self.delete(txn_id, &children[i], bounds));
                    }
                    node.data.rebalance = true;
                    node.sync().await?;

                    let last_delete = self.delete(txn_id, &children[r], bounds);
                    try_join(try_join_all(deletes), last_delete).await?;
                    Ok(())
                } else {
                    self.delete(txn_id, &children[r], bounds).await
                }
            }
        })
    }
}

#[derive(Clone)]
pub struct BTreeRange(Bound<Vec<Value>>, Bound<Vec<Value>>);

impl BTreeRange {
    fn all() -> BTreeRange {
        BTreeRange(Bound::Unbounded, Bound::Unbounded)
    }

    fn bisect(&self, keys: &[&[Value]], collator: &collator::Collator) -> (usize, usize) {
        use Bound::*;
        let l = match &self.0 {
            Unbounded => 0,
            Included(start) => collator.bisect_left(keys, &start),
            Excluded(start) => collator.bisect_right(keys, &start),
        };

        let r = match &self.1 {
            Unbounded => keys.len(),
            Included(end) => collator.bisect_right(keys, &end),
            Excluded(end) => collator.bisect_left(keys, &end),
        };

        (l, r)
    }

    fn validate(&self, schema: &Schema) -> TCResult<()> {
        use Bound::*;
        match (&self.0, &self.1) {
            (Unbounded, Unbounded) => Ok(()),
            (Included(start), Included(end)) if start.len() == end.len() => {
                Self::validate_dtypes(schema, &start, &end)
            }
            (Included(start), Excluded(end)) if start.len() == end.len() => {
                Self::validate_dtypes(schema, &start, &end)
            }
            (Excluded(start), Included(end)) if start.len() == end.len() => {
                Self::validate_dtypes(schema, &start, &end)
            }
            (Excluded(start), Excluded(end)) if start.len() == end.len() => {
                Self::validate_dtypes(schema, &start, &end)
            }
            _ => Err(error::bad_request(
                "BTree received invalid range",
                "start and end bounds must be the same length",
            )),
        }
    }

    fn validate_dtypes(schema: &Schema, start: &[Value], end: &[Value]) -> TCResult<()> {
        assert!(start.len() == end.len());

        for ((start_val, end_val), column) in
            start.iter().zip(end).zip(&schema.columns()[0..start.len()])
        {
            if start_val.class() != end_val.class() {
                return Err(error::bad_request(
                    &format!(
                        "BTreeRange expected [{}, {}] but found",
                        column.dtype, column.dtype
                    ),
                    format!("[{}, {}]", start_val.class(), end_val.class()),
                ));
            }

            if start_val.class() != column.dtype {
                return Err(error::bad_request(
                    &format!("BTreeRange expected {} but found", column.dtype),
                    start_val.class(),
                ));
            }
        }

        Ok(())
    }
}

impl From<Key> for BTreeRange {
    fn from(key: Key) -> BTreeRange {
        BTreeRange(Bound::Included(key.clone()), Bound::Included(key))
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

    fn validate(&self, schema: &Schema) -> TCResult<()> {
        match self {
            Self::Key(key) => Selector::validate_key(key, schema),
            Self::Range(range, _) => range.validate(schema),
        }
    }

    fn validate_key(key: &[Value], schema: &Schema) -> TCResult<()> {
        if schema.len() != key.len() {
            return Err(error::bad_request(
                &format!("Invalid key {}, expected", Value::Vector(key.to_vec())),
                schema,
            ));
        }

        Selector::validate_selector(key, schema)
    }

    fn validate_selector(selector: &[Value], schema: &Schema) -> TCResult<()> {
        if selector.len() > schema.len() {
            return Err(error::bad_request(
                &format!(
                    "Invalid selector {}, expected",
                    Value::Vector(selector.to_vec())
                ),
                schema,
            ));
        }

        for (val, col) in selector.iter().zip(&schema.columns()[0..selector.len()]) {
            if !val.is_a(col.dtype) {
                return Err(error::bad_request(
                    &format!("Expected {} for", col.dtype),
                    &col.name,
                ));
            }

            let key_size = bincode::serialized_size(&val)?;
            if let Some(size) = col.max_len {
                if key_size as usize > size {
                    return Err(error::bad_request(
                        "Column value exceeds the maximum length",
                        &col.name,
                    ));
                }
            }
        }

        Ok(())
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

impl TryFrom<Value> for Selector {
    type Error = error::TCError;

    fn try_from(_value: Value) -> TCResult<Selector> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Collect for BTree {
    type Selector = Selector;
    type Item = Key;

    async fn get(self: Arc<Self>, txn: Arc<Txn>, bounds: Selector) -> GetResult {
        Ok(Box::pin(
            self.slice(txn.id().clone(), bounds)
                .await?
                .map(|row| State::Value(row.into())),
        ))
    }

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        key: Self::Item,
    ) -> TCResult<()> {
        selector.validate(&self.schema)?;
        Selector::validate_key(&key, &self.schema)?;

        let root_id = self.root.read(txn.id()).await?;

        match selector {
            Selector::Key(selector) if self.collator.compare(selector, &key) == Ordering::Equal => {
                let root = self.get_node(txn.id(), &root_id.0).await?;

                if root.data.keys.len() == (2 * self.order) - 1 {
                    let mut root_id = root_id.upgrade().await?;
                    let old_root_id = root_id.0.clone();
                    let old_root = root;

                    root_id.0 = self.file.unique_id(txn.id()).await?;
                    let mut new_root =
                        Node::create(self.file.clone(), txn.id(), root_id.0.clone(), false, None)
                            .await?
                            .upgrade()
                            .await?;
                    new_root.data.children.push(old_root_id.clone());
                    self.split_child(txn.id(), old_root_id, old_root.upgrade().await?, 0)
                        .await?;

                    self.insert(txn.id(), new_root.sync_and_downgrade(txn.id()).await?, key)
                        .await
                } else {
                    self.insert(txn.id(), root, key).await
                }
            }
            selector => self.update(txn.id(), &root_id.0, selector, &key).await,
        }
    }
}

#[async_trait]
impl Transact for BTree {
    async fn commit(&self, txn_id: &TxnId) {
        join(self.file.commit(txn_id), self.root.commit(txn_id)).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        join(self.file.rollback(txn_id), self.root.rollback(txn_id)).await;
    }
}
