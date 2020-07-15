use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{self, join, try_join, try_join_all, BoxFuture, Future};
use futures::stream::{self, FuturesOrdered, Stream, StreamExt, TryStreamExt};
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

impl From<(ValueId, ValueType, Option<usize>)> for Column {
    fn from(column: (ValueId, ValueType, Option<usize>)) -> Column {
        Column {
            name: column.0,
            dtype: column.1,
            max_len: column.2,
        }
    }
}

pub struct Schema(Vec<Column>);

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

    fn validate(&self, selector: &Selector) -> TCResult<()> {
        match selector {
            Selector::Key(key) => self.validate_prefix(key),
            Selector::Range(range, _) => self.validate_range(range),
        }
    }

    fn validate_key(&self, key: &[Value]) -> TCResult<()> {
        if self.len() != key.len() {
            return Err(error::bad_request(
                &format!("Invalid key {} for schema", Value::Vector(key.to_vec())),
                self,
            ));
        }

        self.validate_prefix(key)
    }

    fn validate_prefix(&self, prefix: &[Value]) -> TCResult<()> {
        if prefix.len() > self.len() {
            return Err(error::bad_request(
                &format!(
                    "Invalid selector {} for schema",
                    Value::Vector(prefix.to_vec())
                ),
                self,
            ));
        }

        for (val, col) in prefix.iter().zip(&self.columns()[0..prefix.len()]) {
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

    fn validate_range(&self, range: &BTreeRange) -> TCResult<()> {
        use Bound::*;
        match (&range.0, &range.1) {
            (Unbounded, Unbounded) => Ok(()),
            (Included(start), Included(end)) if start.len() == end.len() => {
                self.validate_dtypes(&start, &end)
            }
            (Included(start), Excluded(end)) if start.len() == end.len() => {
                self.validate_dtypes(&start, &end)
            }
            (Excluded(start), Included(end)) if start.len() == end.len() => {
                self.validate_dtypes(&start, &end)
            }
            (Excluded(start), Excluded(end)) if start.len() == end.len() => {
                self.validate_dtypes(&start, &end)
            }
            _ => Err(error::bad_request(
                "BTree received invalid range",
                "start and end bounds must be the same length",
            )),
        }
    }

    fn validate_dtypes(&self, start: &[Value], end: &[Value]) -> TCResult<()> {
        assert!(start.len() == end.len());

        for ((start_val, end_val), column) in
            start.iter().zip(end).zip(&self.columns()[0..start.len()])
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

impl From<Vec<Column>> for Schema {
    fn from(columns: Vec<Column>) -> Schema {
        Schema(columns)
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
    pub async fn create(txn_id: TxnId, schema: Schema, file: Arc<File>) -> TCResult<BTree> {
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

    async fn get_root(&self, txn_id: &TxnId) -> TCResult<Node> {
        let root_id = self.root.read(txn_id).await?;
        self.get_node(txn_id, &root_id.0).await
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        Ok(self.get_root(txn_id).await?.data.keys.is_empty())
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
        self.schema.validate(&bounds)?;

        let root = self.get_root(&txn_id).await?;

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

    pub async fn update<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Selector,
        value: &'a [Value],
    ) -> TCResult<()> {
        let root_id = self.root.read(txn_id).await?;
        self._update(txn_id, &root_id.0, bounds, value).await
    }

    fn _update<'a>(
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
                        updates.push(self._update(txn_id, &child, bounds, value));
                    }
                    node.sync().await?;

                    let last_update = self._update(txn_id, &children[r], bounds, value);
                    try_join(try_join_all(updates), last_update).await?;
                    Ok(())
                } else {
                    self._update(txn_id, &children[r], bounds, value).await
                }
            }
        })
    }

    pub async fn insert_from<S: Stream<Item = Key>>(
        &self,
        txn_id: &TxnId,
        source: S,
    ) -> TCResult<()> {
        source
            .map(|k| self.schema.validate_key(&k).map(|()| k))
            .map_ok(|key| self.insert(txn_id, key))
            .try_buffer_unordered(2 * self.order)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    pub async fn insert(&self, txn_id: &TxnId, key: Key) -> TCResult<()> {
        let root_id = self.root.read(&txn_id).await?;
        let root = self.get_node(&txn_id, &root_id.0).await?;

        if root.data.keys.len() == (2 * self.order) - 1 {
            let mut root_id = root_id.upgrade().await?;
            let old_root_id = root_id.0.clone();
            let old_root = root;

            root_id.0 = self.file.unique_id(&txn_id).await?;
            let mut new_root =
                Node::create(self.file.clone(), &txn_id, root_id.0.clone(), false, None)
                    .await?
                    .upgrade()
                    .await?;
            new_root.data.children.push(old_root_id.clone());
            self.split_child(&txn_id, old_root_id, old_root.upgrade().await?, 0)
                .await?;

            self._insert(&txn_id, new_root.sync_and_downgrade(&txn_id).await?, key)
                .await
        } else {
            self._insert(&txn_id, root, key).await
        }
    }

    fn _insert<'a>(
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

                self._insert(txn_id, child, key).await
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

    pub async fn delete(&self, txn_id: &TxnId, bounds: Selector) -> TCResult<()> {
        let root_id = self.root.read(txn_id).await?;
        self._delete(txn_id, &root_id.0, &bounds).await
    }

    fn _delete<'a>(
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
                        deletes.push(self._delete(txn_id, &children[i], bounds));
                    }
                    node.data.rebalance = true;
                    node.sync().await?;

                    let last_delete = self._delete(txn_id, &children[r], bounds);
                    try_join(try_join_all(deletes), last_delete).await?;
                    Ok(())
                } else {
                    self._delete(txn_id, &children[r], bounds).await
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
        self.schema.validate(selector)?;
        self.schema.validate_key(&key)?;

        match selector {
            Selector::Key(selector) if self.collator.compare(selector, &key) == Ordering::Equal => {
                self.insert(txn.id(), key).await
            }
            selector => self.update(txn.id(), selector, &key).await,
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
