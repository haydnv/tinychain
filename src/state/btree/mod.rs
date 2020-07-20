use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::ops::Bound;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{self, join, try_join, try_join_all, BoxFuture, Future, TryFutureExt};
use futures::stream::{self, FuturesOrdered, Stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error;
use crate::state::file::{Block, BlockData, BlockId, BlockMut, BlockOwned, File};
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::class::{Impl, ValueClass, ValueType};
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::{Collect, GetResult, State};

mod collator;

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

impl BlockData for Node {}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(BTree node)")
    }
}

pub type Key = Vec<Value>;
type Selection = FuturesOrdered<Pin<Box<dyn Future<Output = TCStream<Key>> + Send + Sync + Unpin>>>;

#[derive(Clone)]
struct BTreeRoot(NodeId);

#[async_trait]
impl Mutate for BTreeRoot {
    type Pending = NodeId;

    fn diverge(&self, _txn_id: &TxnId) -> Self::Pending {
        self.0.clone()
    }

    async fn converge(&mut self, new_value: Self::Pending) {
        self.0 = new_value;
    }
}

#[derive(Eq, PartialEq)]
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

#[derive(Eq, PartialEq)]
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

        let expect = |column: &Column, value: &Value| {
            value.expect(
                column.dtype,
                format!("for column {} in BTreeRange", column.name),
            )
        };

        for (i, column) in self.columns().iter().enumerate() {
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
    file: Arc<File<Node>>,
    schema: Schema,
    order: usize,
    collator: collator::Collator,
    root: TxnLock<BTreeRoot>,
}

impl BTree {
    pub async fn create(txn_id: TxnId, schema: Schema, file: Arc<File<Node>>) -> TCResult<BTree> {
        if !file.is_empty(&txn_id).await? {
            return Err(error::internal(
                "Tried to create a new BTree without a new File",
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
            .create_block(txn_id.clone(), root.clone(), Node::new(true, None))
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

    pub fn collator(&'_ self) -> &'_ collator::Collator {
        &self.collator
    }

    pub fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    async fn get_root<'a>(&'a self, txn_id: &'a TxnId) -> TCResult<Block<'a, Node>> {
        self.root
            .read(txn_id)
            .and_then(|root_id| self.file.get_block(txn_id, (*root_id).clone()))
            .await
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.get_root(txn_id).await.map(|root| root.keys.is_empty())
    }

    pub async fn len(self: Arc<Self>, txn_id: TxnId, selector: Selector) -> TCResult<u64> {
        Ok(self
            .slice(txn_id, selector)
            .await?
            .fold(0u64, |len, _| future::ready(len + 1))
            .await)
    }

    pub async fn slice(
        self: Arc<Self>,
        txn_id: TxnId,
        selector: Selector,
    ) -> TCResult<TCStream<Key>> {
        self.schema.validate(&selector)?;

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

    fn _slice(
        self: Arc<Self>,
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
                .collect();
            Box::pin(stream::iter(keys))
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
        self: Arc<Self>,
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

    pub async fn update<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Selector,
        value: &'a [Value],
    ) -> TCResult<()> {
        let root_id = self.root.read(txn_id).await?;
        self._update(txn_id, (*root_id).clone(), bounds, value)
            .await
    }

    fn _update<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: NodeId,
        bounds: &'a Selector,
        value: &'a [Value],
    ) -> BoxFuture<'a, TCResult<()>> {
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
    }

    fn _insert<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node: Block<'a, Node>,
        key: Key,
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let keys = &node.keys;
            let i = self.collator.bisect_left(&node.values(), &key);
            if node.leaf {
                if i == node.keys.len()
                    || self.collator.compare(&keys[i].value, &key) != Ordering::Equal
                {
                    let mut node = node.upgrade().await?;
                    node.keys.insert(i, key.into());
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

    pub async fn delete(&self, txn_id: &TxnId, bounds: Selector) -> TCResult<()> {
        let root_id = self.root.read(txn_id).await?;
        self._delete(txn_id, (*root_id).clone(), &bounds).await
    }

    fn _delete<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: NodeId,
        bounds: &'a Selector,
    ) -> BoxFuture<'a, TCResult<()>> {
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

    pub fn contains(&self, other: &BTreeRange, schema: Vec<ValueType>) -> TCResult<bool> {
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
            self.clone()
                .slice(txn.id().clone(), bounds)
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
