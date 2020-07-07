use std::cmp::Ordering;
use std::collections::HashSet;
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
use crate::value::{TCResult, TCStream, TCType, Value, ValueId};

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

type Key = Vec<Value>;
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
    dtype: TCType,
    max_len: Option<usize>,
}

pub struct BTree {
    file: Arc<File>,
    schema: Vec<Column>,
    order: usize,
    collator: collator::Collator,
    root: TxnLock<BTreeRoot>,
}

impl BTree {
    // TODO: add `slice` method to iterate over all nodes within a range of keys

    async fn create(txn_id: TxnId, schema: Vec<Column>, file: Arc<File>) -> TCResult<BTree> {
        if !file.is_empty(&txn_id).await? {
            return Err(error::bad_request(
                "Tried to create a new BTree without a new File",
                file,
            ));
        }

        let mut key_size = 0;
        for col in &schema {
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

        let collator = collator::Collator::new(schema.iter().map(|c| c.dtype.clone()).collect())?;
        Ok(BTree {
            file,
            schema,
            order,
            collator,
            root: TxnLock::new(txn_id, BTreeRoot(root)),
        })
    }

    fn validate_key(&self, key: &[Value]) -> TCResult<()> {
        if self.schema.len() != key.len() {
            return Err(error::bad_request(
                &format!("Invalid key {}, expected", Value::Vector(key.to_vec())),
                schema_to_string(&self.schema),
            ));
        }

        self.validate_selector(key)
    }

    fn validate_selector(&self, selector: &[Value]) -> TCResult<()> {
        if selector.len() > self.schema.len() {
            return Err(error::bad_request(
                &format!(
                    "Invalid selector {}, expected",
                    Value::Vector(selector.to_vec())
                ),
                schema_to_string(&self.schema),
            ));
        }

        for (i, val) in selector.iter().enumerate() {
            let column = &self.schema[i];
            if !val.is_a(&column.dtype) {
                return Err(error::bad_request(
                    &format!("Expected {} for", column.dtype),
                    &column.name,
                ));
            }

            let key_size = bincode::serialized_size(&val)?;
            if let Some(size) = column.max_len {
                if key_size as usize > size {
                    return Err(error::bad_request(
                        "Column value exceeds the maximum length",
                        &column.name,
                    ));
                }
            }
        }

        Ok(())
    }

    async fn get_node(&self, txn_id: &TxnId, node_id: &NodeId) -> TCResult<Node> {
        if let Some(block) = self.file.get_block(txn_id, node_id).await? {
            Node::try_from(block).await
        } else {
            Err(error::internal(ERR_CORRUPT))
        }
    }

    fn select(self: Arc<Self>, txn_id: TxnId, node: Node, key: Key) -> TCStream<Key> {
        let keys = node.data.values();
        let l = self.collator.bisect_left(&keys, &key);
        let r = self.collator.bisect(&keys, &key);

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
                let index = self.clone();
                let child = node.data.children[i].clone();
                let txn_id = txn_id.clone();
                let key = key.clone();

                let selection = Box::pin(async move {
                    let node = index.get_node(&txn_id, &child).await.unwrap();
                    index.select(txn_id.clone(), node, key)
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
                self.select(txn_id, node, key)
            });
            selected.push(Box::pin(selection));

            Box::pin(selected.flatten())
        }
    }

    fn update<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: &'a NodeId,
        selector: &'a [Value],
        value: &'a [Value],
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let node = self.get_node(txn_id, node_id).await?;
            let keys = node.data.values();
            let l = self.collator.bisect_left(&keys, &selector);
            let r = self.collator.bisect(&keys, &selector);

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
                        updates.push(self.update(txn_id, &child, selector, value));
                    }
                    node.sync().await?;

                    let last_update = self.update(txn_id, &children[r], selector, value);
                    try_join(try_join_all(updates), last_update).await?;
                    Ok(())
                } else {
                    self.update(txn_id, &children[r], selector, value).await
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
        let new_node_id = new_node_id(self.file.block_ids(txn_id).await?);

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
        key: Key,
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let node = self.get_node(txn_id, node_id).await?;
            let keys = node.data.values();
            let l = self.collator.bisect_left(&keys, &key);
            let r = self.collator.bisect(&keys, &key);

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
                        deletes.push(self.delete(txn_id, &children[i], key.clone()));
                    }
                    node.data.rebalance = true;
                    node.sync().await?;

                    let last_delete = self.delete(txn_id, &children[r], key);
                    try_join(try_join_all(deletes), last_delete).await?;
                    Ok(())
                } else {
                    self.delete(txn_id, &children[r], key).await
                }
            }
        })
    }
}

#[async_trait]
impl Collect for BTree {
    type Selector = Key;
    type Item = Key;

    async fn get(self: Arc<Self>, txn: Arc<Txn>, selector: Self::Selector) -> GetResult {
        self.validate_key(&selector)?;
        let root_id = &self.root.read(txn.id()).await?.0;
        let root_node = self.get_node(txn.id(), root_id).await?;
        Ok(Box::pin(
            self.clone()
                .select(txn.id().clone(), root_node, selector)
                .map(|row| State::Value(row.into())),
        ))
    }

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        key: Self::Item,
    ) -> TCResult<()> {
        self.validate_selector(selector)?;
        self.validate_key(&key)?;

        let root_id = self.root.read(txn.id()).await?;

        if self.collator.compare(selector, &key) == Ordering::Equal {
            let root = self.get_node(txn.id(), &root_id.0).await?;

            if root.data.keys.len() == (2 * self.order) - 1 {
                let mut root_id = root_id.upgrade().await?;
                let old_root_id = root_id.0.clone();
                let old_root = root;

                root_id.0 = new_node_id(self.file.block_ids(txn.id()).await?);
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
        } else {
            self.update(txn.id(), &root_id.0, &selector, &key).await
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

fn schema_to_string(schema: &[Column]) -> String {
    schema
        .iter()
        .map(|c| c.dtype.to_string())
        .collect::<Vec<String>>()
        .join(",")
}

fn new_node_id(existing_ids: HashSet<NodeId>) -> NodeId {
    loop {
        let id: ValueId = Uuid::new_v4().into();
        if !existing_ids.contains(&id) {
            return id;
        }
    }
}
