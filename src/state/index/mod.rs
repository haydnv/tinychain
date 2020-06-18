use std::cmp::Ordering;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{self, BoxFuture, Future};
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
const ERR_CORRUPT: &str = "Index corrupted! Please restart Tinychain and file a bug report";

type NodeId = BlockId;

// TODO: have node retain a TxnLock[Read|Write]Guard, for locking
#[derive(Deserialize, Serialize)]
struct NodeData {
    leaf: bool,
    keys: Vec<Vec<Value>>,
    children: Vec<NodeId>,
}

impl NodeData {
    fn new(leaf: bool) -> NodeData {
        NodeData {
            leaf,
            keys: vec![],
            children: vec![],
        }
    }
}

struct Node {
    block: TxnLockReadGuard<Block>,
    data: NodeData,
}

impl Node {
    async fn create(file: Arc<File>, txn_id: TxnId, node_id: NodeId, leaf: bool) -> TCResult<Node> {
        Self::try_from(
            file.create_block(
                txn_id,
                node_id,
                bincode::serialize(&NodeData::new(leaf))?.into(),
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

    async fn sync_if_updated(self, updated: bool) -> TCResult<()> {
        if updated {
            self.upgrade().await?.sync().await
        } else {
            Ok(())
        }
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

    async fn sync_and_downgrade(self) -> TCResult<Node> {
        self.block
            .rewrite(bincode::serialize(&self.data)?.into())
            .await;
        Ok(Node {
            block: self.block.downgrade().await?,
            data: self.data,
        })
    }
}

type Key = Vec<Value>;
type Selection = FuturesOrdered<Pin<Box<dyn Future<Output = TCStream<Key>> + Send + Sync + Unpin>>>;

#[derive(Clone)]
struct IndexRoot(NodeId);

#[async_trait]
impl Mutate for IndexRoot {
    fn diverge(&self, _txn_id: &TxnId) -> Self {
        self.clone()
    }

    async fn converge(&mut self, new_value: IndexRoot) {
        self.0 = new_value.0
    }
}

pub struct Column {
    name: ValueId,
    dtype: TCType,
    max_len: Option<usize>,
}

pub struct Index {
    file: Arc<File>,
    schema: Vec<Column>,
    order: usize,
    collator: collator::Collator,
    root: TxnLock<IndexRoot>,
}

impl Index {
    async fn create(txn_id: TxnId, schema: Vec<Column>, file: Arc<File>) -> TCResult<Index> {
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
        // the "leaf" boolean adds 1 byte to each key as-stored
        key_size += 1;

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

        if file.is_empty(txn_id.clone()).await? {
            let root: BlockId = Uuid::new_v4().into();
            file.clone()
                .create_block(
                    txn_id.clone(),
                    root.clone(),
                    Bytes::from(bincode::serialize(&NodeData::new(true))?),
                )
                .await?;

            let collator =
                collator::Collator::new(schema.iter().map(|c| c.dtype.clone()).collect())?;
            Ok(Index {
                file,
                schema,
                order,
                collator,
                root: TxnLock::new(txn_id, IndexRoot(root)),
            })
        } else {
            Err(error::bad_request(
                "Tried to create a new Index without a new File",
                file,
            ))
        }
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

    async fn get_node(&self, txn_id: TxnId, node_id: &NodeId) -> TCResult<Node> {
        if let Some(block) = self.file.get_block(txn_id, node_id).await? {
            Node::try_from(block).await
        } else {
            Err(error::internal(ERR_CORRUPT))
        }
    }

    fn select(self: Arc<Self>, txn_id: TxnId, mut node: Node, key: Key) -> TCStream<Key> {
        let l = self.collator.bisect_left(&node.data.keys, &key);
        let r = self.collator.bisect(&node.data.keys, &key);

        if node.data.leaf {
            let keys: Vec<Key> = node.data.keys.drain(l..r).collect();
            Box::pin(stream::iter(keys))
        } else {
            let mut selected: Selection = FuturesOrdered::new();
            for i in l..r {
                let index = self.clone();
                let child = node.data.children[i].clone();
                let txn_id = txn_id.clone();
                let key = key.clone();

                let selection = Box::pin(async move {
                    let node = index.get_node(txn_id.clone(), &child).await.unwrap();
                    index.select(txn_id, node, key)
                });
                selected.push(Box::pin(selection));

                let key_at_i = node.data.keys[i].clone(); // TODO: drain the node instead of cloning its keys
                let key_at_i: TCStream<Key> = Box::pin(stream::once(future::ready(key_at_i)));
                selected.push(Box::pin(future::ready(key_at_i)));
            }

            let last_child = node.data.children[r].clone();
            let selection = Box::pin(async move {
                let node = self.get_node(txn_id.clone(), &last_child).await.unwrap();
                self.select(txn_id, node, key)
            });
            selected.push(Box::pin(selection));

            Box::pin(selected.flatten())
        }
    }

    fn update<'a>(
        &'a self,
        txn_id: TxnId,
        node_id: &'a NodeId,
        selector: &'a [Value],
        value: &'a [Value],
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let mut node = self.get_node(txn_id.clone(), node_id).await?;
            let l = self.collator.bisect_left(&node.data.keys, &selector);
            let r = self.collator.bisect(&node.data.keys, &selector);
            let mut updated = false;

            if node.data.leaf {
                for i in l..r {
                    node.data.keys[i] = value.to_vec();
                    updated = true;
                }

                node.sync_if_updated(updated).await
            } else {
                for i in l..r {
                    self.update(txn_id.clone(), &node.data.children[i], selector, value)
                        .await?;
                    node.data.keys[i] = value.to_vec();
                    updated = true;
                }

                let last_child = node.data.children.last().unwrap().clone();
                node.sync_if_updated(updated).await?;
                self.update(txn_id, &last_child, selector, value).await
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
            let i = self.collator.bisect_left(&node.data.keys, &key);
            if node.data.leaf {
                if i == node.data.keys.len()
                    || self.collator.compare(&node.data.keys[i], &key) != Ordering::Equal
                {
                    let mut node = node.upgrade().await?;
                    node.data.keys.insert(i, key);
                    node.sync().await
                } else {
                    Ok(())
                }
            } else {
                let mut child = self
                    .get_node(txn_id.clone(), &node.data.children[i])
                    .await?;
                if child.data.keys.len() == (2 * self.order) - 1 {
                    let this_key = &node.data.keys[i].clone();
                    node = self
                        .split_child(txn_id.clone(), node.upgrade().await?, i)
                        .await?;
                    if self.collator.compare(&key, &this_key) == Ordering::Greater {
                        child = self
                            .get_node(txn_id.clone(), &node.data.children[i + 1])
                            .await?;
                    }
                }

                self.insert(txn_id, child, key).await
            }
        })
    }

    async fn split_child(&self, txn_id: TxnId, mut node: NodeMut, i: usize) -> TCResult<Node> {
        let mut child = self
            .get_node(txn_id.clone(), &node.data.children[i])
            .await?;
        let new_node_id = new_node_id(self.file.block_ids(txn_id.clone()).await?);

        node.data.children.insert(i + 1, new_node_id.clone());
        node.data
            .keys
            .insert(i, child.data.keys.remove(self.order - 1));
        let mut new_node = Node::create(
            self.file.clone(),
            txn_id,
            new_node_id.clone(),
            node.data.leaf,
        )
        .await?
        .upgrade()
        .await?;
        new_node.data.keys = child.data.keys.drain((self.order - 1)..).collect();

        if !child.data.leaf {
            new_node.data.children = child.data.children.drain(self.order..).collect();
        }

        let (_, node) = try_join!(new_node.sync(), node.sync_and_downgrade(),)?;

        Ok(node)
    }
}

#[async_trait]
impl Collect for Index {
    type Selector = Key;
    type Item = Key;

    async fn get(self: Arc<Self>, txn: Arc<Txn>, selector: Self::Selector) -> GetResult {
        self.validate_key(&selector)?;
        let root_id = &self.root.read(txn.id().clone()).await?.0;
        let root_node = self.get_node(txn.id().clone(), root_id).await?;
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

        let root_id = self.root.read(txn.id().clone()).await?;

        if self.collator.compare(selector, &key) == Ordering::Equal {
            let root = self.get_node(txn.id().clone(), &root_id.0).await?;

            if root.data.keys.len() == (2 * self.order) - 1 {
                let mut root_id = root_id.upgrade().await?;
                let old_root_id = root_id.0.clone();
                let old_root = root;

                root_id.0 = new_node_id(self.file.block_ids(txn.id().clone()).await?);
                let mut new_root = Node::create(
                    self.file.clone(),
                    txn.id().clone(),
                    root_id.0.clone(),
                    false,
                )
                .await?
                .upgrade()
                .await?;
                new_root.data.children.push(old_root_id.clone());
                self.split_child(txn.id().clone(), old_root.upgrade().await?, 0)
                    .await?;

                self.insert(txn.id(), new_root.sync_and_downgrade().await?, key)
                    .await
            } else {
                self.insert(txn.id(), root, key).await
            }
        } else {
            self.update(txn.id().clone(), &root_id.0, &selector, &key)
                .await
        }
    }
}

#[async_trait]
impl Transact for Index {
    async fn commit(&self, txn_id: &TxnId) {
        self.file.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.file.rollback(txn_id).await
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
