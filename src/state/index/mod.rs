use std::cmp::Ordering;
use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{self, BoxFuture, Future};
use futures::stream::{self, FuturesOrdered, StreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error;
use crate::internal::{BlockId, File};
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{TCResult, TCStream, TCType, Value, ValueId};

use super::{Collect, GetResult, State};

mod collator;

const DEFAULT_BLOCK_SIZE: usize = 4_000;
const BLOCK_ID_SIZE: usize = 128; // UUIDs are 128-bit
const ERR_CORRUPT: &str = "Index corrupted! Please restart Tinychain and file a bug report";

type NodeId = BlockId;

#[derive(Deserialize, Serialize)]
struct Node {
    leaf: bool,
    keys: Vec<Vec<Value>>,
    children: Vec<NodeId>,
}

impl Node {
    fn new(leaf: bool) -> Node {
        Node {
            leaf,
            keys: vec![],
            children: vec![],
        }
    }
}

type Key = Vec<Value>;
type Selection = FuturesOrdered<Pin<Box<dyn Future<Output = TCStream<Key>> + Send + Sync + Unpin>>>;

#[derive(Clone)]
struct IndexRoot(NodeId);

#[async_trait]
impl Mutate for IndexRoot {
    async fn commit(&mut self, new_value: IndexRoot) {
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

        if file.is_empty(&txn_id).await {
            let root: BlockId = Uuid::new_v4().into();
            file.new_block(
                txn_id.clone(),
                root.clone(),
                Bytes::from(bincode::serialize(&Node::new(true))?),
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

    async fn get_node(&self, txn_id: &TxnId, node_id: &NodeId) -> TCResult<Node> {
        if let Some(node) = self.file.get_block(txn_id, node_id).await {
            bincode::deserialize(&node).map_err(|_| error::internal(ERR_CORRUPT))
        } else {
            Err(error::internal(ERR_CORRUPT))
        }
    }

    async fn put_node(&self, txn_id: &TxnId, node_id: NodeId, node: &Node) -> TCResult<()> {
        self.file
            .put_block(
                txn_id.clone(),
                node_id,
                (&bincode::serialize(node)?[..]).into(),
            )
            .await
    }

    async fn put_node_if_updated(
        &self,
        txn_id: &TxnId,
        node_id: NodeId,
        node: Node,
        updated: bool,
    ) -> TCResult<()> {
        if updated {
            self.put_node(txn_id, node_id, &node).await
        } else {
            Ok(())
        }
    }

    fn select(self: Arc<Self>, txn_id: TxnId, mut node: Node, key: Key) -> TCStream<Key> {
        let l = self.collator.bisect_left(&node.keys, &key);
        let r = self.collator.bisect(&node.keys, &key);

        if node.leaf {
            let keys: Vec<Key> = node.keys.drain(l..r).collect();
            Box::pin(stream::iter(keys))
        } else {
            let mut selected: Selection = FuturesOrdered::new();
            for i in l..r {
                let index = self.clone();
                let child = node.children[i].clone();
                let txn_id = txn_id.clone();
                let key = key.clone();

                let selection = Box::pin(async move {
                    let node = index.get_node(&txn_id, &child).await.unwrap();
                    index.select(txn_id, node, key)
                });
                selected.push(Box::pin(selection));

                let key_at_i = node.keys[i].clone(); // TODO: drain the node instead of cloning its keys
                let key_at_i: TCStream<Key> = Box::pin(stream::once(future::ready(key_at_i)));
                selected.push(Box::pin(future::ready(key_at_i)));
            }

            let last_child = node.children[r].clone();
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
            let mut node = self.get_node(txn_id, node_id).await?;
            let l = self.collator.bisect_left(&node.keys, &selector);
            let r = self.collator.bisect(&node.keys, &selector);
            let mut updated = false;

            if node.leaf {
                for i in l..r {
                    node.keys[i] = value.to_vec();
                    updated = true;
                }

                self.put_node_if_updated(txn_id, node_id.clone(), node, updated)
                    .await?;
                Ok(())
            } else {
                for i in l..r {
                    self.update(txn_id, &node.children[i], selector, value)
                        .await?;
                    node.keys[i] = value.to_vec();
                    updated = true;
                }

                let last_child = node.children.last().unwrap().clone();
                self.put_node_if_updated(txn_id, node_id.clone(), node, updated)
                    .await?;
                self.update(txn_id, &last_child, selector, value).await
            }
        })
    }

    fn insert<'a>(
        &'a self,
        txn_id: &'a TxnId,
        node_id: &'a NodeId,
        node: &'a mut Node,
        key: Key,
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let mut i = self.collator.bisect_left(&node.keys, &key);
            if node.leaf {
                if i == node.keys.len()
                    || self.collator.compare(&node.keys[i], &key) != Ordering::Equal
                {
                    node.keys.insert(i, key);
                    self.put_node(txn_id, node_id.clone(), &node).await?;
                }

                Ok(())
            } else {
                let mut child = self.get_node(txn_id, &node.children[i]).await?;
                if child.keys.len() == (2 * self.order) - 1 {
                    self.split_child(txn_id, node_id.clone(), node, i).await?;
                    if self.collator.compare(&key, &node.keys[i]) == Ordering::Greater {
                        i += 1
                    }
                }

                self.insert(txn_id, &node.children[i], &mut child, key)
                    .await
            }
        })
    }

    async fn split_child(
        &self,
        txn_id: &TxnId,
        node_id: NodeId,
        node: &mut Node,
        i: usize,
    ) -> TCResult<()> {
        let mut child = self.get_node(txn_id, &node.children[i]).await?;
        let mut new_node = Node::new(child.leaf);
        let new_node_id = new_node_id(self.file.block_ids(txn_id).await);

        node.children.insert(i + 1, new_node_id.clone());
        node.keys.insert(i, child.keys.remove(self.order - 1));

        new_node.keys = child.keys.drain((self.order - 1)..).collect();

        if !child.leaf {
            new_node.children = child.children.drain(self.order..).collect();
        }

        self.put_node(txn_id, new_node_id, &new_node).await?;
        self.put_node(txn_id, node_id, node).await?;

        Ok(())
    }
}

#[async_trait]
impl Collect for Index {
    type Selector = Key;
    type Item = Key;

    async fn get(self: Arc<Self>, txn: Arc<Txn>, selector: Self::Selector) -> GetResult {
        self.validate_key(&selector)?;
        let txn_id = txn.id().clone();
        let root_id = &self.root.read(&txn_id).await?.0;
        let root_node = self.get_node(&txn_id, root_id).await?;
        Ok(Box::pin(
            self.clone()
                .select(txn_id, root_node, selector)
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

        let mut root_id = self.root.write(txn.id()).await?;

        if self.collator.compare(selector, &key) == Ordering::Equal {
            let mut root = self.get_node(txn.id(), &root_id.0).await?;

            if root.keys.len() == (2 * self.order) - 1 {
                let old_root_id = root_id.0.clone();
                let mut old_root = root;

                root_id.0 = new_node_id(self.file.block_ids(txn.id()).await);
                let mut new_root = Node::new(false);
                new_root.children.push(old_root_id.clone());
                self.split_child(txn.id(), old_root_id, &mut old_root, 0)
                    .await?;

                self.insert(txn.id(), &root_id.0, &mut new_root, key).await
            } else {
                self.insert(txn.id(), &root_id.0, &mut root, key).await
            }
        } else {
            self.update(txn.id(), &root_id.0, &selector, &key).await
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
