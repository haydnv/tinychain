use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use collate::Collate;
use destream::{de, en};
use futures::future::{self, Future, TryFutureExt};
use futures::stream::{self, FuturesOrdered, TryStreamExt};
use log::debug;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{Block, BlockData, BlockId, Dir, File};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::{Transaction, TxnId};
use tc_value::{Value, ValueCollator};
use tcgeneric::{TCTryStream, Tuple};

use super::{Key, Range, RowSchema};

type Selection = FuturesOrdered<
    Pin<Box<dyn Future<Output = TCResult<TCTryStream<'static, Key>>> + Send + Unpin>>,
>;

const DEFAULT_BLOCK_SIZE: usize = 4_000;
const BLOCK_ID_SIZE: usize = 128; // UUIDs are 128-bit

type NodeId = BlockId;

#[derive(Clone)]
struct NodeKey {
    deleted: bool,
    value: Vec<Value>,
}

impl NodeKey {
    fn new(value: Vec<Value>) -> Self {
        Self {
            deleted: false,
            value,
        }
    }
}

impl AsRef<[Value]> for NodeKey {
    fn as_ref(&self) -> &[Value] {
        &self.value
    }
}

#[async_trait]
impl de::FromStream for NodeKey {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(cxt, decoder)
            .map_ok(|(deleted, value)| Self { deleted, value })
            .await
    }
}

impl<'en> en::ToStream<'en> for NodeKey {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.deleted, &self.value), encoder)
    }
}

#[cfg(debug_assertions)]
impl fmt::Display for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BTree node key: {}{}",
            Value::from_iter(self.value.to_vec()),
            if self.deleted { " (DELETED)" } else { "" }
        )
    }
}

#[derive(Clone)]
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

impl BlockData for Node {
    fn ext() -> &'static str {
        "node"
    }
}

#[async_trait]
impl de::FromStream for Node {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(cxt, decoder)
            .map_ok(|(leaf, keys, parent, children, rebalance)| Self {
                leaf,
                keys,
                parent,
                children,
                rebalance,
            })
            .await
    }
}

impl<'en> en::ToStream<'en> for Node {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            (
                &self.leaf,
                &self.keys,
                &self.parent,
                &self.children,
                &self.rebalance,
            ),
            encoder,
        )
    }
}

#[cfg(debug_assertions)]
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
            Tuple::<NodeKey>::from_iter(self.keys.iter().cloned())
        )?;
        write!(f, "\t {} children", self.children.len())
    }
}

struct Inner<F, D, T> {
    file: F,
    schema: RowSchema,
    order: usize,
    collator: ValueCollator,
    root: TxnLock<Mutable<NodeId>>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

#[derive(Clone)]
pub struct BTreeFile<F, D, T> {
    inner: Arc<Inner<F, D, T>>,
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> BTreeFile<F, D, T>
where
    Self: Clone + 'static,
{
    pub async fn create(txn: T, file: F, schema: RowSchema) -> TCResult<Self> {
        if !file.is_empty(txn.id()).await? {
            return Err(TCError::internal(
                "Tried to create a new BTree without a new File",
            ));
        }

        let mut key_size = 0;
        for col in &schema {
            if let Some(size) = col.dtype().size() {
                key_size += size;
                if col.max_len().is_some() {
                    return Err(TCError::bad_request(
                        "Maximum length is not applicable to",
                        col.dtype(),
                    ));
                }
            } else if let Some(size) = col.max_len() {
                key_size += size;
            } else {
                return Err(TCError::bad_request(
                    "Type requires a maximum length",
                    col.dtype(),
                ));
            }
        }
        // each individual column requires 1-2 bytes of type data
        key_size += schema.len() * 2;
        // the "leaf" and "deleted" booleans each add two bytes to a key as-stored
        key_size += 4;

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
            .create_block(*txn.id(), root.clone(), Node::new(true, None))
            .await?;

        Ok(BTreeFile {
            inner: Arc::new(Inner {
                file,
                schema,
                order,
                collator: ValueCollator::default(),
                root: TxnLock::new("BTree root", root.into()),
                dir: PhantomData,
                txn: PhantomData,
            }),
        })
    }

    pub fn collator(&'_ self) -> &'_ ValueCollator {
        &self.inner.collator
    }

    fn slice<B: Deref<Target = Node> + 'static>(
        self,
        txn_id: TxnId,
        node: B,
        range: Range,
    ) -> TCResult<TCTryStream<'static, Key>> {
        let (l, r) = self.inner.collator.bisect(&node.keys[..], &range);

        debug!("_slice {} from {} to {}", node.deref(), l, r);

        if node.leaf {
            let stream: TCTryStream<Key> = if l == r && l < node.keys.len() {
                if node.keys[l].deleted {
                    Box::pin(stream::empty())
                } else {
                    let key = TCResult::Ok(node.keys[l].value.to_vec());
                    Box::pin(stream::once(future::ready(key)))
                }
            } else {
                let keys = node.keys[l..r]
                    .iter()
                    .filter(|k| !k.deleted)
                    .map(|k| k.value.to_vec())
                    .map(TCResult::Ok)
                    .collect::<Vec<TCResult<Key>>>();

                Box::pin(stream::iter(keys))
            };

            Ok(stream)
        } else {
            let mut selected: Selection = FuturesOrdered::new();
            for i in l..r {
                let child_id = node.children[i].clone();
                let range_clone = range.clone();

                let this = self.clone();
                let selection = Box::pin(async move {
                    let node = this.inner.file.read_block(&txn_id, &child_id).await?;

                    this.slice(txn_id, node, range_clone)
                });
                selected.push(Box::pin(selection));

                if !node.keys[i].deleted {
                    let key_at_i = TCResult::Ok(node.keys[i].value.to_vec());
                    let key_at_i: TCTryStream<Key> =
                        Box::pin(stream::once(future::ready(key_at_i)));

                    selected.push(Box::pin(future::ready(Ok(key_at_i))));
                }
            }

            let last_child_id = node.children[r].clone();

            let selection = Box::pin(async move {
                let node = self.inner.file.read_block(&txn_id, &last_child_id).await?;

                self.slice(txn_id, node, range)
            });
            selected.push(Box::pin(selection));

            Ok(Box::pin(selected.try_flatten()))
        }
    }

    fn slice_reverse<B: Deref<Target = Node> + 'static>(
        self,
        _txn_id: TxnId,
        _node: B,
        _range: Range,
    ) -> TCResult<TCTryStream<'static, Key>> {
        unimplemented!()
    }

    async fn split_child<'a>(
        &'a self,
        txn_id: TxnId,
        node_id: NodeId,
        block: &'a F::Block,
        i: usize,
    ) -> TCResult<()> {
        let file = &self.inner.file;
        let order = self.inner.order;

        let mut node = block.write().await;
        let child_id = node.children[i].clone();
        let mut child = file.write_block(txn_id, child_id).await?;

        debug!(
            "child to split has {} keys and {} children",
            child.keys.len(),
            child.children.len()
        );

        let new_node_id = file.unique_id(&txn_id).await?;

        node.children.insert(i + 1, new_node_id.clone());
        node.keys.insert(i, child.keys.remove(order - 1));

        let mut new_node = Node::new(child.leaf, Some(node_id));
        new_node.keys = child.keys.drain((order - 1)..).collect();

        if child.leaf {
            debug!("child is a leaf node");
        } else {
            new_node.children = child.children.drain(order..).collect();
        }

        file.create_block(txn_id, new_node_id, new_node).await?;

        Ok(())
    }
}
