use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Deref;

use async_trait::async_trait;
use collate::Collator;
use destream::{de, en};
use futures::TryFutureExt;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, Dir, File};
use tc_transact::lock::{Mutable, TxnLock};
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::Tuple;

use super::RowSchema;

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

impl Deref for NodeKey {
    type Target = [Value];

    fn deref(&self) -> &[Value] {
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

#[derive(Clone)]
pub struct BTreeFile<F, D, T> {
    file: F,
    schema: RowSchema,
    order: usize,
    collator: Collator<Value>,
    root: TxnLock<Mutable<NodeId>>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> BTreeFile<F, D, T> {
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
            file,
            schema,
            order,
            collator: Collator::default(),
            root: TxnLock::new("BTree root", root.into()),
            dir: PhantomData,
            txn: PhantomData,
        })
    }

    pub fn collator(&'_ self) -> &'_ Collator<Value> {
        &self.collator
    }
}
