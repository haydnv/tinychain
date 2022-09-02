use std::cmp::Ordering;
use std::fmt;
#[cfg(debug_assertions)]
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use collate::Collate;
use destream::{de, en};
use futures::future::{self, Future, TryFutureExt};
use futures::stream::{self, FuturesOrdered, FuturesUnordered, TryStreamExt};
use futures::{join, try_join};
use log::debug;
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::*;
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Value, ValueCollator};
use tcgeneric::{Instance, TCBoxTryFuture, TCBoxTryStream, Tuple};

use super::{BTree, BTreeInstance, BTreeSlice, BTreeType, BTreeWrite, Key, Range, RowSchema};

type Selection<'a> = FuturesOrdered<
    Pin<Box<dyn Future<Output = TCResult<TCBoxTryStream<'a, Key>>> + Send + Unpin + 'a>>,
>;

const DEFAULT_BLOCK_SIZE: usize = 4_000;
const BLOCK_ID_SIZE: usize = 128; // UUIDs are 128-bit

type NodeId = BlockId;

#[derive(Clone, Eq, PartialEq)]
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
            .map_err(|e| de::Error::custom(format!("error decoding BTree node key: {}", e)))
            .await
    }
}

impl<'en> en::ToStream<'en> for NodeKey {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.deleted, &self.value), encoder)
    }
}

impl<'en> en::IntoStream<'en> for NodeKey {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((self.deleted, self.value), encoder)
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

/// A [`BTree`] node
#[derive(Clone, Eq, PartialEq)]
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

    #[cfg(debug_assertions)]
    fn validate(&self, schema: &[super::Column], range: &Range) {
        debug!("validate {}", self);

        debug!("range: {:?}", range);
        for key in &self.keys {
            debug!("key: {}", key);
            assert_eq!(key.value.len(), schema.len());
            assert!(range.prefix().len() <= key.value.len());
            assert!(range.len() <= key.value.len());
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
            .map_err(|e| de::Error::custom(format!("error decoding BTree node: {}", e)))
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

impl<'en> en::IntoStream<'en> for Node {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            (
                self.leaf,
                self.keys,
                self.parent,
                self.children,
                self.rebalance,
            ),
            encoder,
        )
    }
}

#[cfg(debug_assertions)]
impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
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
    root: TxnLock<NodeId>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

/// The base type of a [`BTree`]
#[derive(Clone)]
pub struct BTreeFile<F, D, T> {
    inner: Arc<Inner<F, D, T>>,
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> BTreeFile<F, D, T>
where
    Self: Clone,
{
    fn new(file: F, schema: RowSchema, order: usize, root: NodeId) -> Self {
        BTreeFile {
            inner: Arc::new(Inner {
                file,
                schema,
                order,
                collator: ValueCollator::default(),
                root: TxnLock::new("BTree root", root.into()),
                dir: PhantomData,
                txn: PhantomData,
            }),
        }
    }

    /// Create a new `BTreeFile`.
    pub async fn create(file: F, schema: RowSchema, txn_id: TxnId) -> TCResult<Self> {
        let mut file_contents = file.write(txn_id).await?;
        if !file_contents.is_empty() {
            return Err(TCError::internal(
                "Tried to create a new BTree without a new File",
            ));
        }

        let order = validate_schema(&schema)?;

        let root: BlockId = Uuid::new_v4().into();
        let node = Node::new(true, None);
        file_contents
            .create_block(root.clone(), node, DEFAULT_BLOCK_SIZE)
            .await?;

        Ok(BTreeFile::new(file, schema, order, root))
    }

    // TODO: this should hold a read lock on self.inner.file
    fn _delete_range<'a>(
        &'a self,
        txn_id: TxnId,
        node_id: NodeId,
        range: &'a Range,
    ) -> TCBoxTryFuture<'a, ()> {
        debug_assert!(range.len() <= self.inner.schema.len());

        Box::pin(async move {
            let collator = &self.inner.collator;
            let file = &self.inner.file;

            let mut node = file.write_block(txn_id, &node_id).await?;

            #[cfg(debug_assertions)]
            node.validate(&self.inner.schema, range);

            let (l, r) = collator.bisect(&node.keys, range);

            #[cfg(debug_assertions)]
            debug!("delete from {} [{}..{}] ({:?})", *node, l, r, range);

            if node.leaf {
                for i in l..r {
                    node.keys[i].deleted = true;
                }

                node.rebalance = true;

                Ok(())
            } else if r > l {
                node.rebalance = true;

                for i in l..r {
                    node.keys[i].deleted = true;
                }

                let deletes: FuturesUnordered<_> = node.children[l..(r + 1)]
                    .iter()
                    .cloned()
                    .map(|child_id| self._delete_range(txn_id, child_id, range))
                    .collect();

                deletes.try_fold((), |(), ()| future::ready(Ok(()))).await
            } else {
                let child_id = node.children[r].clone();
                self._delete_range(txn_id, child_id, range).await
            }
        })
    }

    fn _insert(
        &self,
        txn_id: TxnId,
        mut node: <F::Read as FileRead<Node>>::Write,
        key: Key,
    ) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            let collator = &self.inner.collator;
            // TODO: this should be a read lock with an upgrade method
            let file = self.inner.file.write(txn_id).await?;
            let order = self.inner.order;

            let i = collator.bisect_left(&node.keys, &key);

            #[cfg(debug_assertions)]
            debug!("insert at index {} into {}", i, *node);

            if node.leaf {
                let key = NodeKey::new(key);

                if i == node.keys.len() {
                    node.keys.insert(i, key);
                    return Ok(());
                }

                #[cfg(debug_assertions)]
                debug!("insert key {} into {} at {}", key, *node, i);

                match collator.compare_slice(&key, &node.keys[i]) {
                    Ordering::Less => node.keys.insert(i, key),
                    Ordering::Equal => {
                        #[cfg(debug_assertions)]
                        debug!("un-delete key at {}: {}", i, key);
                        node.keys[i].deleted = false
                    }
                    Ordering::Greater => panic!("error in Collate::bisect_left"),
                }

                Ok(())
            } else {
                let child_id = node.children[i].clone();
                let child = file.write_block(&child_id).await?;

                if child.keys.len() == (2 * order) - 1 {
                    let child_id = node.children[i].clone();
                    let (file, mut node) =
                        Self::split_child(self.inner.order, file, node, child_id, child, i).await?;

                    match collator.compare_slice(&key, &node.keys[i]) {
                        Ordering::Less => self._insert(txn_id, node, key).await,
                        Ordering::Equal => {
                            if node.keys[i].deleted {
                                node.keys[i].deleted = false;
                            }

                            return Ok(());
                        }
                        Ordering::Greater => {
                            let child_id = node.children[i + 1].clone();
                            let child = file.write_block(&child_id).await?;
                            self._insert(txn_id, child, key).await
                        }
                    }
                } else {
                    self._insert(txn_id, child, key).await
                }
            }
        })
    }

    fn _slice<'a, B: Deref<Target = Node>>(
        self,
        txn_id: TxnId,
        node: B,
        range: Range,
    ) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        let (l, r) = self.inner.collator.bisect(&node.keys[..], &range);

        #[cfg(debug_assertions)]
        debug!("_slice {} from {} to {} ({:?})", *node, l, r, range);

        if node.leaf {
            let keys = node.keys[l..r]
                .iter()
                .filter(|k| !k.deleted)
                .map(|k| k.value.to_vec())
                .map(TCResult::Ok)
                .collect::<Vec<TCResult<Key>>>();

            Ok(Box::pin(stream::iter(keys)))
        } else {
            let mut selected: Selection<'a> = FuturesOrdered::new();
            for i in l..r {
                let child_id = node.children[i].clone();
                let range_clone = range.clone();

                let this = self.clone();
                let selection = Box::pin(async move {
                    let node = this.inner.file.read_block(txn_id, &child_id).await?;
                    this._slice(txn_id, node, range_clone)
                });

                selected.push(Box::pin(selection));

                if !node.keys[i].deleted {
                    let key_at_i = TCResult::Ok(node.keys[i].value.to_vec());
                    let key_at_i: TCBoxTryStream<Key> =
                        Box::pin(stream::once(future::ready(key_at_i)));

                    selected.push(Box::pin(future::ready(Ok(key_at_i))));
                }
            }

            let last_child_id = node.children[r].clone();

            let selection = Box::pin(async move {
                let node = self.inner.file.read_block(txn_id, &last_child_id).await?;
                self._slice(txn_id, node, range)
            });
            selected.push(Box::pin(selection));

            Ok(Box::pin(selected.try_flatten()))
        }
    }

    fn _slice_reverse<'a, B: Deref<Target = Node>>(
        self,
        txn_id: TxnId,
        node: B,
        range: Range,
    ) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        let (l, r) = self.inner.collator.bisect(&node.keys, &range);

        #[cfg(debug_assertions)]
        debug!("_slice_reverse {} from {} to {} ({:?})", *node, r, l, range);

        if node.leaf {
            let keys = node.keys[l..r]
                .iter()
                .filter(|k| !k.deleted)
                .rev()
                .map(|k| k.value.to_vec())
                .map(TCResult::Ok)
                .collect::<Vec<TCResult<Key>>>();

            Ok(Box::pin(stream::iter(keys)))
        } else {
            let mut selected: Selection<'a> = FuturesOrdered::new();

            let last_child = node.children[r].clone();
            let range_clone = range.clone();
            let this = self.clone();
            let selection = Box::pin(async move {
                let node = this.inner.file.read_block(txn_id, &last_child).await?;
                this._slice_reverse(txn_id, node, range_clone)
            });
            selected.push(Box::pin(selection));

            for i in (l..r).rev() {
                let child_id = node.children[i].clone();
                let range_clone = range.clone();

                let this = self.clone();
                let selection = Box::pin(async move {
                    let node = this.inner.file.read_block(txn_id, &child_id).await?;
                    this._slice_reverse(txn_id, node, range_clone)
                });

                if !node.keys[i].deleted {
                    let key_at_i = TCResult::Ok(node.keys[i].value.to_vec());
                    let key_at_i: TCBoxTryStream<Key> =
                        Box::pin(stream::once(future::ready(key_at_i)));

                    selected.push(Box::pin(future::ready(Ok(key_at_i))));
                }

                selected.push(Box::pin(selection));
            }

            Ok(Box::pin(selected.try_flatten()))
        }
    }

    pub(super) async fn rows_in_range<'a>(
        self,
        txn_id: TxnId,
        range: Range,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        let root_id = self.inner.root.read(txn_id).await?;

        let root = self.inner.file.read_block(txn_id, &*root_id).await?;

        if reverse {
            self._slice_reverse(txn_id, root, range)
        } else {
            self._slice(txn_id, root, range)
        }
    }

    async fn split_child(
        order: usize,
        mut file: F::Write,
        mut node: <F::Read as FileRead<Node>>::Write,
        node_id: NodeId,
        mut child: <F::Read as FileRead<Node>>::Write,
        i: usize,
    ) -> TCResult<(F::Write, <F::Read as FileRead<Node>>::Write)> {
        debug!("BTree::split_child");

        assert_eq!(node_id, node.children[i].clone());

        debug!(
            "child to split has {} keys and {} children",
            child.keys.len(),
            child.children.len()
        );

        let new_node = Node::new(child.leaf, Some(node_id));
        let (new_node_id, mut new_node) = file
            .create_block_unique(new_node, DEFAULT_BLOCK_SIZE)
            .await?;

        debug!("BTree::split_child created new node {}", new_node_id);

        node.children.insert(i + 1, new_node_id.clone());
        node.keys.insert(i, child.keys.remove(order - 1));

        new_node.keys = child.keys.drain((order - 1)..).collect();

        if child.leaf {
            debug!("child is a leaf node");
        } else {
            new_node.children = child.children.drain(order..).collect();
        }

        Ok((file, node))
    }
}

impl<F, D, T> Instance for BTreeFile<F, D, T>
where
    Self: Send + Sync,
{
    type Class = BTreeType;

    fn class(&self) -> Self::Class {
        BTreeType::File
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> BTreeInstance for BTreeFile<F, D, T>
where
    Self: Clone,
    BTreeSlice<F, D, T>: 'static,
{
    type Slice = BTreeSlice<F, D, T>;

    fn collator(&'_ self) -> &'_ ValueCollator {
        &self.inner.collator
    }

    fn schema(&'_ self) -> &'_ RowSchema {
        &self.inner.schema
    }

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice> {
        BTreeSlice::new(BTree::File(self), range, reverse)
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        let root_id = self.inner.root.read(txn_id).await?;
        let root = self.inner.file.read_block(txn_id, &*root_id).await?;
        Ok(root.keys.is_empty())
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Key>> {
        self.rows_in_range(txn_id, Range::default(), false).await
    }

    fn validate_key(&self, key: Key) -> TCResult<Key> {
        if key.len() != self.inner.schema.len() {
            return Err(TCError::bad_request("invalid key length", Tuple::from(key)));
        }

        key.into_iter()
            .zip(&self.inner.schema)
            .map(|(val, col)| {
                val.into_type(col.dtype)
                    .ok_or_else(|| TCError::bad_request("invalid value for column", &col.name))
            })
            .collect()
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> BTreeWrite for BTreeFile<F, D, T>
where
    Self: Clone,
    BTreeSlice<F, D, T>: 'static,
{
    async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()> {
        if range == Range::default() {
            let (mut root, mut file) =
                try_join!(self.inner.root.write(txn_id), self.inner.file.write(txn_id))?;

            file.truncate().await?;

            *root = Uuid::new_v4().into();
            let node = Node::new(true, None);
            file.create_block((*root).clone(), node, DEFAULT_BLOCK_SIZE)
                .await?;

            return Ok(());
        }

        let root_id = self.inner.root.read(txn_id).await?;
        self._delete_range(txn_id, (*root_id).clone(), &range).await
    }

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        let key = self.validate_key(key)?;

        let mut file = self.inner.file.write(txn_id).await?;
        let order = self.inner.order;

        // get a write lock on the root_id while we check if a split_child is needed,
        // to avoid getting out of sync in the case of a concurrent insert in the same txn
        let mut root_id = self.inner.root.write(txn_id).await?;
        debug!("insert into BTree with root node ID {}", *root_id);

        let root = file.write_block(&*root_id).await?;

        #[cfg(debug_assertions)]
        debug!(
            "insert {} into BTree, root node {} has {} keys and {} children (order is {})",
            <Tuple<Value> as std::iter::FromIterator<Value>>::from_iter(key.to_vec()),
            *root_id,
            root.keys.len(),
            root.children.len(),
            order
        );

        #[cfg(debug_assertions)]
        debug!("root node {} is {}", *root_id, *root);

        assert_eq!(root.children.is_empty(), root.leaf);

        if root.keys.len() == (2 * order) - 1 {
            debug!("split root node");

            let old_root_id = (*root_id).clone();

            let mut new_root = Node::new(false, None);
            new_root.children.push(old_root_id.clone());

            let (new_root_id, new_root) = file
                .create_block_unique(new_root, DEFAULT_BLOCK_SIZE)
                .await?;

            (*root_id) = new_root_id;

            let (_file, new_root) =
                Self::split_child(self.inner.order, file, new_root, old_root_id, root, 0).await?;

            self._insert(txn_id, new_root, key).await
        } else {
            self._insert(txn_id, root, key).await
        }
    }
}

#[async_trait]
impl<F: File<Node> + Transact, D: Dir, T: Transaction<D>> Transact for BTreeFile<F, D, T> {
    async fn commit(&self, txn_id: &TxnId) {
        join!(
            self.inner.file.commit(txn_id),
            self.inner.root.commit(txn_id)
        );
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(
            self.inner.file.finalize(txn_id),
            self.inner.root.finalize(txn_id)
        );
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> Persist<D> for BTreeFile<F, D, T> {
    type Schema = RowSchema;
    type Store = F;
    type Txn = T;

    fn schema(&self) -> &Self::Schema {
        &self.inner.schema
    }

    async fn load(txn: &T, schema: RowSchema, file: F) -> TCResult<Self> {
        debug!("BTreeFile::load {:?}", schema);

        let file_contents = file.write(*txn.id()).await?;

        let order = validate_schema(&schema)?;

        let txn_id = *txn.id();
        let mut root = None;
        for block_id in file_contents.block_ids() {
            debug!("BTreeFile::load block {}", block_id);

            let block = file.read_block(txn_id, block_id).await?;

            debug!("BTreeFile::loaded block {}", block_id);

            if block.parent.is_none() {
                root = Some(block_id.clone());
                break;
            }
        }

        let root = root.ok_or_else(|| TCError::internal("BTree corrupted (missing root block)"))?;

        Ok(BTreeFile::new(file, schema, order, root))
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> Restore<D> for BTreeFile<F, D, T> {
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        if self.inner.schema != backup.inner.schema {
            return Err(TCError::unsupported(
                "cannot restore a BTree from a backup with a different schema",
            ));
        }

        self.inner
            .file
            .copy_from(txn_id, &backup.inner.file, true)
            .await
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>, I: BTreeInstance + 'static> CopyFrom<D, I>
    for BTreeFile<F, D, T>
{
    async fn copy_from(source: I, file: F, txn: &T) -> TCResult<Self> {
        let txn_id = *txn.id();
        let schema = source.schema().clone();
        let dest = Self::create(file, schema, txn_id).await?;
        let keys = source.keys(txn_id).await?;
        dest.try_insert_from(txn_id, keys).await?;
        Ok(dest)
    }
}

impl<F, D, T> fmt::Display for BTreeFile<F, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a BTree")
    }
}

fn validate_schema(schema: &RowSchema) -> TCResult<usize> {
    let mut key_size = 0;
    for col in schema {
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

    Ok(order)
}
