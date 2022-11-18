use std::cmp::Ordering;
use std::convert::TryFrom;
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
use log::{debug, trace};
use uuid::Uuid;

use tc_error::*;
use tc_transact::fs::*;
use tc_transact::lock::{TxnLock, TxnLockCommitGuard};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Value, ValueCollator};
use tcgeneric::{Instance, TCBoxTryFuture, TCBoxTryStream, Tuple};

use super::{
    BTree, BTreeInstance, BTreeSlice, BTreeType, BTreeWrite, Key, NodeId, Range, RowSchema,
};

type Selection<'a> = FuturesOrdered<
    Pin<Box<dyn Future<Output = TCResult<TCBoxTryStream<'a, Key>>> + Send + Unpin + 'a>>,
>;

const DEFAULT_BLOCK_SIZE: usize = 4_000;
const BLOCK_ID_SIZE: usize = 128; // UUIDs are 128-bit

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

impl<F: File<Key = NodeId, Block = Node>, D: Dir, T: Transaction<D>> BTreeFile<F, D, T>
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

    fn _delete_range<'a>(
        file: &'a F::Read,
        schema: &'a RowSchema,
        collator: &'a ValueCollator,
        node_id: NodeId,
        range: &'a Range,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let mut node = file.write_block(&node_id).await?;

            #[cfg(debug_assertions)]
            node.validate(&schema, range);

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
                    .map(|child_id| Self::_delete_range(file, schema, collator, child_id, range))
                    .collect();

                deletes.try_fold((), |(), ()| future::ready(Ok(()))).await
            } else {
                let child_id = node.children[r].clone();
                Self::_delete_range(file, schema, collator, child_id, range).await
            }
        })
    }

    fn _insert(
        file: F::ReadExclusive,
        order: usize,
        collator: &ValueCollator,
        txn_id: TxnId,
        node: F::BlockReadExclusive,
        key: Key,
    ) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            let i = collator.bisect_left(&node.keys, &key);

            #[cfg(debug_assertions)]
            debug!("insert at index {} into {}", i, *node);

            if node.leaf {
                let mut node = node.upgrade();
                let key = NodeKey::new(key);
                #[cfg(debug_assertions)]
                debug!("insert key {} into {} at {}", key, *node, i);

                if i == node.keys.len() {
                    node.keys.insert(i, key);
                    return Ok(());
                }

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
                debug!("locking child node {} for writing...", child_id);
                let child = file.read_block_exclusive(&child_id).await?;

                if child.keys.len() == (2 * order) - 1 {
                    let file = file.upgrade();
                    let node = node.upgrade();
                    let child = child.upgrade();
                    let child_id = node.children[i].clone();
                    let (file, mut node) =
                        Self::split_child(order, file, node, child_id, child, i).await?;

                    let file = file.downgrade();
                    match collator.compare_slice(&key, &node.keys[i]) {
                        Ordering::Less => {
                            let node = node.downgrade();
                            Self::_insert(file, order, collator, txn_id, node, key).await
                        }
                        Ordering::Equal => {
                            if node.keys[i].deleted {
                                node.keys[i].deleted = false;
                            }

                            return Ok(());
                        }
                        Ordering::Greater => {
                            let child_id = node.children[i + 1].clone();
                            let child = file.read_block_exclusive(&child_id).await?;
                            Self::_insert(file, order, collator, txn_id, child, key).await
                        }
                    }
                } else {
                    Self::_insert(file, order, collator, txn_id, child, key).await
                }
            }
        })
    }

    fn _slice<'a, B: Deref<Target = Node>>(
        file: F::Read,
        collator: ValueCollator,
        node: B,
        range: Range,
    ) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        let (l, r) = collator.bisect(&node.keys[..], &range);

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

                let file = file.clone();
                let collator = collator.clone();
                let selection = Box::pin(async move {
                    let node = file.read_block(&child_id).await?;
                    Self::_slice(file, collator, node, range_clone)
                });

                selected.push_back(Box::pin(selection));

                if !node.keys[i].deleted {
                    let key_at_i = TCResult::Ok(node.keys[i].value.to_vec());
                    let key_at_i: TCBoxTryStream<Key> =
                        Box::pin(stream::once(future::ready(key_at_i)));

                    selected.push_back(Box::pin(future::ready(Ok(key_at_i))));
                }
            }

            let last_child_id = node.children[r].clone();

            let selection = Box::pin(async move {
                let node = file.read_block(&last_child_id).await?;
                Self::_slice(file, collator, node, range)
            });
            selected.push_back(Box::pin(selection));

            Ok(Box::pin(selected.try_flatten()))
        }
    }

    fn _slice_reverse<'a, B: Deref<Target = Node>>(
        file: F::Read,
        collator: ValueCollator,
        node: B,
        range: Range,
    ) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        let (l, r) = collator.bisect(&node.keys, &range);

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
            let file_clone = file.clone();
            let collator_clone = collator.clone();
            let selection = Box::pin(async move {
                let node = file_clone.read_block(&last_child).await?;
                Self::_slice_reverse(file_clone, collator_clone, node, range_clone)
            });
            selected.push_back(Box::pin(selection));

            for i in (l..r).rev() {
                let child_id = node.children[i].clone();
                let range_clone = range.clone();

                let file = file.clone();
                let collator = collator.clone();
                let file = file.clone();
                let selection = Box::pin(async move {
                    let node = file.read_block(&child_id).await?;
                    Self::_slice_reverse(file, collator, node, range_clone)
                });

                if !node.keys[i].deleted {
                    let key_at_i = TCResult::Ok(node.keys[i].value.to_vec());
                    let key_at_i: TCBoxTryStream<Key> =
                        Box::pin(stream::once(future::ready(key_at_i)));

                    selected.push_back(Box::pin(future::ready(Ok(key_at_i))));
                }

                selected.push_back(Box::pin(selection));
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
        let file = self.inner.file.read(txn_id).await?;
        let root_id = self.inner.root.read(txn_id).await?;
        assert!(file.contains(&*root_id));

        let root = file.read_block(&*root_id).await?;
        let collator = self.collator().clone();

        if reverse {
            Self::_slice_reverse(file, collator, root, range)
        } else {
            Self::_slice(file, collator, root, range)
        }
    }

    async fn split_child(
        order: usize,
        mut file: F::Write,
        mut node: F::BlockWrite,
        node_id: NodeId,
        mut child: F::BlockWrite,
        i: usize,
    ) -> TCResult<(F::Write, F::BlockWrite)> {
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
impl<F, D, T> BTreeInstance for BTreeFile<F, D, T>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    BTreeSlice<F, D, T>: 'static,
    Self: Clone,
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
        let file = self.inner.file.read(txn_id).await?;
        let root_id = self.inner.root.read(txn_id).await?;
        assert!(file.contains(&*root_id));
        let root = file.read_block(&*root_id).await?;
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
impl<F, D, T> BTreeWrite for BTreeFile<F, D, T>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    BTreeSlice<F, D, T>: 'static,
    Self: Clone,
{
    async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()> {
        if range == Range::default() {
            let mut file = self.inner.file.write(txn_id).await?;
            let mut root_id = self.inner.root.write(txn_id).await?;
            assert!(file.contains(&*root_id));

            file.truncate().await?;

            let node = Node::new(true, None);
            let (new_root_id, _) = file.create_block_unique(node, DEFAULT_BLOCK_SIZE).await?;
            *root_id = new_root_id;

            return Ok(());
        }

        let file = self.inner.file.read(txn_id).await?;
        let root_id = self.inner.root.read(txn_id).await?;
        assert!(file.contains(&*root_id));

        let schema = &self.inner.schema;

        Self::_delete_range(&file, schema, self.collator(), (*root_id).clone(), &range).await
    }

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        let key = self.validate_key(key)?;

        let file = self.inner.file.read_exclusive(txn_id).await?;
        let root_id = self.inner.root.read_exclusive(txn_id).await?;
        assert!(file.contains(&*root_id));

        debug!("insert into BTree with root node ID {}", *root_id);

        let order = self.inner.order;

        let root = file.read_block_exclusive(&*root_id).await?;

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

            let mut file = file.upgrade();
            let mut root_id = root_id.upgrade();
            let root = root.upgrade();

            let old_root_id = (*root_id).clone();

            let mut new_root = Node::new(false, None);
            new_root.children.push(old_root_id.clone());

            let (new_root_id, new_root) = file
                .create_block_unique(new_root, DEFAULT_BLOCK_SIZE)
                .await?;

            (*root_id) = new_root_id;

            let (file, new_root) =
                Self::split_child(self.inner.order, file, new_root, old_root_id, root, 0).await?;

            let file = file.downgrade();
            let new_root = new_root.downgrade();
            Self::_insert(file, order, self.collator(), txn_id, new_root, key).await
        } else {
            Self::_insert(file, order, self.collator(), txn_id, root, key).await
        }
    }
}

#[async_trait]
impl<F, D, T> Transact for BTreeFile<F, D, T>
where
    F: File<Key = NodeId, Block = Node> + Transact,
    D: Dir,
    T: Transaction<D>,
{
    type Commit = TxnLockCommitGuard<NodeId>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        let guard = self.inner.root.commit(txn_id).await;
        self.inner.file.commit(txn_id).await;
        guard
    }

    async fn finalize(&self, txn_id: &TxnId) {
        join!(
            self.inner.file.finalize(txn_id),
            self.inner.root.finalize(txn_id)
        );
    }
}

#[async_trait]
impl<F, D, T> Persist<D> for BTreeFile<F, D, T>
where
    F: File<Key = NodeId, Block = Node> + TryFrom<D::Store, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::Store: From<F>,
{
    type Txn = T;
    type Schema = RowSchema;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: D::Store) -> TCResult<Self> {
        debug!("BTreeFile::create");

        let order = validate_schema(&schema)?;

        let store = F::try_from(store)?;
        let mut file = store.write(*txn.id()).await?;
        trace!("BTreeFile::create got file write lock");

        if !file.is_empty() {
            return Err(TCError::internal(
                "Tried to create a new BTree without a new File",
            ));
        }

        let root: NodeId = Uuid::new_v4();
        let node = Node::new(true, None);
        file.create_block(root.clone(), node, DEFAULT_BLOCK_SIZE)
            .await?;

        assert!(file.contains(&root));

        Ok(BTreeFile::new(store, schema, order, root))
    }

    async fn load(txn: &T, schema: RowSchema, store: D::Store) -> TCResult<Self> {
        debug!("BTreeFile::load {:?}", schema);

        let order = validate_schema(&schema)?;

        let txn_id = *txn.id();
        let file = F::try_from(store)?;
        let blocks = file.read(txn_id).await?;

        let mut root = None;
        for block_id in blocks.block_ids() {
            debug!("BTreeFile::load block {}", block_id);

            let block = blocks.read_block(block_id).await?;

            debug!("BTreeFile::loaded block {}", block_id);

            if block.parent.is_none() {
                root = Some(block_id.clone());
            }
        }

        let root =
            root.ok_or_else(|| TCError::internal("BTree corrupted (no root block configured)"))?;

        if blocks.contains(&root) {
            Ok(BTreeFile::new(file, schema, order, root))
        } else {
            Err(TCError::internal("BTree corrupted (missing root block)"))
        }
    }
}

#[async_trait]
impl<F, D, T> Restore<D> for BTreeFile<F, D, T>
where
    F: File<Key = NodeId, Block = Node> + TryFrom<D::Store, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::Store: From<F>,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        if self.inner.schema != backup.inner.schema {
            return Err(TCError::unsupported(
                "cannot restore a BTree from a backup with a different schema",
            ));
        }

        let (mut file, source) = try_join!(
            self.inner.file.write(txn_id),
            backup.inner.file.read(txn_id)
        )?;

        let (mut root_id, new_root_id) = try_join!(
            self.inner.root.write(txn_id),
            backup.inner.root.read(txn_id)
        )?;

        file.copy_from(&source, true).await?;
        *root_id = (*new_root_id).clone();

        Ok(())
    }
}

#[async_trait]
impl<F, D, T, I> CopyFrom<D, I> for BTreeFile<F, D, T>
where
    F: File<Key = NodeId, Block = Node> + TryFrom<D::Store, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    I: BTreeInstance + 'static,
    D::Store: From<F>,
{
    async fn copy_from(txn: &T, store: D::Store, source: I) -> TCResult<Self> {
        let txn_id = *txn.id();
        let schema = source.schema().clone();
        let dest = Self::create(txn, schema, store).await?;
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
