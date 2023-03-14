//! A [`BTree`], an ordered transaction-aware collection of [`Key`]s

use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use b_table::b_tree;
use futures::Stream;
use safecast::{as_type, AsType};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{
    path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf, ThreadSafe,
};

pub use file::BTreeFile;
pub use schema::{Column, Schema};
pub use slice::BTreeSlice;
pub(crate) use stream::BTreeView;
pub use stream::Keys;

mod file;
mod schema;
mod slice;
mod stream;

const PREFIX: PathLabel = path_label(&["state", "collection", "btree"]);

/// A key in a [`BTree`]
pub type Key = b_tree::Key<Value>;

/// A node in a [`BTree`]
pub type Node = b_tree::Node<Vec<Key>>;

/// A range used to slice a [`BTree`]
pub type Range = b_tree::Range<Value>;

/// The [`Class`] of a [`BTree`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum BTreeType {
    File,
    Slice,
}

impl Class for BTreeType {}

impl NativeClass for BTreeType {
    // These functions are only used for serialization,
    // and there's no way to transmit a BTreeSlice.

    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if &path[..] == &PREFIX[..] {
            Some(Self::File)
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        PREFIX.into()
    }
}

impl Default for BTreeType {
    fn default() -> Self {
        Self::File
    }
}

impl<D: Digest> Hash<D> for BTreeType {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self.path())
    }
}

impl fmt::Debug for BTreeType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::File => f.write_str("type BTree"),
            Self::Slice => f.write_str("type BTreeSlice"),
        }
    }
}

/// A slice of a B+Tree
#[async_trait]
pub trait BTreeInstance: Clone + Instance {
    type Slice: BTreeInstance;

    /// Borrow to this `BTree`'s schema.
    fn schema(&self) -> &Schema;

    /// Return the number of [`Key`]s in this `BTree`.
    async fn count(&self, txn_id: TxnId) -> TCResult<u64>;

    /// Return `true` if this `BTree` has no [`Key`]s.
    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool>;

    /// Return a `Stream` of this `BTree`'s [`Key`]s.
    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<Keys<'a>>
    where
        Self: 'a;

    /// Return a slice of this [`BTreeInstance`] with the given range.
    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice>;
}

/// B+Tree write methods.
#[async_trait]
pub trait BTreeWrite: BTreeInstance {
    /// Delete all the [`Key`]s in this `BTree`.
    async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()>;

    /// Insert the given [`Key`] into this `BTree`.
    ///
    /// If the [`Key`] is already present, this is a no-op.
    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()>;

    /// Insert all the keys from the given [`Stream`] into this `BTree`.
    /// The stream of `keys` does not need to be collated.
    /// This will stop and return an error if it encounters an invalid [`Key`].
    async fn try_insert_from<S>(&self, txn_id: TxnId, keys: S) -> TCResult<()>
    where
        S: Stream<Item = TCResult<Key>> + Send + Unpin;
}

pub enum BTree<Txn, FE> {
    File(BTreeFile<Txn, FE>),
    Slice(BTreeSlice<Txn, FE>),
}

as_type!(BTree<Txn, FE>, File, BTreeFile<Txn, FE>);
as_type!(BTree<Txn, FE>, Slice, BTreeSlice<Txn, FE>);

impl<Txn, FE> Clone for BTree<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::File(file) => Self::File(file.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
        }
    }
}

impl<Txn, FE> Instance for BTree<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: Send + Sync,
{
    type Class = BTreeType;

    fn class(&self) -> Self::Class {
        match self {
            Self::File(file) => file.class(),
            Self::Slice(slice) => slice.class(),
        }
    }
}

#[async_trait]
impl<Txn, FE> BTreeInstance for BTree<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Slice = Self;

    fn schema(&self) -> &Schema {
        match self {
            Self::File(file) => BTreeInstance::schema(file),
            Self::Slice(slice) => BTreeInstance::schema(slice),
        }
    }

    fn slice(self, range: Range, reverse: bool) -> TCResult<Self> {
        if range == Range::default() && !reverse {
            return Ok(self);
        }

        match self {
            Self::File(file) => file.slice(range, reverse).map(BTree::Slice),
            Self::Slice(slice) => slice.slice(range, reverse).map(BTree::Slice),
        }
    }

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::File(file) => file.count(txn_id).await,
            Self::Slice(slice) => slice.count(txn_id).await,
        }
    }

    async fn is_empty(&self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::File(file) => file.is_empty(txn_id).await,
            Self::Slice(slice) => slice.is_empty(txn_id).await,
        }
    }

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<Keys<'a>>
    where
        Self: 'a,
    {
        match self {
            Self::File(file) => file.keys(txn_id).await,
            Self::Slice(slice) => slice.keys(txn_id).await,
        }
    }
}

#[async_trait]
impl<Txn, FE> BTreeWrite for BTree<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn delete(&self, txn_id: TxnId, range: Range) -> TCResult<()> {
        match self {
            Self::File(file) => file.delete(txn_id, range).await,
            slice => Err(bad_request!(
                "{:?} does not support write operations",
                slice
            )),
        }
    }

    async fn insert(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        match self {
            Self::File(file) => file.insert(txn_id, key).await,
            slice => Err(bad_request!(
                "{:?} does not support write operations",
                slice
            )),
        }
    }

    async fn try_insert_from<S>(&self, txn_id: TxnId, keys: S) -> TCResult<()>
    where
        S: Stream<Item = TCResult<Key>> + Send + Unpin,
    {
        match self {
            Self::File(file) => file.try_insert_from(txn_id, keys).await,
            slice => Err(bad_request!(
                "{:?} does not support write operations",
                slice
            )),
        }
    }
}

impl<Txn, FE> fmt::Debug for BTree<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a BTree")
    }
}
