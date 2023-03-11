//! A [`BTree`], an ordered transaction-aware collection of [`Key`]s

use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use safecast::{as_type, AsType};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{
    path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCBoxTryStream, TCPathBuf,
};

pub use schema::Schema;
pub use slice::BTreeSlice;
pub(crate) use stream::BTreeView;
pub use tree::BTreeFile;

mod schema;
mod slice;
mod stream;
mod tree;

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
    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a;

    /// Return a slice of this [`BTreeInstance`] with the given range.
    fn slice(self, range: Range, reverse: bool) -> TCResult<Self::Slice>;
}

pub enum BTree<Txn, FE> {
    File(BTreeFile<Txn, FE>),
    Slice(BTreeSlice<Txn, FE>),
}

as_type!(BTree<Txn, FE>, File, BTreeFile<Txn, FE>);

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
    FE: AsType<Node> + Send + Sync,
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

    async fn keys<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Key>>
    where
        Self: 'a,
    {
        match self {
            Self::File(file) => file.keys(txn_id).await,
            Self::Slice(slice) => slice.keys(txn_id).await,
        }
    }
}
