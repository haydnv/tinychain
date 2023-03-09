use std::fmt;

use async_trait::async_trait;

use tc_error::*;
use tc_transact::TxnId;
use tc_value::Value;
use tcgeneric::{
    path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCBoxTryStream, TCPathBuf,
};

pub use schema::Schema;

mod schema;

const PREFIX: PathLabel = path_label(&["state", "collection", "btree"]);

/// A key in a B+Tree
pub type Key = b_tree::Key<Value>;

/// A range in a B+Tree
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

impl fmt::Display for BTreeType {
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
