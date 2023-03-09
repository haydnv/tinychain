use async_trait::async_trait;

use tc_error::*;
use tc_transact::TxnId;
use tc_value::Value;
use tcgeneric::{Instance, TCBoxTryStream};

pub use schema::Schema;

mod schema;

/// A key in a B+Tree
pub type Key = b_tree::Key<Value>;

/// A range in a B+Tree
pub type Range = b_tree::Range<Value>;

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
