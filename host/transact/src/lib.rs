//! Provides traits and data structures to define a distributed transaction context.
//!
//! This library is part of TinyChain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use async_trait::async_trait;
use destream::en;
use sha2::digest::Output;
use sha2::Sha256;

use tc_error::*;
use tcgeneric::Id;

mod id;

pub mod fs;
pub mod lock;

pub use id::{TxnId, MIN_ID};

/// Defines a method to compute the hash of this state as of a given [`TxnId`]
#[async_trait]
pub trait AsyncHash<D: fs::Dir> {
    /// The type of [`Transaction`] which this state supports
    type Txn: Transaction<D>;

    /// Compute the hash of this state as of a given [`TxnId`]
    async fn hash(self, txn: &Self::Txn) -> TCResult<Output<Sha256>>;
}

/// Access a view which can be encoded with [`en::IntoStream`].
#[async_trait]
pub trait IntoView<'en, D: fs::Dir> {
    /// The type of [`Transaction`] which this state supports
    type Txn: Transaction<D>;

    /// The type of encodable view returned by `into_view`
    type View: en::IntoStream<'en> + Sized + 'en;

    /// Return a `View` which can be encoded with [`en::IntoStream`].
    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View>;
}

/// Transaction lifecycle callbacks
// TODO: add a `rollback` method separate from `finalize`
#[async_trait]
pub trait Transact {
    /// A guard which blocks concurrent commits
    type Commit: Send + Sync;

    /// Commit this transaction.
    async fn commit(&self, txn_id: &TxnId) -> Self::Commit;

    /// Delete any version data specific to this transaction.
    async fn finalize(&self, txn_id: &TxnId);
}

/// Common transaction context properties.
#[async_trait]
pub trait Transaction<D: fs::Dir>: Clone + Sized + Send + Sync + 'static {
    /// The [`TxnId`] of this transaction context.
    fn id(&'_ self) -> &'_ TxnId;

    /// Borrow the [`fs::Dir`] of this transaction context.
    fn context(&'_ self) -> &'_ D;

    /// Return a transaction subcontext with its own [`fs::Dir`].
    async fn subcontext(&self, id: Id) -> TCResult<Self>;

    /// Return a transaction subcontext with its own unique [`fs::Dir`].
    async fn subcontext_unique(&self) -> TCResult<Self>;
}
