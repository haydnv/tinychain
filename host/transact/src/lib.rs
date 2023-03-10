//! Provides traits and data structures to define a distributed transaction context.
//!
//! This library is part of TinyChain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use async_hash::Output;
use async_trait::async_trait;
use destream::en;
use freqfs::DirLock;

pub use sha2::Sha256; // TODO: should this be exported by the async_hash crate?

use tc_error::*;
use tcgeneric::Id;

pub mod fs;
mod id;

pub mod lock {
    use super::TxnId;

    /// A transactional read-write lock on a scalar value
    pub type TxnLock<T> = txn_lock::scalar::TxnLock<TxnId, T>;

    /// A transactional read-write lock on a key-value map
    pub type TxnMapLock<K, V> = txn_lock::map::TxnMapLock<TxnId, K, V>;

    /// A transactional read-write lock on a set of values
    pub type TxnSetLock<T> = txn_lock::set::TxnSetLock<TxnId, T>;
}

pub use id::{TxnId, MIN_ID};

/// Defines a method to compute the hash of this state as of a given [`TxnId`]
#[async_trait]
pub trait AsyncHash<FE> {
    /// The type of [`Transaction`] which this state supports
    type Txn: Transaction<FE>;

    /// Compute the hash of this state as of a given [`TxnId`]
    async fn hash(self, txn: &Self::Txn) -> TCResult<Output<Sha256>>;
}

/// Access a view which can be encoded with [`en::IntoStream`].
#[async_trait]
pub trait IntoView<'en, FE> {
    /// The type of [`Transaction`] which this state supports
    type Txn: Transaction<FE>;

    /// The type of encodable view returned by `into_view`
    type View: en::IntoStream<'en> + Sized + 'en;

    /// Return a `View` which can be encoded with [`en::IntoStream`].
    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View>;
}

/// Transaction lifecycle callbacks
#[async_trait]
pub trait Transact {
    /// A guard which blocks concurrent commits
    type Commit: Send + Sync;

    /// Commit this transaction.
    async fn commit(&self, txn_id: TxnId) -> Self::Commit;

    /// Roll back this transaction.
    async fn rollback(&self, txn_id: &TxnId);

    /// Delete any version data specific to this transaction.
    async fn finalize(&self, txn_id: &TxnId);
}

/// Common transaction context properties.
#[async_trait]
pub trait Transaction<FE>: Clone + Sized + Send + Sync + 'static {
    /// The [`TxnId`] of this transaction context.
    fn id(&'_ self) -> &'_ TxnId;

    /// Allows locking the filesystem directory of this transaction context,
    /// e.g. to cache un-committed state or to compute an intermediate result.
    fn context(&'_ self) -> &'_ DirLock<FE>;

    /// Create a new transaction context with the given `id`.
    async fn subcontext(&self, id: Id) -> TCResult<Self>;

    /// Create a new transaction subcontext with its own unique [`Dir`].
    async fn subcontext_unique(&self) -> TCResult<Self>;
}
