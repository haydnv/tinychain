//! Provides traits and data structures to define a distributed transaction context.
//!
//! This library is part of TinyChain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use async_hash::Output;
use async_trait::async_trait;
use destream::en;
use freqfs::DirLock;
use safecast::CastInto;

use tc_error::*;
use tc_value::{ToUrl, Value};
use tcgeneric::Id;

pub use id::{TxnId, MIN_ID};
use public::StateInstance;

pub mod fs;
mod id;
pub mod public;

pub mod lock {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use super::TxnId;

    pub use txn_lock::semaphore::{PermitRead, PermitWrite};

    /// A semaphore used to gate access to a transactional resource
    pub type Semaphore<C, R> = txn_lock::semaphore::Semaphore<TxnId, C, R>;

    /// A transactional read-write lock on a scalar value
    pub type TxnLock<T> = txn_lock::scalar::TxnLock<TxnId, T>;

    /// A read guard on a committed transactional version
    pub type TxnLockVersionGuard<T> = Arc<T>;

    /// A transactional read-write lock on a key-value map
    pub type TxnMapLock<K, V> = txn_lock::map::TxnMapLock<TxnId, K, V>;

    /// A read guard on a committed transactional version of a set
    pub type TxnMapLockVersionGuard<K, V> = HashMap<K, V>;

    /// A transactional read-write lock on a set of values
    pub type TxnSetLock<T> = txn_lock::set::TxnSetLock<TxnId, T>;

    /// A read guard on a committed transactional version of a set
    pub type TxnSetLockVersionGuard<T> = HashSet<T>;
}

/// Defines a method to compute the hash of this state as of a given [`TxnId`]
#[async_trait]
pub trait AsyncHash {
    /// Compute the hash of this state as of a given [`TxnId`]
    async fn hash(self, txn_id: TxnId) -> TCResult<Output<async_hash::Sha256>>;
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
    // TODO: this should take an owned TxnId
    async fn finalize(&self, txn_id: &TxnId);
}

/// Common transaction context properties.
#[async_trait]
pub trait Transaction<FE>: Clone + Sized + Send + Sync + 'static {
    /// The [`TxnId`] of this transaction context.
    fn id(&self) -> &TxnId;

    /// Allows locking the filesystem directory of this transaction context,
    /// e.g. to cache un-committed state or to compute an intermediate result.
    async fn context(&self) -> TCResult<DirLock<FE>>;

    /// Create a new transaction context with the given `id`.
    fn subcontext<I: Into<Id> + Send>(&self, id: I) -> Self;

    /// Create a new transaction subcontext with its own unique [`Dir`].
    fn subcontext_unique(&self) -> Self;
}

/// A transactional remote procedure call client
#[async_trait]
pub trait RPCClient<State: StateInstance<Txn = Self>>: Transaction<State::FE> {
    /// Resolve a GET op within this transaction context.
    async fn get<'a, L, V>(&'a self, link: L, key: V) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send;

    /// Resolve a PUT op within this transaction context.
    async fn put<'a, L, K, V>(&'a self, link: L, key: K, value: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        K: CastInto<Value> + Send,
        V: CastInto<State> + Send;

    /// Resolve a POST op within this transaction context.
    async fn post<'a, L, P>(&'a self, link: L, params: P) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        P: CastInto<State> + Send;

    /// Resolve a DELETE op within this transaction context.
    async fn delete<'a, L, V>(&'a self, link: L, key: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send;
}
