//! Provides traits and data structures to define a distributed transaction context.
//!
//! This library is part of Tinychain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use async_trait::async_trait;
use destream::en;

use tc_error::*;
use tcgeneric::Id;

mod id;

pub mod fs;
pub mod lock;

pub use id::TxnId;

#[async_trait]
pub trait IntoView<'en, D: fs::Dir> {
    type Txn: Transaction<D>;
    type View: en::IntoStream<'en> + Sized;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View>;
}

/// Transaction lifecycle callbacks
#[async_trait]
pub trait Transact {
    /// Commit this transaction.
    async fn commit(&self, txn_id: &TxnId);

    /// Delete any version data specific to this transaction.
    async fn finalize(&self, txn_id: &TxnId);
}

/// Common transaction context properties.
#[async_trait]
pub trait Transaction<D: fs::Dir>: Sized {
    /// The [`TxnId`] of this transaction context.
    fn id(&'_ self) -> &'_ TxnId;

    /// The [`fs::Dir`] of this transaction context.
    fn context(&'_ self) -> &'_ D;

    /// A transaction subcontext with its own [`fs::Dir`].
    async fn subcontext(&self, id: Id) -> TCResult<Self>;

    /// A transaction subcontext with its own unique [`fs::Dir`].
    async fn subcontext_tmp(&self) -> TCResult<Self>;
}
