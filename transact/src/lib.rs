use async_trait::async_trait;
use std::convert::TryFrom;

use error::*;

pub mod fs;
mod id;
pub mod lock;

pub use id::TxnId;

#[async_trait]
pub trait Transact {
    async fn commit(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}

#[async_trait]
pub trait Transaction<E: fs::FileEntry>: Sized {
    type Subcontext: Transaction<E>;

    fn id(&'_ self) -> &'_ TxnId;

    async fn context<B: fs::BlockData>(&self) -> TCResult<fs::File<B>>
    where
        fs::File<B>: TryFrom<E>;

    async fn subcontext(&self) -> TCResult<Self::Subcontext>;
}
