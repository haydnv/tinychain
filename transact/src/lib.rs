use async_trait::async_trait;

pub mod fs;
pub mod lock;
mod txn;

pub use txn::*;

#[async_trait]
pub trait Transact {
    async fn commit(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}
