use async_trait::async_trait;

pub mod fs;
mod id;
pub mod lock;

pub use id::TxnId;

#[async_trait]
pub trait Transact {
    async fn commit(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}

pub trait Transaction {
    fn id(&'_ self) -> &'_ TxnId;
}
