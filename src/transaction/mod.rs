use async_trait::async_trait;

pub mod lock;
mod txn;

pub type Txn = txn::Txn;
pub type TxnId = txn::TxnId;

#[async_trait]
pub trait Transact: Send + Sync {
    async fn commit(&self, txn_id: &TxnId);

    async fn rollback(&self, txn_id: &TxnId);
}
