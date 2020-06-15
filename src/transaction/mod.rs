use async_trait::async_trait;

mod context;
mod txn;

pub type Txn = txn::Txn;
pub type TxnContext = context::TxnContext;
pub type TxnId = txn::TxnId;

#[async_trait]
pub trait Transact: Send + Sync {
    async fn commit(&self, txn_id: &TxnId);

    async fn rollback(&self, txn_id: &TxnId);
}
