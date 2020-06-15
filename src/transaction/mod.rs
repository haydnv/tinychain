use async_trait::async_trait;

mod state;
mod txn;

pub type Txn = txn::Txn;
pub type TxnId = txn::TxnId;
pub type TxnState = state::TxnState;

#[async_trait]
pub trait Transact: Send + Sync {
    async fn commit(&self, txn_id: &TxnId);

    async fn rollback(&self, txn_id: &TxnId);
}
