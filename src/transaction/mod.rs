use std::sync::Arc;

use async_trait::async_trait;

pub mod lock;
mod server;
mod txn;

pub type Txn = txn::Txn;
pub type TxnId = txn::TxnId;
pub type TxnServer = server::TxnServer;

#[async_trait]
pub trait Transact: Send + Sync {
    async fn commit(&self, txn_id: &TxnId);

    async fn rollback(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}

#[async_trait]
impl<T: Transact> Transact for Arc<T> {
    async fn commit(&self, txn_id: &TxnId) {
        <T as Transact>::commit(self, txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        <T as Transact>::rollback(self, txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        <T as Transact>::finalize(self, txn_id).await
    }
}
