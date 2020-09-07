use async_trait::async_trait;

use crate::collection::CollectionBase;
use crate::transaction::{Transact, TxnId};

#[derive(Clone)]
pub struct NullChain {
    collection: CollectionBase,
}

#[async_trait]
impl Transact for NullChain {
    async fn commit(&self, txn_id: &TxnId) {
        self.collection.commit(txn_id).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.collection.rollback(txn_id).await;
    }
}
