use std::sync::Arc;

use async_trait::async_trait;

use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{TCResult, Value};

use super::index::Index;
use super::{Collect, GetResult};

pub struct Table {
    contents: Arc<Index>,
}

#[async_trait]
impl Collect for Table {
    type Selector = Vec<Value>;
    type Item = Vec<Value>;

    async fn get(self: Arc<Self>, txn: Arc<Txn>, selector: Self::Selector) -> GetResult {
        self.contents.clone().get(txn, selector).await
    }

    async fn put(
        &self,
        txn: &Arc<Txn>,
        selector: &Self::Selector,
        value: Self::Item,
    ) -> TCResult<()> {
        self.contents.put(txn, selector, value).await
    }
}

#[async_trait]
impl Transact for Table {
    async fn commit(&self, txn_id: &TxnId) {
        self.contents.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.contents.rollback(txn_id).await
    }
}
