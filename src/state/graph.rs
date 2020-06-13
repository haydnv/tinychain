use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{TCResult, Value};

use super::{Collect, GetResult};

pub struct Graph;

#[async_trait]
impl Collect for Graph {
    type Selector = Vec<Value>;
    type Item = Vec<Value>;

    async fn get(self: Arc<Self>, _txn: Arc<Txn>, _selector: Self::Selector) -> GetResult {
        Err(error::not_implemented())
    }

    async fn put(
        &self,
        _txn: &Arc<Txn>,
        _selector: &Self::Selector,
        _value: Self::Item,
    ) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Transact for Graph {
    async fn commit(&self, _txn_id: &TxnId) {
        // TODO
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // TODO
    }
}
