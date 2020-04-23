use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::state::State;
use crate::transaction::{Transaction, TransactionId};
use crate::value::TCValue;

#[derive(Debug)]
pub struct Graph {}

#[async_trait]
impl Persistent for Graph {
    async fn commit(self: &Arc<Self>, _txn_id: TransactionId) {
        // TODO
    }

    async fn get(self: &Arc<Self>, _txn: Arc<Transaction>, _node_id: &TCValue) -> TCResult<State> {
        Err(error::not_implemented())
    }

    async fn put(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        _node_id: TCValue,
        _node: State,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }
}
