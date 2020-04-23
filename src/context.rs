use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::state::TCState;
use crate::transaction::{Transaction, TransactionId};
use crate::value::TCValue;

pub type TCResult<T> = Result<T, error::TCError>;

#[async_trait]
pub trait TCContext: Send + Sync {
    async fn commit(self: &Arc<Self>, txn_id: TransactionId);

    async fn get(self: &Arc<Self>, txn: Arc<Transaction>, key: &TCValue) -> TCResult<TCState>;

    async fn put(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        key: TCValue,
        state: TCState,
    ) -> TCResult<TCState>;
}
