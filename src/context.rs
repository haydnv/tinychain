use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::state::State;
use crate::transaction::{Transaction, TransactionId};
use crate::value::TCValue;

pub type TCResult<T> = Result<T, error::TCError>;

#[async_trait]
pub trait Persistent: Send + Sync {
    type Key: TryFrom<TCValue>;

    async fn commit(self: &Arc<Self>, txn_id: TransactionId);

    async fn get(self: &Arc<Self>, txn: Arc<Transaction>, key: &Self::Key) -> TCResult<State>;

    async fn put(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        key: Self::Key,
        state: State,
    ) -> TCResult<State>;
}
