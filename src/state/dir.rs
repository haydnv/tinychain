use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::state::chain::Chain;
use crate::state::TCState;
use crate::value::Link;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Dir {
    chain: Arc<Chain>,
}

#[async_trait]
impl TCContext for Dir {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _path: Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, _txn: Arc<Transaction>, _state: TCState) -> TCResult<()> {
        Err(error::not_implemented())
    }
}
