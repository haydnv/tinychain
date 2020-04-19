use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::state::chain::Chain;
use crate::state::TCState;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCValue};

#[derive(Hash)]
pub struct Dir {
    chain: Arc<Chain>,
}

#[async_trait]
impl TCContext for Dir {
    async fn commit(self: &Arc<Self>, _txn_id: TransactionId) {
        // TODO
    }

    async fn get(self: &Arc<Self>, _txn: Arc<Transaction>, path: TCValue) -> TCResult<TCState> {
        let _path: Link = path.try_into()?;

        Err(error::not_implemented())
    }

    async fn put(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        path: TCValue,
        _state: TCState,
    ) -> TCResult<TCState> {
        let _path: Link = path.try_into()?;

        Err(error::not_implemented())
    }
}
