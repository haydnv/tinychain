use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::state::chain::Chain;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::Link;

#[derive(Hash)]
pub struct Dir {
    chain: Arc<Chain>,
}

#[async_trait]
impl TCContext for Dir {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, _path: Link) -> TCResult<TCState> {
        Err(error::not_implemented())
    }

    async fn put(self: Arc<Self>, txn: Arc<Transaction>, args: TCState) -> TCResult<()> {
        let path: Link = args.get_arg("path")?.to_value()?.to_link()?;
        let state = args.get_arg("state")?;

        if let TCState::Value(val) = state {
            return Err(error::bad_request(
                "A Dir can only store States, not Values--found",
                val,
            ));
        }

        txn.clone()
            .put(txn.clone().context().append(&path), state)
            .await?;

        self.chain.clone().put(txn, path.into()).await?;

        Ok(())
    }
}
