use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::state::chain::Chain;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

#[derive(Hash)]
pub struct Dir {
    chain: Arc<Chain>,
}

#[async_trait]
impl TCContext for Dir {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, path: TCValue) -> TCResult<TCState> {
        let _path: Link = path.try_into()?;

        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: TCValue,
        state: TCState,
    ) -> TCResult<TCState> {
        let path: Link = path.try_into()?;

        let constructor = match state {
            TCState::Chain(_) => Link::to("/sbin/chain")?,
            TCState::Value(val) => {
                return Err(error::bad_request(
                    "A Dir can only store States, not Values--found",
                    val,
                ));
            }
            _ => return Err(error::not_implemented()),
        };

        txn.clone()
            .put(txn.clone().context().append(&path), state)
            .await?;

        self.chain
            .clone()
            .put(txn, path.into(), constructor.into())
            .await?;

        Ok(().into())
    }
}
