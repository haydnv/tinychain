use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;

use crate::chain::Chain;
use crate::context::*;
use crate::error;
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

    async fn get(self: &Arc<Self>, _txn: Arc<Transaction>, _path: &TCValue) -> TCResult<TCState> {
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

struct DirContext;

#[async_trait]
impl TCExecutable for DirContext {
    async fn post(self: &Arc<Self>, txn: Arc<Transaction>, method: &Link) -> TCResult<TCState> {
        if method != "/new" {
            return Err(error::bad_request("DirContext has no such method", method));
        }

        Ok(Arc::new(Dir {
            chain: Chain::new(txn.context().reserve(&"/.chain".try_into()?)?),
        })
        .into())
    }
}
