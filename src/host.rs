use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::error;
use crate::fs;
use crate::state::chain::ChainContext;
use crate::state::table::TableContext;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, Op};

pub struct Host {
    chain_context: Arc<ChainContext>,
    table_context: Arc<TableContext>,
}

impl Host {
    pub fn new(data_dir: Arc<fs::Dir>) -> TCResult<Arc<Host>> {
        let chain_context = ChainContext::new(data_dir.reserve(Link::to("/chain")?)?);
        let table_context = TableContext::new();
        Ok(Arc::new(Host {
            chain_context,
            table_context,
        }))
    }

    pub fn time(&self) -> u128 {
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    pub fn new_transaction(self: Arc<Self>, op: Op) -> TCResult<Arc<Transaction>> {
        Transaction::of(self, op)
    }

    pub async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<TCState> {
        match path[0].as_str() {
            "sbin" => match path[1].as_str() {
                "chain" => {
                    self.chain_context
                        .clone()
                        .get(txn, path.from("/sbin/chain")?.into())
                        .await
                }
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }

    pub async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Link,
        state: TCState,
    ) -> TCResult<TCState> {
        if path.len() != 2 {
            return Err(error::not_found(path));
        }

        match path[0].as_str() {
            "sbin" => match path[1].as_str() {
                "chain" => {
                    self.chain_context
                        .clone()
                        .put(txn, path.from("/sbin/chain")?.into(), state)
                        .await
                }
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }

    pub async fn post(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<TCState> {
        match path[0].as_str() {
            "sbin" => match path[1].as_str() {
                "table" => {
                    self.table_context
                        .clone()
                        .post(txn, path.from("/sbin/table")?)
                        .await
                }
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }
}
