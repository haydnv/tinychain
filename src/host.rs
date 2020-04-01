use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::drive::Drive;
use crate::error;
use crate::state::block::BlockContext;
use crate::state::chain::ChainContext;
use crate::state::table::TableContext;
use crate::state::value::ValueContext;
use crate::transaction::Transaction;

pub struct Host {
    block_context: Arc<BlockContext>,
    chain_context: Arc<ChainContext>,
    table_context: Arc<TableContext>,
    value_context: Arc<ValueContext>,
}

impl Host {
    pub fn new(workspace: Arc<Drive>) -> Host {
        Host {
            block_context: BlockContext::new(workspace),
            chain_context: ChainContext::new(),
            table_context: TableContext::new(),
            value_context: ValueContext::new(),
        }
    }

    pub fn time(&self) -> u128 {
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    pub fn transaction(self: Arc<Self>) -> Arc<Transaction> {
        Transaction::new(self)
    }

    pub async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<Arc<TCState>> {
        let segments = path.segments();
        match segments[0] {
            "sbin" => match segments[1] {
                "table" => {
                    self.table_context
                        .clone()
                        .get(txn, path.from("/sbin/table")?)
                        .await
                }
                "value" => {
                    self.value_context
                        .clone()
                        .get(txn, path.from("/sbin/value")?)
                        .await
                }
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }

    pub async fn post(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Link,
    ) -> TCResult<Arc<TCState>> {
        let segments = path.segments();

        match segments[0] {
            "sbin" => match segments[1] {
                "table" => {
                    self.table_context
                        .clone()
                        .post(txn, path.from("/sbin/table")?)
                        .await
                }
                "value" => {
                    self.value_context
                        .clone()
                        .post(txn, path.from("/sbin/value")?)
                        .await
                }
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }
}
