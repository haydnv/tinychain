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
    table_context: Arc<TableContext>,
    value_context: Arc<ValueContext>,
}

impl Host {
    pub fn new(workspace: Arc<Drive>) -> Host {
        let block_context = BlockContext::new(workspace);
        let chain_context = ChainContext::new(block_context);
        let table_context = TableContext::new(chain_context);
        let value_context = ValueContext::new();
        Host {
            table_context,
            value_context,
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
        let segments: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();

        match segments[..2] {
            ["sbin", "table"] => {
                self.table_context
                    .clone()
                    .get(txn, path.from("/sbin/table")?)
                    .await
            }
            ["sbin", "value"] => {
                self.value_context
                    .clone()
                    .get(txn, path.from("/sbin/value")?)
                    .await
            }
            _ => Err(error::not_found(path)),
        }
    }

    pub async fn post(
        self: Arc<Self>,
        path: Link,
        txn: Arc<Transaction>,
    ) -> TCResult<Arc<TCState>> {
        let segments = path.segments();
        let segments: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();

        let resource = match segments[..2] {
            ["sbin", "table"] => self.table_context.clone(),
            _ => {
                return Err(error::not_found(path));
            }
        };

        match resource.post(txn, &segments[2..].join("/")).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                let reason = cause.reason();
                let message = cause.message();
                Err(error::TCError::of(
                    reason.clone(),
                    format!("{}: {}", path, message),
                ))
            }
        }
    }
}
