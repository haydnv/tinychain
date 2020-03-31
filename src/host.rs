use std::sync::Arc;
use std::time;

use async_trait::async_trait;

use crate::context::*;
use crate::drive::Drive;
use crate::error;
use crate::state::block::BlockContext;
use crate::state::chain::ChainContext;
use crate::state::table::TableContext;
use crate::transaction::Transaction;

pub struct HostContext {
    table_context: Arc<TableContext>,
}

impl HostContext {
    pub fn new(workspace: Arc<Drive>) -> HostContext {
        let block_context = BlockContext::new(workspace);
        let chain_context = ChainContext::new(block_context);
        let table_context = TableContext::new(chain_context);
        HostContext { table_context }
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
}

#[async_trait]
impl TCContext for HostContext {
    fn post(self: Arc<Self>, path: String, txn: Arc<Transaction>) -> TCResult<Arc<TCState>> {
        if !path.starts_with('/') {
            return Err(error::bad_request(
                "Expected an absolute path starting with '/' but found",
                path,
            ));
        }

        let segments: Vec<&str> = path[1..].split('/').collect();

        match segments[..2] {
            ["sbin", "table"] => self
                .table_context
                .clone()
                .post(segments[2..].join("/"), txn),
            _ => Err(error::not_found(path)),
        }
    }
}
