use std::sync::Arc;
use std::time;

use async_trait::async_trait;

use crate::context::*;
use crate::drive::Drive;
use crate::error;
use crate::table::TableContext;
use crate::transaction::{Pending, Transaction};

pub struct HostContext {
    table_context: Arc<TableContext>,
}

impl HostContext {
    pub fn new(_workspace: Drive) -> HostContext {
        let table_context = TableContext::new();
        HostContext { table_context }
    }

    pub fn time(&self) -> u128 {
        let since_the_epoch = time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap();
        since_the_epoch.as_nanos()
    }

    pub fn transaction(self: Arc<Self>) -> Arc<Transaction> {
        Transaction::new(self)
    }
}

#[async_trait]
impl TCContext for HostContext {
    fn post(self: Arc<Self>, path: String) -> TCResult<Pending> {
        if !path.starts_with('/') {
            return Err(error::bad_request(
                "Expected an absolute path starting with '/' but found",
                path,
            ));
        }

        let segments: Vec<&str> = path[1..].split('/').collect();

        match segments[..2] {
            ["sbin", "table"] => self.table_context.clone().post(segments[2..].join("/")),
            _ => Err(error::not_found(path)),
        }
    }
}
