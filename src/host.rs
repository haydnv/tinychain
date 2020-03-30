use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::drive::Drive;
use crate::transaction::Transaction;

pub struct HostContext {
    workspace: Drive,
}

impl HostContext {
    pub fn new(workspace: Drive) -> HostContext {
        HostContext { workspace }
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

impl TCContext for HostContext {}
