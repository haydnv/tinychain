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

    pub fn new_transaction(self: Arc<Self>) -> TCResult<Arc<Transaction>> {
        Transaction::new(self)
    }

    fn route(self: Arc<Self>, path: &Link) -> TCResult<(Arc<dyn TCContext>, Link)> {
        match path[0].as_str() {
            "sbin" => match path[1].as_str() {
                "block" => Ok((self.block_context.clone(), path.from("/sbin/block")?)),
                "chain" => Ok((self.chain_context.clone(), path.from("/sbin/chain")?)),
                "table" => Ok((self.table_context.clone(), path.from("/sbin/table")?)),
                "value" => Ok((self.value_context.clone(), path.from("/sbin/value")?)),
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }

    pub async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<Arc<TCState>> {
        let (context, child_path) = self.route(&path)?;
        match context.get(txn, child_path).await {
            Ok(state) => Ok(state),
            Err(cause) => Err(error::TCError::of(
                cause.reason().clone(),
                format!("\n{}: {}", path, cause.message()),
            )),
        }
    }

    pub async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Link,
        value: TCValue,
    ) -> TCResult<()> {
        let (context, child_path) = self.route(&path)?;
        if child_path.as_str() != "/" {
            return Err(error::method_not_allowed());
        }

        match context.put(txn, value).await {
            Ok(state) => Ok(state),
            Err(cause) => Err(error::TCError::of(
                cause.reason().clone(),
                format!("\n{}: {}", path, cause.message()),
            )),
        }
    }

    pub async fn post(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Link,
    ) -> TCResult<Arc<TCState>> {
        let (context, child_path) = self.route(&path)?;
        match context.post(txn, child_path).await {
            Ok(state) => Ok(state),
            Err(cause) => Err(error::TCError::of(
                cause.reason().clone(),
                format!("\n{}: {}", path, cause.message()),
            )),
        }
    }
}
