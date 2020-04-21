use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::error;
use crate::fs;
use crate::state::TCState;
use crate::state::TableContext;
use crate::transaction::Transaction;
use crate::value::{Link, Op, TCValue, ValueContext};

#[derive(Debug)]
pub struct Host {
    table_context: Arc<TableContext>,
    value_context: Arc<ValueContext>,
    workspace: Arc<fs::Dir>,
}

impl Host {
    pub fn new(
        _data_dir: Arc<fs::Dir>,
        workspace: Arc<fs::Dir>,
        _hosted: Vec<Link>,
    ) -> TCResult<Arc<Host>> {
        let table_context = TableContext::new();
        let value_context = ValueContext::new();

        Ok(Arc::new(Host {
            table_context,
            value_context,
            workspace,
        }))
    }

    pub fn time(&self) -> u128 {
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    pub fn new_transaction(self: &Arc<Self>, op: Op) -> TCResult<Arc<Transaction>> {
        Transaction::of(self.clone(), op, self.workspace.clone())
    }

    pub async fn get(self: &Arc<Self>, path: Link, _key: TCValue) -> TCResult<TCState> {
        println!("GET {}", path);
        Err(error::not_found(path))
    }

    pub async fn put(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        path: Link,
        _key: TCValue,
        _state: TCState,
    ) -> TCResult<TCState> {
        println!("PUT {}", path);
        Err(error::not_found(path))
    }

    pub async fn post(self: &Arc<Self>, txn: Arc<Transaction>, path: &Link) -> TCResult<TCState> {
        println!("POST {}", path);
        if path.is_empty() {
            return Ok(TCValue::None.into());
        }

        match path.as_str(0) {
            "sbin" => match path.as_str(1) {
                "table" => self.table_context.post(txn, &path.slice_from(2)).await,
                "value" => self.value_context.post(txn, &path.slice_from(2)).await,
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }
}
