use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::error;
use crate::fs;
use crate::state::ChainContext;
use crate::state::TCState;
use crate::state::TableContext;
use crate::transaction::Transaction;
use crate::value::{Link, Op, TCValue};

#[derive(Debug)]
pub struct Host {
    chain_context: Arc<ChainContext>,
    table_context: Arc<TableContext>,
}

impl Host {
    pub fn new(data_dir: Arc<fs::Dir>) -> TCResult<Arc<Host>> {
        let chain_context = ChainContext::new(data_dir.reserve(&Link::to("/chain")?)?);
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

    pub fn new_transaction(self: &Arc<Self>, op: Op) -> TCResult<Arc<Transaction>> {
        Transaction::of(self.clone(), op)
    }

    pub async fn get(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        path: Link,
        _key: TCValue,
    ) -> TCResult<TCState> {
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
                "chain" => {
                    self.chain_context
                        .clone()
                        .post(txn, &path.slice_from(2))
                        .await
                }
                "table" => {
                    self.table_context
                        .clone()
                        .post(txn, &path.slice_from(2))
                        .await
                }
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }
}
