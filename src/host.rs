use std::convert::TryInto;
use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::error;
use crate::internal::FsDir;
use crate::state::{State, TableContext};
use crate::transaction::Transaction;
use crate::value::{Link, Op, TCValue};

const RESERVED: [&str; 1] = ["/sbin"];

#[derive(Debug)]
pub struct Host {
    table_context: Arc<TableContext>,
    workspace: Arc<FsDir>,
}

impl Host {
    pub fn new(_data_dir: Arc<FsDir>, workspace: Arc<FsDir>) -> TCResult<Arc<Host>> {
        let table_context = TableContext::new();

        Ok(Arc::new(Host {
            table_context,
            workspace,
        }))
    }

    pub async fn claim(self: &Arc<Self>, path: Link) -> TCResult<()> {
        let txn = Transaction::new(self.clone(), self.workspace.clone())?;
        for reserved in RESERVED.iter() {
            if path.starts_with(reserved) {
                return Err(error::bad_request(
                    "Attempted to host a reserved path",
                    reserved,
                ));
            }
        }

        // TODO

        txn.commit().await;
        Ok(())
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

    pub async fn get(self: &Arc<Self>, path: &Link, key: TCValue) -> TCResult<State> {
        println!("GET {}", path);
        if path.len() < 3 {
            return Err(error::not_found(path));
        }

        match path.as_str(0) {
            "sbin" => match path.as_str(1) {
                "value" => match path.as_str(2) {
                    "string" => {
                        let s: String = key.try_into()?;
                        Ok(State::Value(s.into()))
                    }
                    _ => Err(error::not_found(path)),
                },
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }

    pub async fn put(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        path: Link,
        _key: TCValue,
        _state: State,
    ) -> TCResult<State> {
        println!("PUT {}", path);
        Err(error::not_found(path))
    }

    pub async fn post(self: &Arc<Self>, txn: Arc<Transaction>, path: &Link) -> TCResult<State> {
        println!("POST {}", path);
        if path.is_empty() {
            return Ok(TCValue::None.into());
        } else if path.len() < 3 {
            return Err(error::not_found(path));
        }

        match path.as_str(0) {
            "sbin" => match path.as_str(1) {
                "table" => self.table_context.new_table(txn, &path.slice_from(2)).await,
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }
}
