use std::convert::TryInto;
use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::error;
use crate::internal::FsDir;
use crate::state::{Dir, DirContext, TCState, TableContext};
use crate::transaction::Transaction;
use crate::value::{Link, Op, TCValue};

const RESERVED: [&str; 1] = ["/sbin"];

#[derive(Debug)]
pub struct Host {
    dir_context: Arc<DirContext>,
    table_context: Arc<TableContext>,
    workspace: Arc<FsDir>,
    root: Arc<Dir>,
}

impl Host {
    pub fn new(data_dir: Arc<FsDir>, workspace: Arc<FsDir>) -> TCResult<Arc<Host>> {
        let dir_context = DirContext::new();
        let table_context = TableContext::new();

        let root = Dir::new(data_dir)?;

        Ok(Arc::new(Host {
            dir_context,
            table_context,
            workspace,
            root,
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

        self.root
            .put(
                txn.clone(),
                path.into(),
                self.dir_context
                    .post(txn.clone(), &"/new".try_into()?)
                    .await?,
            )
            .await?;
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

    pub async fn get(self: &Arc<Self>, path: &Link, key: TCValue) -> TCResult<TCState> {
        println!("GET {}", path);
        if path.len() < 3 {
            return Err(error::not_found(path));
        }

        match path.as_str(0) {
            "sbin" => match path.as_str(1) {
                "value" => match path.as_str(2) {
                    "string" => {
                        let s: String = key.try_into()?;
                        Ok(TCState::Value(s.into()))
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
        _state: TCState,
    ) -> TCResult<TCState> {
        println!("PUT {}", path);
        Err(error::not_found(path))
    }

    pub async fn post(self: &Arc<Self>, txn: Arc<Transaction>, path: &Link) -> TCResult<TCState> {
        println!("POST {}", path);
        if path.is_empty() {
            return Ok(TCValue::None.into());
        } else if path.len() < 3 {
            return Err(error::not_found(path));
        }

        match path.as_str(0) {
            "sbin" => match path.as_str(1) {
                "table" => self.table_context.post(txn, &path.slice_from(2)).await,
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }
}
