use std::convert::TryInto;
use std::sync::Arc;
use std::time;

use crate::error;
use crate::internal::block::Store;
use crate::internal::cache::Map;
use crate::internal::file::File;
use crate::internal::Directory;
use crate::state::{Persistent, State, Table};
use crate::transaction::Transaction;
use crate::value::{Link, Op, TCResult, TCValue};

const RESERVED: [&str; 1] = ["/sbin"];

pub struct Host {
    data_dir: Arc<Store>,
    workspace: Arc<Store>,
    root: Map<Link, Arc<Directory>>,
}

impl Host {
    pub async fn new(
        data_dir: Arc<Store>,
        workspace: Arc<Store>,
        hosted: Vec<Link>,
    ) -> TCResult<Arc<Host>> {
        let host = Arc::new(Host {
            data_dir,
            workspace,
            root: Map::new(),
        });

        let txn = Transaction::new(host.clone(), host.workspace.clone())?;
        for path in hosted {
            for reserved in RESERVED.iter() {
                if path.starts_with(reserved) {
                    return Err(error::bad_request(
                        "Attempted to host a reserved path",
                        reserved,
                    ));
                }
            }

            let dir = if host.data_dir.exists(&path).await? {
                Directory::from_store(host.data_dir.reserve(&path)?).await
            } else {
                Directory::create(txn.clone(), TCValue::None).await?
            };
            host.root.insert(path.clone(), dir);
        }

        Ok(host)
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
        if path.is_empty() {
            return Err(error::method_not_allowed(path));
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
                "table" => Ok(
                    Table::create(txn.clone(), txn.require("schema")?.try_into()?)
                        .await?
                        .into(),
                ),
                _ => Err(error::not_found(path)),
            },
            _ => Err(error::not_found(path)),
        }
    }
}
