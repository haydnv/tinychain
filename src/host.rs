use std::convert::TryInto;
use std::sync::Arc;
use std::time;

use crate::error;
use crate::internal::block::Store;
use crate::internal::cache::Map;
use crate::internal::file::File;
use crate::internal::Directory;
use crate::state::{Collection, Persistent, State, Table};
use crate::transaction::Transaction;
use crate::value::{Op, TCPath, TCResult, TCValue};

const RESERVED: [&str; 1] = ["/sbin"];

pub struct Host {
    data_dir: Arc<Store>,
    workspace: Arc<Store>,
    root: Map<TCPath, Arc<Directory>>,
}

impl Host {
    pub async fn new(
        data_dir: Arc<Store>,
        workspace: Arc<Store>,
        hosted: Vec<TCPath>,
    ) -> TCResult<Arc<Host>> {
        let host = Arc::new(Host {
            data_dir,
            workspace,
            root: Map::new(),
        });

        for path in hosted {
            for reserved in RESERVED.iter() {
                if path.to_string().starts_with(reserved) {
                    return Err(error::bad_request(
                        "Attempted to host a reserved path",
                        reserved,
                    ));
                }
            }

            let dir = if let Some(store) = host.data_dir.get_store(&path) {
                Directory::from_store(store).await
            } else {
                Directory::new(host.data_dir.reserve(path.clone())?)?
            };

            host.root.insert(path, dir);
        }

        Ok(host)
    }

    pub fn time(&self) -> u128 {
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    pub fn new_transaction(self: &Arc<Self>) -> TCResult<Arc<Transaction>> {
        Transaction::new(self.clone(), self.workspace.clone())
    }

    pub fn transact(self: &Arc<Self>, op: Op) -> TCResult<Arc<Transaction>> {
        Transaction::of(self.clone(), self.workspace.clone(), op)
    }

    pub async fn get(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        path: &TCPath,
        key: TCValue,
    ) -> TCResult<State> {
        println!("GET {}", path);
        if path.is_empty() {
            return Err(error::method_not_allowed(path));
        }

        if path[0] == "sbin" {
            match path[1].as_str() {
                "table" => Ok(Table::create(txn.clone(), key.try_into()?).await?.into()),
                "value" if path.len() > 2 => match path[2].as_str() {
                    "string" => {
                        let s: String = key.try_into()?;
                        Ok(State::Value(s.into()))
                    }
                    _ => Err(error::not_found(path)),
                },
                _ => Err(error::not_found(path)),
            }
        } else if let Some(dir) = self.root.get(&path[0].clone().into()) {
            let state = dir.get(txn.clone(), &path.slice_from(1)).await?;
            state.get(txn, key).await
        } else {
            Err(error::not_found(path))
        }
    }

    pub async fn put(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        path: TCPath,
        key: TCValue,
        state: State,
    ) -> TCResult<State> {
        println!("PUT {}", path);
        if path.is_empty() {
            Err(error::method_not_allowed(path))
        } else if let Some(dir) = self.root.get(&path[0].clone().into()) {
            let key: TCPath = key.try_into()?;
            let mut path = path.slice_from(1).clone();
            path.extend(key.into_iter());
            dir.put(txn, path, state).await?;
            Ok(().into())
        } else {
            Err(error::not_found(path))
        }
    }

    // TODO: remove this method
    pub async fn post(self: &Arc<Self>, _txn: Arc<Transaction>, path: &TCPath) -> TCResult<State> {
        println!("POST {}", path);
        if path.is_empty() {
            Ok(TCValue::None.into())
        } else {
            Err(error::not_found(path))
        }
    }
}
