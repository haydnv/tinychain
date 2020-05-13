use std::convert::TryInto;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time;

use structopt::StructOpt;

use crate::error;
use crate::http;
use crate::internal::block::Store;
use crate::internal::cache::Map;
use crate::internal::file::File;
use crate::internal::Directory;
use crate::state::{Collection, Persistent, State, Table};
use crate::transaction::Txn;
use crate::value::{Args, Link, TCPath, TCResult, TCValue};

const RESERVED: [&str; 1] = ["/sbin"];

#[derive(Clone, StructOpt)]
pub struct HostConfig {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[structopt(long = "data_dir", default_value = "/tmp/tc/data")]
    pub data_dir: PathBuf,

    #[structopt(long = "workspace", default_value = "/tmp/tc/tmp")]
    pub workspace: PathBuf,

    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "host")]
    pub hosted: Vec<TCPath>,

    #[structopt(long = "peer")]
    pub peers: Vec<Link>,
}

#[derive(Clone)]
pub struct NetworkTime {
    nanos: u128,
}

impl NetworkTime {
    pub fn as_millis(&self) -> u64 {
        (self.nanos / 1_000_000).try_into().unwrap()
    }

    pub fn as_nanos(&self) -> u128 {
        self.nanos
    }

    pub fn from_nanos(nanos: u128) -> NetworkTime {
        NetworkTime { nanos }
    }
}

impl std::ops::Add<std::time::Duration> for NetworkTime {
    type Output = Self;

    fn add(self, other: std::time::Duration) -> Self {
        NetworkTime {
            nanos: self.nanos + other.as_nanos(),
        }
    }
}

pub struct Host {
    address: IpAddr,
    http_port: u16,
    data_dir: Arc<Store>,
    workspace: Arc<Store>,
    root: Map<TCPath, Arc<Directory>>,
}

impl Host {
    pub async fn new(config: HostConfig) -> TCResult<Arc<Host>> {
        // TODO: figure out a way to populate `root` without locking

        let data_dir = Store::new(config.data_dir);
        let workspace = Store::new_tmp(config.workspace);

        let host = Arc::new(Host {
            address: config.address,
            http_port: config.http_port,
            data_dir,
            workspace,
            root: Map::new(),
        });

        let txn = host.new_transaction().await?;
        let txn_id = &txn.id();

        for path in config.hosted {
            for reserved in RESERVED.iter() {
                if path.to_string().starts_with(reserved) {
                    return Err(error::bad_request(
                        "Attempted to host a reserved path",
                        reserved,
                    ));
                }
            }

            let dir = if let Some(store) = host.data_dir.get_store(txn_id, &path).await {
                Directory::from_store(txn_id, store).await
            } else {
                Directory::new(txn_id, host.data_dir.reserve(txn_id, path.clone()).await?).await?
            };

            host.root.insert(path, dir);
        }

        host.data_dir.commit(&txn.id()).await;

        Ok(host)
    }

    pub async fn http_listen(
        self: &Arc<Self>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        http::listen(self.clone(), &(self.address, self.http_port).into()).await
    }

    pub fn time(&self) -> NetworkTime {
        NetworkTime::from_nanos(
            time::SystemTime::now()
                .duration_since(time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        )
    }

    pub async fn new_transaction(self: &Arc<Self>) -> TCResult<Arc<Txn>> {
        Txn::new(self.clone(), self.workspace.clone()).await
    }

    pub async fn get(
        self: &Arc<Self>,
        txn: Arc<Txn>,
        link: &Link,
        key: TCValue,
    ) -> TCResult<State> {
        println!("GET {}", link);
        if let Some(host) = link.host() {
            if host.address() != &self.address {
                return Err(error::not_implemented());
            }
        }

        let path = link.path();
        if path.is_empty() {
            return Err(error::method_not_allowed(path));
        }

        if path[0] == "sbin" && path.len() > 2 {
            match path[1].as_str() {
                "state" => match path[2].as_str() {
                    "table" => Ok(Table::create(txn.clone(), key.try_into()?).await?.into()),
                    _ => Err(error::not_found(path)),
                },
                "value" => match path[2].as_str() {
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
        txn: Arc<Txn>,
        dest: Link,
        key: TCValue,
        state: State,
    ) -> TCResult<State> {
        println!("PUT {}", dest);
        if let Some(host) = dest.host() {
            if host.address() != &self.address {
                return Err(error::not_implemented());
            }
        }

        let path = dest.path();

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
    pub async fn post(
        self: &Arc<Self>,
        _txn: Arc<Txn>,
        dest: &Link,
        _args: Args,
    ) -> TCResult<State> {
        println!("POST {}", dest);
        if let Some(host) = dest.host() {
            if host.address() != &self.address {
                return Err(error::not_implemented());
            }
        }

        if dest.path().is_empty() {
            Ok(TCValue::None.into())
        } else {
            Err(error::not_found(dest))
        }
    }
}
