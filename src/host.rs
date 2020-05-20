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
use crate::object::actor::{Actor, Token};
use crate::state::{Cluster, Collection, Directory, Persistent, State, Table};
use crate::transaction::Txn;
use crate::value::link::{Link, LinkHost, TCPath};
use crate::value::{TCResult, TCValue};

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
    pub peers: Vec<LinkHost>,
}

#[derive(Clone)]
pub struct NetworkTime {
    nanos: u128,
}

impl NetworkTime {
    pub fn as_millis(&self) -> u64 {
        const MILLIS_PER_NANO: u128 = 1_000_000;
        (self.nanos / MILLIS_PER_NANO).try_into().unwrap()
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

    pub async fn new_transaction<'a>(self: &Arc<Self>) -> TCResult<Arc<Txn<'a>>> {
        Txn::new(self.clone(), self.workspace.clone()).await
    }

    pub async fn get(
        self: &Arc<Self>,
        txn: &Arc<Txn<'_>>,
        link: &Link,
        key: TCValue,
        auth: &Option<Token>,
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

        if path[0] == "sbin" {
            match path[1].as_str() {
                "object" if path.len() > 2 => match path[2].as_str() {
                    "actor" => {
                        let actor: Actor = key.try_into()?;
                        Ok(actor.into())
                    }
                    _ => Err(error::not_found(path)),
                },
                "state" if path.len() > 2 => match path[2].as_str() {
                    "cluster" => {
                        let cluster = Cluster::create(txn, TCValue::None).await?;

                        let mut dest: TCPath = key.try_into()?;
                        let name: TCPath = if let Some(name) = dest.pop() {
                            name.into()
                        } else {
                            return Err(error::bad_request("Cluster context cannot be '/'", dest));
                        };

                        let dir = self
                            .put(txn, dest.into(), name.clone().into(), cluster.into(), auth)
                            .await?;
                        dir.get(txn, name.into(), auth).await
                    }
                    "table" => Ok(Table::create(txn, key.try_into()?).await?.into()),
                    _ => Err(error::not_found(path)),
                },
                "value" if path.len() == 2 => Ok(State::Value(key)),
                "value" if path.len() > 2 => match path[2].as_str() {
                    "link" if path.len() == 3 => {
                        let link: Link = key.try_into()?;
                        Ok(State::Value(link.into()))
                    }
                    "link" if path.len() > 3 => match path[3].as_str() {
                        "host" => {
                            let address: Link = key.try_into()?;
                            let address: LinkHost = address.try_into()?;
                            let address: Link = address.into();
                            Ok(State::Value(address.into()))
                        }
                        _ => Err(error::not_found(path)),
                    },
                    "string" => {
                        let s: String = key.try_into()?;
                        Ok(State::Value(s.into()))
                    }
                    _ => Err(error::not_found(path)),
                },
                _ => Err(error::not_found(path)),
            }
        } else if let Some(dir) = self.root.get(&path[0].clone().into()) {
            let state = dir.get(txn, &path.slice_from(1), auth).await?;
            state.get(txn, key, auth).await
        } else {
            Err(error::not_found(path))
        }
    }

    pub async fn put(
        self: &Arc<Self>,
        txn: &Arc<Txn<'_>>,
        dest: Link,
        key: TCValue,
        state: State,
        auth: &Option<Token>,
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
            Ok(dir.put(txn, path, state, auth).await?.into())
        } else {
            Err(error::not_found(path))
        }
    }
}
