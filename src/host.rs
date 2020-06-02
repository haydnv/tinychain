use std::convert::TryInto;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time;

use bytes::Bytes;
use sha2::{Digest, Sha256};
use structopt::StructOpt;

use crate::auth::Token;
use crate::error;
use crate::gateway::Hosted;
use crate::http;
use crate::internal::file::File;
use crate::internal::Dir;
use crate::state::*;
use crate::transaction::{Transact, Txn};
use crate::value::link::{Link, LinkHost, TCPath};
use crate::value::{TCResult, Value, ValueId};

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

#[derive(Clone)]
pub struct Host {
    address: IpAddr,
    http_port: u16,
    data_dir: Arc<Dir>,
    workspace: Arc<Dir>,
    root: Hosted,
}

impl Host {
    pub async fn new(config: HostConfig) -> TCResult<Arc<Host>> {
        // TODO: figure out a way to populate `root` without locking

        let data_dir = Dir::new(config.data_dir);
        let workspace = Dir::new_tmp(config.workspace);

        let mut hosted = config.hosted;
        hosted.sort_by(|a, b| b.len().partial_cmp(&a.len()).unwrap());

        let mut host = Host {
            address: config.address,
            http_port: config.http_port,
            data_dir,
            workspace,
            root: Hosted::new(),
        };

        let txn = Arc::new(host.clone()).new_transaction().await?;
        let txn_id = &txn.id();

        while let Some(path) = hosted.pop() {
            for reserved in RESERVED.iter() {
                if path.to_string().starts_with(reserved) {
                    return Err(error::bad_request(
                        "Attempted to host a reserved path",
                        reserved,
                    ));
                }
            }

            if host.root.get(&path).is_some() {
                return Err(error::bad_request(
                    "Cannot host a subdirectory of a hosted directory",
                    path,
                ));
            }

            let hosted_cluster = if let Some(dir) = host.data_dir.get_dir(txn_id, &path).await? {
                cluster::Cluster::from_dir(txn_id, dir).await
            } else {
                cluster::Cluster::new(
                    txn_id,
                    host.data_dir.create_dir(txn_id, path.clone()).await?,
                )
                .await?
            };

            host.root.push(path, hosted_cluster);
        }

        host.data_dir.commit(&txn.id()).await;
        if host.data_dir.is_empty().await {
            panic!("Committed to data_dir but it's still empty!");
        }

        Ok(Arc::new(host))
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
        key: Value,
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
            let name = &path[1];
            let path = &path.slice_from(2);
            match name.as_str() {
                "auth" => Ok(Sbin::auth(path, key)?.into()),
                "state" => Sbin::state(txn, path, key).await,
                "value" => Ok(Sbin::value(path, key)?.into()),
                _ => Err(error::not_found(path)),
            }
        } else if let Some((path, cluster)) = self.root.get(path) {
            let state = cluster.get(txn, &path).await?;
            state.get(txn, key, auth).await
        } else {
            Err(error::not_found(path))
        }
    }

    pub async fn put(
        self: &Arc<Self>,
        txn: &Arc<Txn<'_>>,
        dest: Link,
        key: Value,
        state: State,
        _auth: &Option<Token>,
    ) -> TCResult<State> {
        // TODO: authorize
        println!("PUT {}", dest);
        if let Some(host) = dest.host() {
            if host.address() != &self.address {
                return Err(error::not_implemented());
            }
        }

        let path = dest.path();

        if path.is_empty() {
            Err(error::method_not_allowed(path))
        } else if path[0] == "sbin" {
            let name = &path[1];
            let path = &path.slice_from(2);
            match name.as_str() {
                "auth" => Err(error::method_not_allowed(path)),
                "state" => Err(error::method_not_allowed(path)),
                "value" => Err(error::method_not_allowed(path)),
                _ => Err(error::not_found(path)),
            }
        } else if let Some((mut path, cluster)) = self.root.get(path) {
            let key: TCPath = key.try_into()?;
            path.extend(key.into_iter());
            Ok(cluster.clone().put(txn, path, state).await?)
        } else {
            Err(error::not_found(path))
        }
    }
}

struct Sbin;

impl Sbin {
    fn auth(path: &TCPath, key: Value) -> TCResult<Value> {
        match path.to_string().as_str() {
            "/hash/sha256" => {
                let data: String = key.try_into()?;
                let data: Bytes = data.into();

                let mut hasher = Sha256::new();
                hasher.input(data);
                Ok(Bytes::copy_from_slice(&hasher.result()[..]).into())
            }
            _ => Err(error::not_found(path)),
        }
    }

    async fn state(txn: &Arc<Txn<'_>>, path: &TCPath, key: Value) -> TCResult<State> {
        match path.to_string().as_str() {
            "/table" => {
                let args: Args = key.try_into()?;
                Ok(table::Table::create(txn, args.try_into()?).await?.into())
            }
            _ => Err(error::not_found(path)),
        }
    }

    fn value(path: &TCPath, key: Value) -> TCResult<Value> {
        match path.to_string().as_str() {
            "/" => Ok(key),
            "/bytes" => {
                let encoded: String = key.try_into()?;
                let decoded = base64::decode(encoded)
                    .map_err(|e| error::bad_request("Unable to decode base64 string", e))?;
                Ok(Value::Bytes(Bytes::from(decoded)))
            }
            "/id" => {
                let id: ValueId = key.try_into()?;
                Ok(id.into())
            }
            "/link" => {
                let link: Link = key.try_into()?;
                Ok(link.into())
            }
            "/link/host" => {
                let address: Link = key.try_into()?;
                let address: LinkHost = address.try_into()?;
                let address: Link = address.into();
                Ok(address.into())
            }
            "/number/int32" => Ok(Value::Int32(key.try_into()?)),
            "/string" => {
                let s: String = key.try_into()?;
                Ok(s.into())
            }
            _ => Err(error::not_found(path)),
        }
    }
}
