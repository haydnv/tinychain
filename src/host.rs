use std::collections::HashMap;
use std::convert::TryInto;
use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time;

use bytes::Bytes;
use sha2::{Digest, Sha256};
use structopt::StructOpt;

use crate::error;
use crate::http;
use crate::internal::block::Store;
use crate::internal::file::File;
use crate::object::actor::{Actor, Token};
use crate::object::TCObject;
use crate::state::table::{Row, Table};
use crate::state::{Cluster, Collection, Directory, Persistent, State};
use crate::transaction::Txn;
use crate::value::link::{Link, LinkHost, PathSegment, TCPath};
use crate::value::{TCResult, TCValue, ValueId};

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
    data_dir: Arc<Store>,
    workspace: Arc<Store>,
    root: Hosted,
}

impl Host {
    pub async fn new(config: HostConfig) -> TCResult<Arc<Host>> {
        // TODO: figure out a way to populate `root` without locking

        let data_dir = Store::new(config.data_dir);
        let workspace = Store::new_tmp(config.workspace);

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

            let dir = if let Some(store) = host.data_dir.get_store(txn_id, &path).await {
                Directory::from_store(txn_id, store).await
            } else {
                Directory::new(txn_id, host.data_dir.reserve(txn_id, path.clone()).await?).await?
            };

            host.root.push(path, dir);
        }

        host.data_dir.commit(&txn.id()).await;

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
            let dir = &path[1];
            let path = &path.slice_from(2);
            match dir.as_str() {
                "auth" => Ok(Sbin::auth(path, key)?.into()),
                "object" => Ok(Sbin::get_object(path, key)?.into()),
                "state" => Sbin::state(txn, path, key).await,
                "value" => Ok(Sbin::value(path, key)?.into()),
                _ => Err(error::not_found(path)),
            }
        } else if let Some((path, dir)) = self.root.get(path) {
            let state = dir.get(txn, &path, auth).await?;
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
        } else if path[0] == "sbin" {
            let dir = &path[1];
            let path = &path.slice_from(2);
            match dir.as_str() {
                "auth" => Err(error::method_not_allowed(path)),
                "object" => Ok(Sbin::put_object(self, path, key, state)?.into()),
                "state" => Err(error::method_not_allowed(path)),
                "value" => Err(error::method_not_allowed(path)),
                _ => Err(error::not_found(path)),
            }
        } else if let Some((mut path, dir)) = self.root.get(path) {
            let key: TCPath = key.try_into()?;
            path.extend(key.into_iter());
            Ok(dir.clone().put(txn, path, state, auth).await?.into())
        } else {
            Err(error::not_found(path))
        }
    }
}

struct Sbin;

impl Sbin {
    fn auth(path: &TCPath, key: TCValue) -> TCResult<TCValue> {
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

    fn get_object(path: &TCPath, key: TCValue) -> TCResult<TCValue> {
        match path.to_string().as_str() {
            "/actor" => {
                let row: Row = key.try_into()?;
                let actor: Actor = row.try_into()?;
                let row: Row = actor.into();
                Ok(TCValue::from(row))
            }
            _ => Err(error::not_found(path)),
        }
    }

    fn put_object(host: &Arc<Host>, path: &TCPath, id: TCValue, state: State) -> TCResult<TCValue> {
        match path.to_string().as_str() {
            "/actor" => {
                let state: TCValue = state.try_into()?;
                let actor =
                    Actor::new((host.address, host.http_port).into(), id, state.try_into()?);
                Ok(TCValue::from(actor.as_row()))
            }
            _ => Err(error::not_found(path)),
        }
    }

    async fn state(txn: &Arc<Txn<'_>>, path: &TCPath, key: TCValue) -> TCResult<State> {
        match path.to_string().as_str() {
            "/cluster" => Ok(Cluster::create(txn, key.try_into()?).await?.into()),
            "/table" => Ok(Table::create(txn, key.try_into()?).await?.into()),
            _ => Err(error::not_found(path)),
        }
    }

    fn value(path: &TCPath, key: TCValue) -> TCResult<TCValue> {
        match path.to_string().as_str() {
            "/" => Ok(key),
            "/bytes" => {
                let encoded: String = key.try_into()?;
                let decoded = base64::decode(encoded)
                    .map_err(|e| error::bad_request("Unable to decode base64 string", e))?;
                Ok(TCValue::Bytes(Bytes::from(decoded)))
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
            "/string" => {
                let s: String = key.try_into()?;
                Ok(s.into())
            }
            _ => Err(error::not_found(path)),
        }
    }
}

#[derive(Clone)]
struct HostedNode {
    children: HashMap<PathSegment, HostedNode>,
}

#[derive(Clone)]
struct Hosted {
    root: HostedNode,
    hosted: HashMap<TCPath, Arc<Directory>>,
}

impl Hosted {
    fn new() -> Hosted {
        Hosted {
            root: HostedNode {
                children: HashMap::new(),
            },
            hosted: HashMap::new(),
        }
    }

    fn get(&self, path: &TCPath) -> Option<(TCPath, Arc<Directory>)> {
        println!("checking for hosted directory {}", path);
        let mut node = &self.root;
        let mut found_path = TCPath::default();
        for segment in path.clone() {
            if let Some(child) = node.children.get(&segment) {
                found_path.push(segment);
                node = child;
                println!("found {}", found_path);
            } else if found_path != TCPath::default() {
                return Some((
                    path.from_path(&found_path).unwrap(),
                    self.hosted.get(&found_path).unwrap().clone(),
                ));
            } else {
                println!("couldn't find {}", segment);
                return None;
            }
        }

        if let Some(dir) = self.hosted.get(&found_path) {
            Some((path.from_path(&found_path).unwrap(), dir.clone()))
        } else {
            None
        }
    }

    fn push(&mut self, path: TCPath, dir: Arc<Directory>) -> Option<Arc<Directory>> {
        let mut node = &mut self.root;
        for segment in path.clone() {
            node = node.children.entry(segment).or_insert(HostedNode {
                children: HashMap::new(),
            });
        }

        println!("Hosted directory: {}", path);
        self.hosted.insert(path, dir)
    }
}
