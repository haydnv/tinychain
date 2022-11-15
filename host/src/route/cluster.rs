use bytes::Bytes;
use futures::{future, TryFutureExt};
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::{Transact, Transaction};
use tc_value::{Link, Value};
use tcgeneric::Tuple;

use crate::chain::BlockChain;
use crate::cluster::dir::{Dir, DirEntry};
use crate::cluster::library::{Library, Version};
use crate::cluster::{Cluster, Legacy, Replica, REPLICAS};
use crate::fs;
use crate::route::*;
use crate::state::State;

pub struct ClusterHandler<'a, T> {
    cluster: &'a Cluster<T>,
}

impl<'a, T> Handler<'a> for ClusterHandler<'a, T>
where
    T: Transact + Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |_txn, key| {
            Box::pin(future::ready((|key: Value| {
                key.expect_none()?;
                let public_key = Bytes::from(self.cluster.public_key().to_vec());
                Ok(Value::from(public_key).into())
            })(key)))
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                key.expect_none()?;

                let participant =
                    value.try_cast_into(|s| TCError::bad_request("expected a Link, not", s))?;

                self.cluster.mutate(&txn, participant).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                // TODO: authorize request using a scope

                if !params.is_empty() {
                    return Err(TCError::bad_request(
                        "unrecognized commit parameters",
                        params,
                    ));
                }

                if txn.is_leader(self.cluster.path()) {
                    self.cluster.distribute_commit(txn).await?;
                } else {
                    self.cluster.commit(txn.id()).await;
                }

                Ok(State::default())
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                if txn.is_leader(self.cluster.path()) {
                    self.cluster.distribute_rollback(txn).await;
                } else {
                    self.cluster.finalize(txn.id()).await;
                }

                Ok(())
            })
        }))
    }
}

impl<'a, T> From<&'a Cluster<T>> for ClusterHandler<'a, T> {
    fn from(cluster: &'a Cluster<T>) -> Self {
        Self { cluster }
    }
}

struct DirHandler<'a, T> {
    dir: &'a Dir<T>,
    path: &'a [PathSegment],
}

impl<'a, T> DirHandler<'a, T> {
    fn new(dir: &'a Dir<T>, path: &'a [PathSegment]) -> Self {
        Self { dir, path }
    }
}

impl<'a, T> Handler<'a> for DirHandler<'a, T>
where
    T: Persist<fs::Dir, Txn = Txn>,
    Dir<T>: Persist<fs::Dir, Txn = Txn, Store = fs::Dir> + Send + Sync,
    DirEntry<T>: Clone,
    Cluster<T>: Route,
    Cluster<BlockChain<Dir<T>>>: Route,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                match self.dir.entry(*txn.id(), &self.path[0]).await? {
                    Some(entry) => match entry {
                        DirEntry::Dir(dir) => dir.get(txn, &self.path[1..], key).await,
                        DirEntry::Item(item) => item.get(txn, &self.path[1..], key).await,
                    },
                    None => Err(TCError::not_found(&self.path[0])),
                }
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, _key, _value| {
            Box::pin(async move { Err(TCError::not_implemented("Dir::PUT")) })
        }))
    }
}

impl<T> Route for Dir<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Route + Clone,
    Dir<T>: Persist<fs::Dir, Txn = Txn, Store = fs::Dir>,
    DirEntry<T>: Clone,
    Cluster<T>: Route,
    Cluster<BlockChain<Dir<T>>>: Route,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(DirHandler::new(self, path)))
    }
}

struct LibHandler<'a> {
    lib: &'a Library,
    path: &'a [PathSegment],
}

impl<'a> LibHandler<'a> {
    fn new(lib: &'a Library, path: &'a [PathSegment]) -> Self {
        Self { lib, path }
    }
}

impl<'a> Handler<'a> for LibHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        assert!(!self.path.is_empty());

        Some(Box::new(|txn, key| {
            Box::pin(async move {
                debug!(
                    "route GET {} to version {}",
                    TCPath::from(&self.path[1..]),
                    &self.path[0]
                );

                let number = self.path[0].as_str().parse()?;
                let version = self.lib.get_version(*txn.id(), number).await?;
                version.get(txn, &self.path[1..], key).await
            })
        }))
    }
}

impl Route for Library {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(LibHandler::new(self, path)))
    }
}

impl<T> Route for DirEntry<T>
where
    Cluster<T>: Route + Send + Sync,
    Cluster<BlockChain<Dir<T>>>: Route + Send + Sync,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::Dir(dir) => dir.route(path),
            Self::Item(item) => item.route(path),
        }
    }
}

struct ReplicaHandler<'a, T> {
    cluster: &'a Cluster<T>,
}

impl<'a, T> Handler<'a> for ReplicaHandler<'a, T>
where
    T: Replica + Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                self.cluster
                    .replicas(*txn.id())
                    .map_ok(|replicas| Value::Tuple(replicas.iter().cloned().collect()))
                    .map_ok(State::from)
                    .await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, link| {
            Box::pin(async move {
                key.expect_none()?;

                let link = link.try_cast_into(|v| {
                    TCError::bad_request("expected a Link to a Cluster, not", v)
                })?;

                self.cluster.add_replica(&txn, link).await
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, replicas| {
            Box::pin(async move {
                let replicas = Tuple::<Link>::try_cast_from(replicas, |v| {
                    TCError::bad_request("expected a Link to a Cluster, not", v)
                })?;

                self.cluster.remove_replicas(txn, &replicas).await
            })
        }))
    }
}

impl<'a, T> From<&'a Cluster<T>> for ReplicaHandler<'a, T> {
    fn from(cluster: &'a Cluster<T>) -> Self {
        Self { cluster }
    }
}

impl Route for Cluster<BlockChain<Library>> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match path {
            path if path.is_empty() => Some(Box::new(ClusterHandler::from(self))),
            path if path == &[REPLICAS] => Some(Box::new(ReplicaHandler::from(self))),
            path => self.state().route(path),
        }
    }
}

impl Route for Cluster<BlockChain<Dir<BlockChain<Library>>>> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match path {
            path if path.is_empty() => Some(Box::new(ClusterHandler::from(self))),
            path if path == &[REPLICAS] => Some(Box::new(ReplicaHandler::from(self))),
            path => self.state().route(path),
        }
    }
}

// TODO: delete
impl Route for Cluster<Legacy> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match path {
            path if path.is_empty() => Some(Box::new(ClusterHandler::from(self))),
            path if path == &[REPLICAS] => Some(Box::new(ReplicaHandler::from(self))),
            path => self.state().route(path),
        }
    }
}

struct VersionHandler<'a> {
    version: &'a Version,
    path: &'a [PathSegment],
}

impl<'a> VersionHandler<'a> {
    fn new(version: &'a Version, path: &'a [PathSegment]) -> Self {
        Self { version, path }
    }
}

impl<'a> Handler<'a> for VersionHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if self.path.is_empty() {
            todo!("library replication")
        }

        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let attr = self
                    .version
                    .attribute(&self.path[0])
                    .ok_or_else(|| TCError::not_found(&self.path[0]))?;

                attr.get(txn, &self.path[1..], key).await
            })
        }))
    }
}

impl Route for Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(VersionHandler::new(self, path)))
    }
}

impl Route for Legacy {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if let Some(chain) = self.chain(&path[0]) {
            debug!("Legacy cluster has a Chain at {}", &path[0]);
            chain.route(&path[1..])
        } else if let Some(class) = self.class(&path[0]) {
            debug!("Legacy cluster has a Class at {}", &path[0]);
            class.route(&path[1..])
        } else {
            None
        }
    }
}
