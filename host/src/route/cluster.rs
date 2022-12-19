use bytes::Bytes;
use futures::{future, TryFutureExt};
use log::{debug, info};
use safecast::{CastInto, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::{Transact, Transaction};
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{TCPathBuf, Tuple};

use crate::chain::BlockChain;
use crate::cluster::dir::{Dir, DirCreate, DirCreateItem, DirEntry, ENTRIES};
use crate::cluster::{class, library, Class, Cluster, DirItem, Legacy, Library, Replica, REPLICAS};
use crate::object::{InstanceClass, Object};
use crate::route::*;
use crate::scalar::{OpRefType, Scalar};
use crate::state::State;
use crate::CLASS;

pub struct ClusterHandler<'a, T> {
    cluster: &'a Cluster<T>,
}

impl<'a, T> Handler<'a> for ClusterHandler<'a, T>
where
    T: Transact + Public + Send + Sync,
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
                if key.is_none() {
                    let participant = value
                        .try_cast_into(|v| TCError::bad_request("invalid participant link", v))?;

                    return self.cluster.mutate(&txn, participant).await;
                }

                let value = if InstanceClass::can_cast_from(&value) {
                    InstanceClass::try_cast_from(value, |v| {
                        TCError::bad_request("invalid class definition", v)
                    })
                    .map(Object::Class)
                    .map(State::Object)?
                } else {
                    value
                };

                self.cluster.state().put(&txn, &[], key, value).await
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

                if txn.is_owner(self.cluster.path()) {
                    return Err(TCError::internal(format!(
                        "{} got commit message for itself",
                        txn.link(self.cluster.link().path().clone())
                    )));
                }

                #[cfg(debug_assertions)]
                info!(
                    "{} got commit message for {}",
                    txn.link(self.cluster.link().path().clone()),
                    txn.id()
                );

                if !txn.has_leader(self.cluster.path()) {
                    // in this case, the kernel did not claim leadership
                    // since a POST request is not necessarily a write
                    // but there's no need to notify the txn owner
                    // because it has already sent a commit message
                    // so just claim leadership on this host and replicate the commit
                    self.cluster.lead_and_distribute_commit(txn.clone()).await?;
                } else if txn.is_leader(self.cluster.path()) {
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

impl<'a, T> DirHandler<'a, T>
where
    T: DirItem,
    Dir<T>: DirCreateItem<T> + DirCreate + Replica,
    DirEntry<T>: Route + Clone + fmt::Display,
    BlockChain<T>: Replica,
    Cluster<BlockChain<T>>: Public,
{
    fn get_entry<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                match self.dir.entry(*txn.id(), &self.path[0]).await? {
                    Some(entry) => entry.get(txn, &self.path[1..], key).await,
                    None => Err(TCError::not_found(&self.path[0])),
                }
            })
        }))
    }

    async fn create_item_or_dir<Item>(
        &self,
        txn: &Txn,
        link: Link,
        name: PathSegment,
        item: Option<Item>,
    ) -> TCResult<()>
    where
        State: From<Item>,
    {
        if let Some(item) = item {
            let cluster = self.dir.create_item(txn, name, link.clone()).await?;

            let replicate_from_this_host = {
                if cluster.link().host().is_none() {
                    true
                } else {
                    let self_link = txn.link(link.path().clone());
                    self_link == link
                }
            };

            if replicate_from_this_host {
                let txn = cluster.lead(txn.clone()).await?;
                let dir_path = TCPathBuf::from(link.path()[..link.path().len() - 1].to_vec());
                debug_assert_eq!(dir_path.len(), link.path().len() - 1);

                let leader = if let Some(host) = link.host() {
                    (host.clone(), dir_path).into()
                } else {
                    dir_path.into()
                };

                txn.put(leader, Value::default(), link.into()).await?;

                cluster
                    .put(&txn, &[], VersionNumber::default().into(), item.into())
                    .await
            } else {
                cluster
                    .add_replica(txn, txn.link(cluster.link().path().clone()))
                    .await
            }
        } else {
            info!("create new cluster directory {}", link);

            let cluster = self.dir.create_dir(txn, name, link).await?;
            debug!("created new cluster directory");

            cluster
                .add_replica(txn, txn.link(cluster.link().path().clone()))
                .await
        }
    }

    fn method_not_allowed<'b, A: Send + 'b, R: Send + 'a>(
        self: Box<Self>,
        method: OpRefType,
    ) -> Box<
        dyn FnOnce(&'b Txn, A) -> Pin<Box<dyn Future<Output = TCResult<R>> + Send + 'a>>
            + Send
            + 'a,
    >
    where
        'b: 'a,
    {
        if self.path.is_empty() {
            Box::new(move |_, _| {
                Box::pin(future::ready(Err(TCError::method_not_allowed(
                    method,
                    self.dir,
                    TCPath::from(self.path),
                ))))
            })
        } else {
            Box::new(move |txn, _: A| {
                Box::pin(async move {
                    if let Some(_version) = self.dir.entry(*txn.id(), &self.path[0]).await? {
                        Err(TCError::internal(format!(
                            "bad routing for {} in {}",
                            TCPath::from(self.path),
                            self.dir
                        )))
                    } else {
                        Err(TCError::not_found(&self.path[0]))
                    }
                })
            })
        }
    }
}

impl<'a> Handler<'a> for DirHandler<'a, Library> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        self.get_entry()
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("{} <- {}: {}", self.dir, key, value);

                let name = key.try_cast_into(|v| {
                    TCError::bad_request("invalid path segment for cluster directory entry", v)
                })?;

                if let Some(_) = self.dir.entry(*txn.id(), &name).await? {
                    return Err(TCError::bad_request(
                        "there is already a directory entry at",
                        name,
                    ))?;
                }

                let (link, lib) = LibraryHandler::lib_version(value)?;

                if link.path().len() <= 1 {
                    return Err(TCError::bad_request(
                        "cannot create a new cluster at",
                        link.path(),
                    ));
                }

                let mut class_path = TCPathBuf::from(CLASS);
                class_path.extend(link.path()[1..].iter().cloned());

                let class_link: Link = if let Some(host) = link.host() {
                    (host.clone(), class_path.clone()).into()
                } else {
                    class_path.clone().into()
                };

                let class_dir_path = TCPathBuf::from(class_path[..class_path.len() - 1].to_vec());

                let parent_dir_path = &link.path()[..link.path().len() - 1];

                if lib.is_empty() {
                    if txn.is_leader(parent_dir_path) {
                        txn.put(
                            class_dir_path.into(),
                            name.clone().into(),
                            class_link.into(),
                        )
                        .await?;
                    }

                    return self
                        .create_item_or_dir::<Map<Scalar>>(txn, link, name, None)
                        .await;
                }

                let (lib, classes) = LibraryHandler::lib_classes(lib)?;

                if !classes.is_empty() && txn.is_leader(parent_dir_path) {
                    txn.put(
                        class_dir_path.into(),
                        name.clone().into(),
                        (class_link, classes).cast_into(),
                    )
                    .await?;
                }

                let version = InstanceClass::anonymous(Some(link.clone()), lib);

                self.create_item_or_dir(txn, link, name, Some(version))
                    .await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(self.method_not_allowed::<Map<State>, State>(OpRefType::Post))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(self.method_not_allowed::<Value, ()>(OpRefType::Delete))
    }
}

impl<'a> Handler<'a> for DirHandler<'a, Class> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        self.get_entry()
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("create new cluster {} at {}", value, key);

                let name = PathSegment::try_cast_from(key, |v| {
                    TCError::bad_request("invalid path segment for class directory entry", v)
                })?;

                let (link, classes): (Link, Option<Map<InstanceClass>>) =
                    if Link::can_cast_from(&value) {
                        let link = value.opt_cast_into().expect("class dir link");
                        (link, None)
                    } else {
                        let (link, classes): (Link, Map<State>) = value.try_cast_into(|s| {
                            TCError::bad_request("expected a tuple (Link, (Class...)) but found", s)
                        })?;

                        let classes = classes
                            .into_iter()
                            .map(|(name, class)| {
                                InstanceClass::try_cast_from(class, |s| {
                                    TCError::bad_request("invalid Class definition", s)
                                })
                                .map(|class| (name, class))
                            })
                            .collect::<TCResult<_>>()?;

                        (link, Some(classes))
                    };

                self.create_item_or_dir(txn, link, name, classes).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(self.method_not_allowed::<Map<State>, State>(OpRefType::Post))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(self.method_not_allowed::<Value, ()>(OpRefType::Delete))
    }
}

struct EntriesHandler<'a, T> {
    dir: &'a Dir<T>,
}

impl<'a, T: Clone + Send + Sync> Handler<'a> for EntriesHandler<'a, T>
where
    Dir<T>: Replica,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| {
            if key.is_none() {
                Box::pin(self.dir.state(*txn.id()))
            } else {
                Box::pin(future::ready(Err(TCError::not_implemented(
                    "cluster entry range query",
                ))))
            }
        }))
    }
}

macro_rules! route_dir {
    ($t:ty) => {
        impl Route for Dir<$t> {
            fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
                if path.len() == 1 && &path[0] == &ENTRIES {
                    Some(Box::new(EntriesHandler { dir: self }))
                } else {
                    Some(Box::new(DirHandler::new(self, path)))
                }
            }
        }
    };
}

route_dir!(Class);
route_dir!(Library);

struct ClassHandler<'a> {
    class: &'a Class,
    path: &'a [PathSegment],
}

impl<'a> ClassHandler<'a> {
    fn new(class: &'a Class, path: &'a [PathSegment]) -> Self {
        Self { class, path }
    }
}

impl<'a> Handler<'a> for ClassHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        assert!(!self.path.is_empty());

        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let number = self.path[0].as_str().parse()?;
                let version = self.class.get_version(*txn.id(), &number).await?;
                version.get(txn, &self.path[1..], key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if self.path.is_empty() {
                    let number =
                        key.try_cast_into(|v| TCError::bad_request("invalid version number", v))?;

                    let version = value.try_into_map(|s| {
                        TCError::bad_request("expected a Map of Classes but found", s)
                    })?;

                    let version = version
                        .into_iter()
                        .map(|(name, class)| {
                            InstanceClass::try_cast_from(class, |s| {
                                TCError::bad_request("expected a Class but found", s)
                            })
                            .map(|class| (name, class))
                        })
                        .collect::<TCResult<Map<InstanceClass>>>()?;

                    self.class.create_version(*txn.id(), number, version).await
                } else {
                    let number = self.path[0].as_str().parse()?;
                    let version = self.class.get_version(*txn.id(), &number).await?;
                    version.put(txn, &self.path[1..], key, value).await
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                if self.path.len() < 2 {
                    return Err(TCError::method_not_allowed(
                        OpRefType::Post,
                        self.class,
                        TCPath::from(self.path),
                    ));
                }

                let number = self.path[0].as_str().parse()?;
                let version = self.class.get_version(*txn.id(), &number).await?;
                let class = version.get_class(*txn.id(), &self.path[1]).await?;
                class.post(txn, &self.path[2..], params).await
            })
        }))
    }
}

impl Route for Class {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(ClassHandler::new(self, path)))
    }
}

struct LibraryHandler<'a> {
    lib: &'a Library,
    path: &'a [PathSegment],
}

impl<'a> LibraryHandler<'a> {
    fn new(lib: &'a Library, path: &'a [PathSegment]) -> Self {
        Self { lib, path }
    }

    fn lib_classes(mut lib: Map<Scalar>) -> TCResult<(Map<Scalar>, Map<InstanceClass>)> {
        let deps = lib
            .iter()
            .filter(|(_, scalar)| scalar.is_ref())
            .map(|(name, _)| name.clone())
            .collect::<Vec<Id>>();

        let classes = deps
            .into_iter()
            .filter_map(|name| lib.remove(&name).map(|dep| (name, dep)))
            .map(|(name, dep)| {
                InstanceClass::try_cast_from(dep, |s| {
                    TCError::bad_request("unable to resolve Library dependency", s)
                })
                .map(|class| (name, class))
            })
            .collect::<TCResult<Map<InstanceClass>>>()?;

        Ok((lib, classes))
    }

    fn lib_version(version: State) -> TCResult<(Link, Map<Scalar>)> {
        let class =
            InstanceClass::try_cast_from(version, |v| TCError::bad_request("invalid Class", v))?;

        let (link, version) = class.into_inner();
        let link =
            link.ok_or_else(|| TCError::bad_request("missing cluster link for", &version))?;

        Ok((link, version))
    }
}

impl<'a> Handler<'a> for LibraryHandler<'a> {
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
                let version = self.lib.get_version(*txn.id(), &number).await?;
                version.get(txn, &self.path[1..], key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            if self.path.is_empty() {
                Box::pin(async move {
                    debug!("{} <- {}: {}", self.lib, key, value);

                    let number = VersionNumber::try_cast_from(key, |v| {
                        TCError::bad_request("invalid version number", v)
                    })?;

                    let (link, version) = Self::lib_version(value)?;
                    let (version, classes) = Self::lib_classes(version)?;

                    if !classes.is_empty() && txn.is_leader(link.path()) {
                        let mut class_path = TCPathBuf::from(CLASS);
                        class_path.extend(link.path()[1..].iter().cloned());

                        txn.put(class_path.into(), number.clone().into(), classes.into())
                            .await?;
                    }

                    self.lib.create_version(*txn.id(), number, version).await
                })
            } else {
                Box::pin(async move {
                    let number = self.path[0].as_str().parse()?;
                    let version = self.lib.get_version(*txn.id(), &number).await?;
                    version.put(txn, &self.path[1..], key, value).await
                })
            }
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                let number = self.path[0].as_str().parse()?;
                let version = self.lib.get_version(*txn.id(), &number).await?;
                version.post(txn, &self.path[1..], params).await
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let number = self.path[0].as_str().parse()?;
                let version = self.lib.get_version(*txn.id(), &number).await?;
                version.delete(txn, &self.path[1..], key).await
            })
        }))
    }
}

impl Route for Library {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(LibraryHandler::new(self, path)))
    }
}

impl<T> Route for DirEntry<T>
where
    T: Send + Sync,
    Cluster<BlockChain<T>>: Route + Send + Sync,
    Cluster<Dir<T>>: Route + Send + Sync,
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

                self.cluster.add_replica(txn, link).await
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

// TODO: consolidate impl Route for Cluster into just one impl
impl Route for Cluster<BlockChain<Class>> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match path {
            path if path.is_empty() => Some(Box::new(ClusterHandler::from(self))),
            path if path == &[REPLICAS] => Some(Box::new(ReplicaHandler::from(self))),
            path => self.state().route(path),
        }
    }
}

// TODO: consolidate impl Route for Cluster into just one impl
impl Route for Cluster<BlockChain<Library>> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match path {
            path if path.is_empty() => Some(Box::new(ClusterHandler::from(self))),
            path if path == &[REPLICAS] => Some(Box::new(ReplicaHandler::from(self))),
            path => self.state().route(path),
        }
    }
}

// TODO: consolidate impl Route for Cluster into just one impl
impl Route for Cluster<Dir<Class>> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match path {
            path if path.is_empty() => Some(Box::new(ClusterHandler::from(self))),
            path if path == &[REPLICAS] => Some(Box::new(ReplicaHandler::from(self))),
            path => self.state().route(path),
        }
    }
}

// TODO: consolidate impl Route for Cluster into just one impl
impl Route for Cluster<Dir<Library>> {
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

struct ClassVersionHandler<'a> {
    class: &'a class::Version,
    path: &'a [PathSegment],
}

impl<'a> ClassVersionHandler<'a> {
    fn new(class: &'a class::Version, path: &'a [PathSegment]) -> Self {
        Self { class, path }
    }
}

impl<'a> Handler<'a> for ClassVersionHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if self.path.is_empty() {
                    let name =
                        key.try_cast_into(|v| TCError::bad_request("invalid class name", v))?;

                    let class = self.class.get_class(*txn.id(), &name).await?;
                    Ok(State::Object(class.clone().into()))
                } else {
                    let class = self.class.get_class(*txn.id(), &self.path[0]).await?;
                    class.get(txn, &self.path[1..], key).await
                }
            })
        }))
    }
}

impl Route for class::Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(ClassVersionHandler::new(self, path)))
    }
}

impl Route for library::Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        assert!(!path.is_empty());

        let attr = self.get_attribute(&path[0])?;
        attr.route(&path[1..])
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
