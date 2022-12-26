use futures::future;
use log::{debug, info};

use tc_error::*;
use tc_transact::Transaction;
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::TCPathBuf;

use crate::chain::BlockChain;
use crate::cluster::dir::{Dir, DirCreate, DirCreateItem, DirEntry, ENTRIES};
use crate::cluster::{Class, Cluster, DirItem, Library, Replica, Service};
use crate::route::*;
use crate::scalar::OpRefType;
use crate::state::State;

pub(super) struct DirHandler<'a, T> {
    pub(super) dir: &'a Dir<T>,
    pub(super) path: &'a [PathSegment],
}

impl<'a, T> DirHandler<'a, T> {
    fn new(dir: &'a Dir<T>, path: &'a [PathSegment]) -> Self {
        Self { dir, path }
    }
}

impl<'a, T> DirHandler<'a, T>
where
    T: DirItem,
{
    pub(super) fn method_not_allowed<'b, A: Send + 'b, R: Send + 'a>(
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

impl<'a, T> DirHandler<'a, T>
where
    T: DirItem,
    Dir<T>: DirCreateItem<T> + DirCreate + Replica,
    BlockChain<T>: Replica,
    Cluster<BlockChain<T>>: Public,
{
    pub(super) async fn create_item_or_dir<Item>(
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
route_dir!(Service);

impl<T> Route for DirEntry<T>
where
    T: Send + Sync,
    Cluster<BlockChain<T>>: Route + Send + Sync,
    Cluster<Dir<T>>: Route + Send + Sync,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Dir::route {}", TCPath::from(path));

        match self {
            Self::Dir(dir) => dir.route(path),
            Self::Item(item) => item.route(path),
        }
    }
}
