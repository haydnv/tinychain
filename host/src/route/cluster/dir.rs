use futures::future;
use log::debug;

use tc_error::*;
use tc_transact::Transaction;
use tc_value::{Link, Version as VersionNumber};

use crate::chain::BlockChain;
use crate::cluster::dir::{Dir, DirCreate, DirCreateItem, DirEntry, ENTRIES};
use crate::cluster::{Class, Cluster, DirItem, Library, Replica, Service};
use crate::object::InstanceClass;
use crate::route::*;
use crate::scalar::{OpRef, Scalar, TCRef};
use crate::state::State;

pub(super) struct DirHandler<'a, T> {
    pub(super) dir: &'a Dir<T>,
}

impl<'a, T> DirHandler<'a, T> {
    fn new(dir: &'a Dir<T>) -> Self {
        Self { dir }
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
        let replicate_from_this_host = match link.host() {
            None => true,
            Some(host) => host == txn.host(),
        };

        if let Some(item) = item {
            let cluster = self.dir.create_item(txn, name, link).await?;

            if replicate_from_this_host {
                let txn = cluster.lead(txn.clone()).await?;

                cluster
                    .put(&txn, &[], VersionNumber::default().into(), item.into())
                    .await?;
            } else {
                cluster.add_replica(txn, txn.host().clone()).await?;
            }
        } else {
            let cluster = self.dir.create_dir(txn, name, link).await?;

            if replicate_from_this_host {
                return Ok(());
            } else {
                cluster.add_replica(txn, txn.host().clone()).await?;
            }
        }

        Ok(())
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
                Box::pin(future::ready(Err(not_implemented!(
                    "cluster entry range query"
                ))))
            }
        }))
    }
}

macro_rules! route_dir {
    ($t:ty) => {
        impl Route for Dir<$t> {
            fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
                if path.is_empty() {
                    Some(Box::new(DirHandler::new(self)))
                } else if path.len() == 1 && &path[0] == &ENTRIES {
                    Some(Box::new(EntriesHandler { dir: self }))
                } else {
                    None
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
        debug!("DirEntry::route {}", TCPath::from(path));

        match self {
            Self::Dir(dir) => dir.route(path),
            Self::Item(item) => item.route(path),
        }
    }
}

pub(super) fn expect_version(version: State) -> TCResult<(Link, Map<Scalar>)> {
    InstanceClass::try_cast_from(version, |v| TCError::bad_request("invalid Class", v))
        .map(|class| class.into_inner())
}

pub(super) fn extract_classes(mut lib: Map<Scalar>) -> TCResult<(Map<Scalar>, Map<InstanceClass>)> {
    let deps = lib
        .iter()
        .filter_map(|(name, scalar)| {
            if let Scalar::Ref(tc_ref) = scalar {
                if let TCRef::Op(OpRef::Post(_)) = &**tc_ref {
                    return Some(name);
                }
            }

            None
        })
        .cloned()
        .collect::<Vec<Id>>();

    let classes = deps
        .into_iter()
        .filter_map(|name| lib.remove(&name).map(|dep| (name, dep)))
        .map(|(name, dep)| {
            InstanceClass::try_cast_from(dep, |s| {
                TCError::bad_request("unable to resolve dependency", s)
            })
            .map(|class| (name, class))
        })
        .collect::<TCResult<Map<InstanceClass>>>()?;

    Ok((lib, classes))
}
