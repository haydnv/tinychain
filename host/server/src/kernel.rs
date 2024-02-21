use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use async_trait::async_trait;
use log::{debug, warn};
use umask::Mode;

use tc_error::*;
use tc_scalar::OpRefType;
use tc_state::CacheBlock;
use tc_transact::public::*;
use tc_transact::{fs, Transact, Transaction, TxnId};
use tc_value::{Host, Link, Value};
use tcgeneric::{label, Label, Map, NetworkTime, PathSegment, TCPath, TCPathBuf};

use crate::cluster::{Class, Cluster, Dir, DirEntry, ReplicateAndJoin, Schema};
use crate::txn::{Hypothetical, Txn, TxnServer};
use crate::{aes256, Authorize, SignedToken, State};

const CLASS: Label = label("class");

pub type ReplicaJoinResult = Result<(), (bool, TCError)>;

pub struct Endpoint<'a> {
    mode: Mode,
    txn: &'a Txn,
    path: &'a [PathSegment],
    handler: Box<dyn Handler<'a, State> + 'a>,
}

impl<'a> Endpoint<'a> {
    pub fn umask(&self) -> Mode {
        self.mode
    }

    pub fn get(self, key: Value) -> TCResult<GetFuture<'a, State>> {
        let get = self
            .handler
            .get()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Get, TCPath::from(self.path)))?;

        if self.mode.may_read() {
            Ok((get)(self.txn, key))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }

    pub fn put(self, key: Value, value: State) -> TCResult<PutFuture<'a>> {
        let put = self
            .handler
            .put()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Put, TCPath::from(self.path)))?;

        if self.mode.may_write() {
            Ok((put)(self.txn, key, value))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }

    pub fn post(self, params: Map<State>) -> TCResult<PostFuture<'a, State>> {
        let post = self
            .handler
            .post()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Post, TCPath::from(self.path)))?;

        if self.mode.may_execute() {
            Ok((post)(self.txn, params))
        } else {
            Err(unauthorized!("execute {}", TCPath::from(self.path)))
        }
    }

    pub fn delete(self, key: Value) -> TCResult<DeleteFuture<'a>> {
        let delete = self.handler.delete().ok_or_else(|| {
            TCError::method_not_allowed(OpRefType::Delete, TCPath::from(self.path))
        })?;

        if self.mode.may_write() {
            Ok((delete)(self.txn, key))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }
}

pub(crate) struct Kernel {
    class: Cluster<Dir<Class>>,
    hypothetical: Cluster<Hypothetical>,
    keys: HashSet<aes256::Key>,
}

impl Kernel {
    pub fn authorize_claim_and_route<'a>(
        &'a self,
        path: &'a [PathSegment],
        txn: &'a Txn,
    ) -> TCResult<Endpoint<'a>> {
        if path.is_empty() {
            Err(unauthorized!("access to /"))
        } else if path[0] == CLASS {
            auth_claim_route(&self.class, &path[1..], txn)
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            auth_claim_route(&self.hypothetical, &path[2..], txn)
        } else {
            Err(not_found!("{}", TCPath::from(path)))
        }
    }

    pub async fn replicate_and_join(
        &self,
        txn_server: TxnServer,
        peers: BTreeSet<Host>,
    ) -> Result<(), bool> {
        let mut progress = false;
        let mut tokens = HashMap::<TCPathBuf, SignedToken>::new();
        let mut unvisited = VecDeque::new();
        unvisited.push_back(Box::new(self.class.clone()));

        while let Some(cluster) = unvisited.pop_front() {
            if !tokens.contains_key(cluster.path()) {
                let txn = txn_server.new_txn(NetworkTime::now(), None);

                // TODO: fetch token
            }

            let mut joined = false;
            let token = tokens.get(cluster.path()).cloned();
            let txn = txn_server.new_txn(NetworkTime::now(), token);
            for peer in &peers {
                match cluster.replicate_and_join(&txn, peer.clone()).await {
                    Ok(entries) => {
                        joined = true;
                        progress = true;

                        for (_name, entry) in entries {
                            match &*entry {
                                DirEntry::Dir(dir) => unvisited.push_back(Box::new(dir.clone())),
                                DirEntry::Item(_) => {}
                            }
                        }
                    }
                    Err(cause) => warn!("failed to replicate from {peer}: {cause}"),
                }
            }

            if !joined {
                return Err(progress);
            }
        }

        // TODO: replicate services in the /services dir

        Ok(())
    }
}

impl Kernel {
    pub async fn commit(&self, txn_id: TxnId) {
        self.hypothetical.rollback(&txn_id).await;
    }

    pub async fn finalize(&self, txn_id: TxnId) {
        self.hypothetical.finalize(&txn_id).await;
    }
}

#[async_trait]
impl fs::Persist<CacheBlock> for Kernel {
    type Schema = (Option<Link>, Option<Link>, HashSet<aes256::Key>);
    type Txn = Txn;

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        let (owner, group, keys) = schema;

        let class_dir: fs::Dir<CacheBlock> = store.create_dir(txn_id, CLASS.into()).await?;
        let schema = Schema::new(CLASS, owner.clone(), group.clone());
        let class = fs::Persist::<CacheBlock>::create(txn_id, schema, class_dir).await?;

        let schema = Schema::new(Hypothetical::PATH, owner, group);
        let hypothetical = Cluster::new(schema, Hypothetical::new(), txn_id);

        Ok(Self {
            class,
            hypothetical,
            keys,
        })
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        let (owner, group, keys) = schema;

        let class_dir: fs::Dir<CacheBlock> = store.create_dir(txn_id, CLASS.into()).await?;
        let schema = Schema::new(CLASS, owner.clone(), group.clone());
        let class = fs::Persist::<CacheBlock>::load(txn_id, schema, class_dir).await?;

        let schema = Schema::new(Hypothetical::PATH, owner, group);
        let hypothetical = Cluster::new(schema, Hypothetical::new(), txn_id);

        Ok(Self {
            class,
            hypothetical,
            keys,
        })
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        unimplemented!("Kernel::inner")
    }
}

fn auth_claim_route<'a, T>(
    cluster: &'a Cluster<T>,
    path: &'a [PathSegment],
    txn: &'a Txn,
) -> TCResult<Endpoint<'a>>
where
    Cluster<T>: Route<State>,
{
    let txn_id = *txn.id();
    let keyring = cluster.keyring(txn_id)?;
    let resource_mode = cluster.umask(txn_id, path);
    let request_mode = txn.mode(keyring, path);
    let mode = resource_mode & request_mode;

    if mode == Mode::new() {
        return Err(unauthorized!("access to {}", TCPath::from(path)));
    } else {
        debug!("request permissions are {mode}");
    }

    let handler = cluster
        .route(&*path)
        .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

    let endpoint = Endpoint {
        mode,
        txn,
        path,
        handler,
    };

    Ok(endpoint)
}
