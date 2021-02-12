//! The transaction context [`Txn`].

use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, TryFutureExt};
use futures_locks::RwLock;
use tokio::sync::mpsc;

use error::*;
use generic::{Id, TCPathBuf};
use transact::fs::Dir;
pub use transact::{Transact, Transaction, TxnId};

use crate::cluster::Cluster;
use crate::fs;
use crate::gateway::Gateway;
use crate::scalar::{Link, Value};
use crate::state::State;

mod request;
mod server;

pub use request::*;
pub use server::*;

#[derive(Clone)]
struct Inner {
    gateway: Arc<Gateway>,
    mutated: RwLock<HashSet<Cluster>>,
    txn_server: mpsc::UnboundedSender<TxnId>,
}

/// A transaction context.
#[derive(Clone)]
pub struct Txn {
    inner: Arc<Inner>,
    request: Arc<Request>,
    dir: fs::Dir,
}

impl Txn {
    fn new(
        txn_server: mpsc::UnboundedSender<TxnId>,
        gateway: Arc<Gateway>,
        dir: fs::Dir,
        request: Request,
    ) -> Self {
        let request = Arc::new(request);
        let mutated = RwLock::new(HashSet::new());

        let inner = Arc::new(Inner {
            gateway,
            mutated,
            txn_server,
        });

        Self {
            inner,
            request,
            dir,
        }
    }

    /// Claim ownership of this transaction.
    pub async fn claim(self, actor: &Actor, cluster_path: TCPathBuf) -> TCResult<Self> {
        if self.owner().is_none() {
            let token = self.request.token.clone();
            let txn_id = self.request.txn_id;

            use rjwt::Resolve;
            let host = Link::from((self.inner.gateway.root().clone(), cluster_path));
            let resolver = Resolver::new(&self.inner.gateway, &host, self.id());
            let (token, claims) = resolver
                .consume_and_sign(actor, vec![SCOPE_ROOT.into()], token, txn_id.time().into())
                .map_err(TCError::unauthorized)
                .await?;

            Ok(Self {
                inner: self.inner.clone(),
                dir: self.dir.clone(),
                request: Arc::new(Request::new(txn_id, token, claims)),
            })
        } else {
            Err(TCError::forbidden(
                "tried to claim owned transaction",
                self.id(),
            ))
        }
    }

    /// Check if the cluster at the specified path on this host is the owner of the transaction.
    pub fn is_owner(&self, cluster_path: TCPathBuf) -> bool {
        if let Some((host, owner_id)) = self.owner() {
            let cluster_link = Link::from((self.inner.gateway.root().clone(), cluster_path));
            host == &cluster_link && owner_id == &Link::from(TCPathBuf::default()).into()
        } else {
            false
        }
    }

    /// Return the owner of this transaction, if there is one.
    pub fn owner(&self) -> Option<(&Link, &Value)> {
        for (host, actor, scopes) in self.request.claims.iter() {
            if scopes.contains(&SCOPE_ROOT.into()) {
                return Some((host, actor));
            }
        }

        None
    }

    /// Return the [`Request`] which initiated this transaction on this host.
    pub fn request(&'_ self) -> &'_ Request {
        &self.request
    }

    /// Return the [`Scope`]s which the given user is authorized for on this transaction.
    pub fn scopes(&'_ self, actor_id: &Value) -> Option<&Vec<Scope>> {
        let host = Link::from(self.inner.gateway.root().clone());
        self.request.claims.get(&host, actor_id)
    }

    /// Register the state of the given [`Cluster`] for synchronization with this transaction.
    pub async fn mutate(&self, cluster: Cluster) -> TCResult<()> {
        let mut mutated = self.inner.mutated.write().await;
        if mutated.contains(&cluster) {
            return Ok(());
        }

        if let Some((link, id)) = self.owner() {
            let cluster_link = Link::from((
                self.inner.gateway.root().clone(),
                TCPathBuf::from(cluster.path().to_vec()),
            ));

            self.put(link.clone(), id.clone(), Value::from(cluster_link).into())
                .await?;
        }

        mutated.insert(cluster);
        Ok(())
    }

    /// Resolve a GET op within this transaction context.
    pub async fn get(&self, link: Link, key: Value) -> TCResult<State> {
        self.inner.gateway.get(self, link, key).await
    }

    /// Resolve a PUT op within this transaction context.
    pub async fn put(&self, link: Link, key: Value, value: State) -> TCResult<()> {
        self.inner.gateway.put(self, link, key, value).await
    }

    /// Resolve a POST op within this transaction context.
    pub async fn post(&self, link: Link, params: State) -> TCResult<State> {
        self.inner.gateway.post(self, link, params).await
    }
}

#[async_trait]
impl Transaction<fs::Dir> for Txn {
    fn id(&'_ self) -> &'_ TxnId {
        &self.request.txn_id
    }

    fn context(&'_ self) -> &'_ fs::Dir {
        &self.dir
    }

    async fn subcontext(&self, id: Id) -> TCResult<Self> {
        let dir = self.dir.create_dir(self.request.txn_id, id).await?;

        Ok(Txn {
            inner: self.inner.clone(),
            request: self.request.clone(),
            dir,
        })
    }
}

#[async_trait]
impl Transact for Txn {
    async fn commit(&self, txn_id: &TxnId) {
        assert_eq!(txn_id, self.id());

        let mutated = self.inner.mutated.read().await;
        join_all(mutated.iter().map(|cluster| cluster.commit(txn_id))).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        assert_eq!(txn_id, self.id());

        let mutated = self.inner.mutated.write().await;
        join_all(mutated.iter().map(|cluster| cluster.finalize(txn_id))).await;
    }
}

impl Hash for Txn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.request.txn_id.hash(state)
    }
}
