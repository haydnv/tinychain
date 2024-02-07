//! The transaction context [`Txn`].

use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::DirLock;
use futures::Future;
use log::{debug, trace};
use safecast::CastInto;

use tc_error::*;
use tc_transact::Transaction;
use tc_value::uuid::Uuid;
use tc_value::{Host, Link, ToUrl, Value};
use tcgeneric::{Id, PathSegment, TCBoxFuture, TCBoxTryFuture, TCPathBuf, ThreadSafe};

use crate::block::CacheBlock;
use crate::state::State;

pub use request::*;
pub use server::*;
pub use tc_transact::TxnId;

pub mod hypothetical;
mod request;
mod server;

/// A transactional directory
pub type Dir = tc_transact::fs::Dir<CacheBlock>;

/// An entry in a transactional directory
pub type DirEntry<B> = tc_transact::fs::DirEntry<CacheBlock, B>;

/// A transactional file
pub type File<B> = tc_transact::fs::File<CacheBlock, B>;

pub trait Gateway: ThreadSafe {
    /// Return this [`Host`]
    fn host(&self) -> &Host;

    /// Return a link to the given path on this host.
    fn link(&self, path: TCPathBuf) -> Link;

    /// Read a simple value.
    fn fetch<'a>(
        &'a self,
        txn_id: &'a TxnId,
        link: ToUrl<'a>,
        key: &'a Value,
    ) -> TCBoxTryFuture<Value>;

    /// Read the [`State`] at `link` with the given `key`.
    fn get<'a>(&'a self, txn: &'a Txn, link: ToUrl<'a>, key: Value) -> TCBoxTryFuture<'a, State>;

    /// Update the [`State`] with the given `key` at `link` to `value`.
    fn put<'a>(
        &'a self,
        txn: &'a Txn,
        link: ToUrl<'a>,
        key: Value,
        value: State,
    ) -> TCBoxTryFuture<'a, ()>;

    /// Execute the POST op at `link` with the `params`
    fn post<'a>(
        &'a self,
        txn: &'a Txn,
        link: ToUrl<'a>,
        params: State,
    ) -> TCBoxTryFuture<'a, State>;

    /// Delete the [`State`] at `link` with the given `key`.
    fn delete<'a>(&'a self, txn: &'a Txn, link: ToUrl<'a>, key: Value) -> TCBoxTryFuture<'a, ()>;

    fn finalize(&self, txn_id: TxnId) -> TCBoxFuture<()>;
}

#[derive(Clone)]
enum LazyDir {
    Workspace(DirLock<CacheBlock>),
    Lazy(Arc<Self>, Id),
}

impl LazyDir {
    fn get_or_create<'a>(
        &'a self,
        txn_id: &'a TxnId,
    ) -> Pin<Box<dyn Future<Output = TCResult<DirLock<CacheBlock>>> + Send + 'a>> {
        Box::pin(async move {
            match self {
                Self::Workspace(workspace) => {
                    let mut parent = workspace.write().await;
                    parent
                        .get_or_create_dir(txn_id.to_string())
                        .map_err(TCError::from)
                }
                Self::Lazy(parent, name) => {
                    let parent = parent.get_or_create(txn_id).await?;
                    let mut parent = parent.write().await;

                    parent
                        .get_or_create_dir(name.to_string())
                        .map_err(TCError::from)
                }
            }
        })
    }

    fn create_dir(self, name: Id) -> Self {
        Self::Lazy(Arc::new(self), name)
    }

    fn create_dir_unique(self) -> Self {
        Self::Lazy(Arc::new(self), Uuid::new_v4().into())
    }
}

/// A transaction context.
#[derive(Clone)]
pub struct Txn {
    gateway: Arc<dyn Gateway>,
    request: Arc<Request>,
    scope: TCPathBuf,
    dir: LazyDir,
}

impl Txn {
    fn new(workspace: DirLock<CacheBlock>, gateway: Arc<dyn Gateway>, request: Request) -> Self {
        let scope = request.txn_id().to_id().into();
        let request = Arc::new(request);

        Self {
            gateway,
            request,
            scope,
            dir: LazyDir::Workspace(workspace),
        }
    }
}

impl Txn {
    /// Return this [`Host`]
    pub fn host(&self) -> &Host {
        self.gateway.host()
    }

    /// Return a link to the given path on this host.
    pub fn link(&self, path: TCPathBuf) -> Link {
        self.gateway.link(path)
    }

    /// Return the [`Request`] which initiated this transaction on this host.
    pub fn request(&self) -> &Request {
        &self.request
    }

    /// Return a new `Txn` which grants the given [`Scope`]s to the given [`Actor`].
    pub fn grant(
        &self,
        actor: &Actor,
        cluster_path: TCPathBuf,
        scopes: Vec<Scope>,
    ) -> TCResult<Self> {
        trace!("grant {scopes:?} for {cluster_path} to {}", actor.id());

        let host_id = self.gateway.link(cluster_path);
        let txn_id = self.request.txn_id();
        let now = txn_id.time().into();
        let token = actor.consume_and_sign(self.request.token().clone(), host_id, scopes, now)?;

        trace!("granted scopes to {}", actor.id());

        Ok(Self {
            gateway: self.gateway.clone(),
            request: Arc::new(Request::new(*txn_id, token)),
            scope: self.scope.clone(),
            dir: self.dir.clone(),
        })
    }

    /// Claim ownership of this transaction.
    pub fn claim(self, actor: &Actor, cluster_path: TCPathBuf) -> TCResult<Self> {
        debug!(
            "{} claims ownership of transaction {}",
            cluster_path,
            self.request.txn_id()
        );

        if actor.id().is_some() {
            return Err(bad_request!("cluster ID must be None, not {}", actor.id()));
        }

        if self.owner().is_none() {
            self.grant(actor, cluster_path, vec![self.scope.clone()])
        } else {
            Err(forbidden!(
                "tried to claim owned transaction {}",
                self.request.txn_id()
            ))
        }
    }

    /// Check if this transaction has an owner.
    pub fn has_owner(&self) -> bool {
        self.owner().is_some()
    }

    /// Check if the cluster at the specified path on this host is the owner of the transaction.
    pub fn is_owner(&self, cluster_path: &[PathSegment]) -> bool {
        if let Some(owner) = self.owner() {
            if owner.host() == Some(self.gateway.host()) {
                return cluster_path == owner.path().deref();
            }
        }

        false
    }

    /// Return the owner of this transaction, if there is one.
    pub fn owner(&self) -> Option<&Link> {
        let (host, _, _) = self
            .request
            .token()
            .first_claim(|(_host, actor_id, scopes)| {
                actor_id == &Value::None && scopes.contains(&self.scope)
            })?;

        Some(host)
    }

    /// Check if this transaction has a leader for the given cluster.
    pub fn has_leader(&self, cluster_path: &[PathSegment]) -> bool {
        self.leader(cluster_path).is_some()
    }

    /// Check if this host is leading the transaction for the specified cluster.
    pub fn is_leader(&self, cluster_path: &[PathSegment]) -> bool {
        if let Some(leader) = self.leader(cluster_path) {
            if leader.host() == Some(self.gateway.host()) {
                return cluster_path == leader.path().deref();
            }
        }

        false
    }

    /// Claim leadership of this transaction for the given cluster.
    pub fn lead(self, actor: &Actor, cluster_path: TCPathBuf) -> TCResult<Self> {
        debug!(
            "{} claim leadership of transaction {}",
            cluster_path,
            self.request.txn_id()
        );

        if actor.id().is_some() {
            return Err(bad_request!("cluster ID must be None, not {}", actor.id()));
        }

        if let Some(leader) = self.leader(&cluster_path) {
            Err(internal!(
                "{} tried to claim leadership of {} but {} is already the leader",
                cluster_path,
                self.request.txn_id(),
                leader
            ))
        } else {
            let scopes = vec![self.scope.clone()];
            self.grant(actor, cluster_path, scopes)
        }
    }

    /// Return the leader of this transaction for the given cluster, if there is one.
    pub fn leader(&self, cluster_path: &[PathSegment]) -> Option<&Link> {
        let (host, _, _) = self.request.token().last_claim(|(host, actor, scopes)| {
            actor == &Value::None
                && scopes.contains(&self.scope)
                && cluster_path.starts_with(host.path())
        })?;

        Some(host)
    }
}

#[async_trait]
impl Transaction<CacheBlock> for Txn {
    #[inline]
    fn id(&self) -> &TxnId {
        self.request.txn_id()
    }

    async fn context(&self) -> TCResult<DirLock<CacheBlock>> {
        self.dir.get_or_create(self.request.txn_id()).await
    }

    fn subcontext<I: Into<Id> + Send>(&self, id: I) -> Self {
        Txn {
            gateway: self.gateway.clone(),
            request: self.request.clone(),
            scope: self.scope.clone(),
            dir: self.dir.clone().create_dir(id.into()),
        }
    }

    fn subcontext_unique(&self) -> Self {
        Txn {
            gateway: self.gateway.clone(),
            request: self.request.clone(),
            scope: self.scope.clone(),
            dir: self.dir.clone().create_dir_unique(),
        }
    }
}

#[async_trait]
impl tc_transact::RPCClient<State> for Txn {
    async fn get<'a, L, V>(&'a self, link: L, key: V) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send,
    {
        self.gateway.get(self, link.into(), key.cast_into()).await
    }

    async fn put<'a, L, K, V>(&'a self, link: L, key: K, value: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        K: CastInto<Value> + Send,
        V: CastInto<State> + Send,
    {
        self.gateway
            .put(self, link.into(), key.cast_into(), value.cast_into())
            .await
    }

    async fn post<'a, L, P>(&'a self, link: L, params: P) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        P: CastInto<State> + Send,
    {
        self.gateway
            .post(self, link.into(), params.cast_into())
            .await
    }

    async fn delete<'a, L, V>(&'a self, link: L, key: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send,
    {
        self.gateway
            .delete(self, link.into(), key.cast_into())
            .await
    }
}

impl Hash for Txn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.request.txn_id().hash(state)
    }
}
