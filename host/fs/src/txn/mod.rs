//! The transaction context [`Txn`].

use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::DirLock;
use futures::future::TryFutureExt;
use log::debug;
use safecast::CastInto;

use tc_error::*;
use tc_transact::public::StateInstance;
use tc_transact::Transaction;
use tc_value::{Host, Link, ToUrl, Value};
use tcgeneric::{
    Id, NetworkTime, PathSegment, TCBoxFuture, TCBoxTryFuture, TCPathBuf, ThreadSafe, Tuple,
};

use crate::block::CacheBlock;

pub use request::*;
pub use server::*;
pub use tc_transact::TxnId;

pub mod hypothetical;
mod request;
mod server;

pub trait Gateway: ThreadSafe {
    type State: StateInstance<FE = CacheBlock>;

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
    fn get<'a>(
        &'a self,
        txn: &'a Txn<Self::State>,
        link: ToUrl<'a>,
        key: Value,
    ) -> TCBoxTryFuture<'a, Self::State>;

    /// Update the [`State`] with the given `key` at `link` to `value`.
    fn put<'a>(
        &'a self,
        txn: &'a Txn<Self::State>,
        link: ToUrl<'a>,
        key: Value,
        value: Self::State,
    ) -> TCBoxTryFuture<'a, ()>;

    /// Execute the POST op at `link` with the `params`
    fn post<'a>(
        &'a self,
        txn: &'a Txn<Self::State>,
        link: ToUrl<'a>,
        params: Self::State,
    ) -> TCBoxTryFuture<'a, Self::State>;

    /// Delete the [`State`] at `link` with the given `key`.
    fn delete<'a>(
        &'a self,
        txn: &'a Txn<Self::State>,
        link: ToUrl<'a>,
        key: Value,
    ) -> TCBoxTryFuture<'a, ()>;

    fn finalize(&self, txn_id: TxnId) -> TCBoxFuture<()>;
}

struct Active {
    workspace: DirLock<CacheBlock>,
    expires: NetworkTime,
    scope: Scope,
}

impl Active {
    fn new(txn_id: &TxnId, workspace: DirLock<CacheBlock>, expires: NetworkTime) -> Self {
        let scope = TCPathBuf::from(txn_id.to_id());

        Self {
            workspace,
            expires,
            scope,
        }
    }

    fn expires(&self) -> &NetworkTime {
        &self.expires
    }

    fn scope(&self) -> &Scope {
        &self.scope
    }
}

/// A transaction context.
#[derive(Clone)]
pub struct Txn<State> {
    active: Arc<Active>,
    gateway: Arc<dyn Gateway<State = State>>,
    request: Arc<Request>,
    dir: DirLock<CacheBlock>,
}

impl<State> Txn<State> {
    fn new(
        active: Arc<Active>,
        gateway: Arc<dyn Gateway<State = State>>,
        request: Request,
    ) -> Self {
        let request = Arc::new(request);
        let dir = active.workspace.clone();

        Self {
            active,
            gateway,
            request,
            dir,
        }
    }
}

impl<State> Txn<State>
where
    State: StateInstance<FE = CacheBlock, Txn = Self>,
{
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
    pub async fn grant(
        &self,
        actor: &Actor,
        cluster_path: TCPathBuf,
        scopes: Vec<Scope>,
    ) -> TCResult<Self> {
        let token = self.request.token().to_string();
        let txn_id = self.request.txn_id();

        use rjwt::Resolve;
        let host = self.gateway.link(cluster_path);
        let resolver = Resolver::new(&*self.gateway, &host, self.request.txn_id());

        debug!(
            "granting scopes {} to {}",
            Tuple::<Scope>::from_iter(scopes.clone()),
            host
        );

        let (token, claims) = resolver
            .consume_and_sign(actor, scopes, token, txn_id.time().into())
            .map_err(|cause| unauthorized!("signature error").consume(cause))
            .await?;

        Ok(Self {
            active: self.active.clone(),
            gateway: self.gateway.clone(),
            dir: self.dir.clone(),
            request: Arc::new(Request::new(*txn_id, token, claims)),
        })
    }

    /// Claim ownership of this transaction.
    pub async fn claim(self, actor: &Actor, cluster_path: TCPathBuf) -> TCResult<Self> {
        debug!(
            "{} claims ownership of transaction {}",
            cluster_path,
            self.request.txn_id()
        );

        if actor.id().is_some() {
            return Err(bad_request!("cluster ID must be None, not {}", actor.id()));
        }

        if self.owner().is_none() {
            self.grant(actor, cluster_path, vec![self.active.scope().clone()])
                .await
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
        let active_scope = self.active.scope();

        self.request
            .scopes()
            .iter()
            .filter(|(_, actor_id, _)| *actor_id == &Value::None)
            .filter_map(|(host, _actor_id, scopes)| {
                if scopes.contains(active_scope) {
                    Some(host)
                } else {
                    None
                }
            })
            .fold(None, |_, host| Some(host))
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
    pub async fn lead(self, actor: &Actor, cluster_path: TCPathBuf) -> TCResult<Self> {
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
            let scopes = vec![self.active.scope().clone()];
            self.grant(actor, cluster_path, scopes).await
        }
    }

    /// Return the leader of this transaction for the given cluster, if there is one.
    pub fn leader(&self, cluster_path: &[PathSegment]) -> Option<&Link> {
        let active_scope = self.active.scope();

        self.request
            .scopes()
            .iter()
            .filter(|(_, actor_id, _)| *actor_id == &Value::None)
            .filter_map(|(host, _actor_id, scopes)| {
                if scopes.contains(active_scope) && cluster_path.starts_with(host.path()) {
                    Some(host)
                } else {
                    None
                }
            })
            .next()
    }
}

#[async_trait]
impl<State: Clone + 'static> Transaction<CacheBlock> for Txn<State> {
    #[inline]
    fn id(&'_ self) -> &'_ TxnId {
        self.request.txn_id()
    }

    fn context(&'_ self) -> &'_ tc_transact::fs::Inner<CacheBlock> {
        &self.dir
    }

    // TODO: accept a ToOwned<Id>
    async fn subcontext(&self, id: Id) -> TCResult<Self> {
        let dir = {
            let mut dir = self.dir.write().await;
            dir.create_dir(id.to_string())?
        };

        Ok(Txn {
            active: self.active.clone(),
            gateway: self.gateway.clone(),
            request: self.request.clone(),
            dir,
        })
    }

    async fn subcontext_unique(&self) -> TCResult<Self> {
        let (_, subcontext) = self.dir.write().await.create_dir_unique()?;

        Ok(Self {
            active: self.active.clone(),
            gateway: self.gateway.clone(),
            request: self.request.clone(),
            dir: subcontext,
        })
    }
}

#[async_trait]
impl<State> tc_transact::RPCClient<State> for Txn<State>
where
    State: StateInstance<FE = CacheBlock, Txn = Self>,
{
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

impl<State> Hash for Txn<State> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.request.txn_id().hash(state)
    }
}
