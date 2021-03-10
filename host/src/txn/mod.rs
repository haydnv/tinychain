//! The transaction context [`Txn`].

use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::TryFutureExt;
use log::debug;

use tc_error::*;
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tcgeneric::{Id, NetworkTime, PathSegment, TCPathBuf, Tuple};

use crate::fs;
use crate::gateway::Gateway;
use crate::scalar::{Link, Value};
use crate::state::State;

mod request;
mod server;

pub use request::*;
pub use server::*;
pub use tc_transact::TxnId;

struct Active {
    expires: NetworkTime,
    scope: Scope,
}

impl Active {
    fn new(txn_id: &TxnId, expires: NetworkTime) -> Self {
        let scope = TCPathBuf::from(txn_id.to_id());
        Self { expires, scope }
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
pub struct Txn {
    active: Arc<Active>,
    gateway: Arc<Gateway>,
    request: Arc<Request>,
    dir: fs::Dir,
}

impl Txn {
    fn new(active: Arc<Active>, gateway: Arc<Gateway>, dir: fs::Dir, request: Request) -> Self {
        let request = Arc::new(request);

        Self {
            active,
            gateway,
            request,
            dir,
        }
    }

    /// Return the current number of strong references to this `Txn`.
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.active)
    }

    /// Claim ownership of this transaction.
    pub async fn claim(self, actor: &Actor, cluster_path: TCPathBuf) -> TCResult<Self> {
        debug!("{} claims transaction {}", cluster_path, self.id());

        if actor.id().is_some() {
            return Err(TCError::bad_request(
                "cluster ID must be None, not",
                actor.id(),
            ));
        }

        if self.owner().is_none() {
            self.grant(actor, cluster_path, vec![self.active.scope().clone()])
                .await
        } else {
            Err(TCError::forbidden(
                "tried to claim owned transaction",
                self.id(),
            ))
        }
    }

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
        let resolver = Resolver::new(&self.gateway, &host, self.id());

        debug!(
            "granting scopes {} to {}",
            Tuple::<Scope>::from_iter(scopes.clone()),
            host
        );

        let (token, claims) = resolver
            .consume_and_sign(actor, scopes, token, txn_id.time().into())
            .map_err(TCError::unauthorized)
            .await?;

        Ok(Self {
            active: self.active.clone(),
            gateway: self.gateway.clone(),
            dir: self.dir.clone(),
            request: Arc::new(Request::new(*txn_id, token, claims)),
        })
    }

    /// Check if the cluster at the specified path on this host is the owner of the transaction.
    pub fn is_owner(&self, cluster_path: &[PathSegment]) -> bool {
        if let Some(host) = self.owner() {
            host.host().as_ref().expect("txn owner hostname") == self.gateway.root()
                && host.path().as_slice() == cluster_path
        } else {
            false
        }
    }

    /// Return the owner of this transaction, if there is one.
    pub fn owner(&self) -> Option<&Link> {
        for (host, _actor_id, scopes) in self.request.scopes().iter() {
            if scopes.contains(self.active.scope()) {
                return Some(host);
            }
        }

        None
    }

    /// Return a link to the given path on this host.
    pub fn link(&self, path: TCPathBuf) -> Link {
        self.gateway.link(path)
    }

    /// Return the [`Request`] which initiated this transaction on this host.
    pub fn request(&'_ self) -> &'_ Request {
        &self.request
    }

    /// Resolve a GET op within this transaction context.
    pub async fn get(&self, link: Link, key: Value) -> TCResult<State> {
        self.gateway.get(self, link, key).await
    }

    /// Resolve a PUT op within this transaction context.
    pub async fn put(&self, link: Link, key: Value, value: State) -> TCResult<()> {
        self.gateway.put(self, link, key, value).await
    }

    /// Resolve a POST op within this transaction context.
    pub async fn post(&self, link: Link, params: State) -> TCResult<State> {
        self.gateway.post(self, link, params).await
    }

    /// Resolve a DELETE op within this transaction context.
    pub async fn delete(&self, link: Link, key: Value) -> TCResult<()> {
        self.gateway.delete(self, link, key).await
    }
}

#[async_trait]
impl Transaction<fs::Dir> for Txn {
    fn id(&'_ self) -> &'_ TxnId {
        self.request.txn_id()
    }

    fn context(&'_ self) -> &'_ fs::Dir {
        &self.dir
    }

    async fn subcontext(&self, id: Id) -> TCResult<Self> {
        let dir = self.dir.create_dir(*self.request.txn_id(), id).await?;

        Ok(Txn {
            active: self.active.clone(),
            gateway: self.gateway.clone(),
            request: self.request.clone(),
            dir,
        })
    }

    async fn subcontext_tmp(&self) -> TCResult<Self> {
        let id = loop {
            let id = uuid::Uuid::new_v4().to_string().parse()?;
            if !self.dir.contains(self.id(), &id).await? {
                break id;
            }
        };

        self.subcontext(id).await
    }
}

impl Hash for Txn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.request.txn_id().hash(state)
    }
}
