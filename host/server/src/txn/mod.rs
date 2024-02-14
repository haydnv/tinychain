use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::DirLock;
use futures::Future;
use rjwt::{Token, VerifyingKey};
use safecast::CastInto;
use umask::Mode;
use uuid::Uuid;

use tc_error::*;
use tc_transact::public::StateInstance;
use tc_transact::{RPCClient, Transaction, TxnId};
use tc_value::{Link, ToUrl, Value};
use tcgeneric::{Id, NetworkTime, PathSegment, TCPathBuf, ThreadSafe};

use crate::claim::Claim;
use crate::gateway::Gateway;
use crate::{Actor, SignedToken};

pub use hypothetical::Hypothetical;
pub use server::TxnServer;

mod hypothetical;
mod server;

pub(super) enum LazyDir<FE> {
    Workspace(DirLock<FE>),
    Lazy(Arc<Self>, Id),
}

impl<FE> Clone for LazyDir<FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Workspace(workspace) => Self::Workspace(workspace.clone()),
            Self::Lazy(parent, id) => Self::Lazy(parent.clone(), id.clone()),
        }
    }
}

impl<FE: Send + Sync> LazyDir<FE> {
    fn get_or_create<'a>(
        &'a self,
        txn_id: &'a TxnId,
    ) -> Pin<Box<dyn Future<Output = TCResult<DirLock<FE>>> + Send + 'a>> {
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

impl<FE> From<DirLock<FE>> for LazyDir<FE> {
    fn from(dir: DirLock<FE>) -> Self {
        Self::Workspace(dir)
    }
}

pub struct Txn<FE> {
    id: TxnId,
    expires: NetworkTime,
    workspace: LazyDir<FE>,
    gateway: Arc<Gateway>,
    token: Option<Arc<SignedToken>>,
}

impl<FE> Clone for Txn<FE> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            expires: self.expires,
            workspace: self.workspace.clone(),
            gateway: self.gateway.clone(),
            token: self.token.clone(),
        }
    }
}

impl<FE> Txn<FE> {
    pub(super) fn new(
        id: TxnId,
        expires: NetworkTime,
        workspace: LazyDir<FE>,
        gateway: Arc<Gateway>,
        token: Option<SignedToken>,
    ) -> Self {
        Self {
            id,
            expires,
            workspace,
            gateway,
            token: token.map(Arc::new),
        }
    }

    /// Grant `mode` permissions on the resource at `path` to the bearer of this [`Txn`]'s token.
    /// `path` is relative to the cluster at `link` whose `actor` will sign the token.
    pub fn grant(&self, actor: &Actor, link: Link, path: TCPathBuf, mode: Mode) -> TCResult<Self> {
        let now = self.gateway.now();
        let claim = Claim::new(path, mode);

        let token = if let Some(token) = &self.token {
            actor.consume_and_sign((**token).clone(), link, claim, now.into())
        } else {
            let ttl = self.expires - now;
            let token = Token::new(link, now.into(), ttl, actor.id().clone(), claim);
            actor.sign_token(token)
        }?;

        Ok(Self {
            id: self.id,
            expires: self.expires,
            workspace: self.workspace.clone(),
            gateway: self.gateway.clone(),
            token: Some(Arc::new(token)),
        })
    }

    pub fn mode(&self, actor: &Link, resource: &Link) -> Option<Mode> {
        todo!("Txn::mode")
    }

    pub fn claim(self, public_key: &VerifyingKey, path: &[PathSegment]) -> Self {
        debug_assert!(self.leader(path).is_none());
        todo!("Txn::claim")
    }

    pub fn leader(&self, path: &[PathSegment]) -> Option<&VerifyingKey> {
        todo!("Txn::leader")
    }

    pub fn owner(&self) -> Option<&VerifyingKey> {
        todo!("Txn::owner")
    }
}

#[async_trait]
impl<FE: ThreadSafe + Clone> Transaction<FE> for Txn<FE> {
    #[inline]
    fn id(&self) -> &TxnId {
        &self.id
    }

    async fn context(&self) -> TCResult<DirLock<FE>> {
        self.workspace.get_or_create(&self.id).await
    }

    fn subcontext<I: Into<Id> + Send>(&self, id: I) -> Self {
        Txn {
            id: self.id,
            expires: self.expires,
            workspace: self.workspace.clone().create_dir(id.into()),
            gateway: self.gateway.clone(),
            token: self.token.clone(),
        }
    }

    fn subcontext_unique(&self) -> Self {
        Txn {
            id: self.id,
            expires: self.expires,
            workspace: self.workspace.clone().create_dir_unique(),
            gateway: self.gateway.clone(),
            token: self.token.clone(),
        }
    }
}

#[async_trait]
impl<FE, State> RPCClient<State> for Txn<FE>
where
    FE: ThreadSafe + Clone,
    State: StateInstance<FE = FE, Txn = Self>,
{
    async fn get<'a, L, V>(&'a self, link: L, key: V) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send,
    {
        todo!()
    }

    async fn put<'a, L, K, V>(&'a self, link: L, key: K, value: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        K: CastInto<Value> + Send,
        V: CastInto<State> + Send,
    {
        todo!()
    }

    async fn post<'a, L, P>(&'a self, link: L, params: P) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        P: CastInto<State> + Send,
    {
        todo!()
    }

    async fn delete<'a, L, V>(&'a self, link: L, key: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send,
    {
        todo!()
    }
}
