use std::ops::Deref;
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
use tc_state::CacheBlock;
use tc_transact::lock::TxnSetLockIter;
use tc_transact::{Gateway, Transaction, TxnId};
use tc_value::{Link, ToUrl, Value};
use tcgeneric::{label, Id, Map, NetworkTime, PathSegment, TCPathBuf};

use crate::claim::Claim;
use crate::client::Client;
use crate::{Actor, RPCClient, SignedToken, State};

pub use hypothetical::Hypothetical;
pub use server::TxnServer;

mod hypothetical;
mod server;

pub(super) enum LazyDir {
    Workspace(DirLock<CacheBlock>),
    Lazy(Arc<Self>, Id),
}

impl Clone for LazyDir {
    fn clone(&self) -> Self {
        match self {
            Self::Workspace(workspace) => Self::Workspace(workspace.clone()),
            Self::Lazy(parent, id) => Self::Lazy(parent.clone(), id.clone()),
        }
    }
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

impl From<DirLock<CacheBlock>> for LazyDir {
    fn from(dir: DirLock<CacheBlock>) -> Self {
        Self::Workspace(dir)
    }
}

pub struct Txn {
    id: TxnId,
    expires: NetworkTime,
    workspace: LazyDir,
    token: Option<Arc<SignedToken>>,
    client: Client,
}

impl Clone for Txn {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            expires: self.expires,
            workspace: self.workspace.clone(),
            token: self.token.clone(),
            client: self.client.clone(),
        }
    }
}

impl Txn {
    pub(super) fn new(
        id: TxnId,
        expires: NetworkTime,
        workspace: LazyDir,
        client: Client,
        token: Option<SignedToken>,
    ) -> Self {
        Self {
            id,
            expires,
            workspace,
            client,
            token: token.map(Arc::new),
        }
    }

    pub(crate) fn token(&self) -> Option<&SignedToken> {
        self.token.as_ref().map(|token| &**token)
    }

    /// Grant `mode` permissions on the resource at `path` to the bearer of this [`Txn`]'s token.
    /// `path` is relative to the cluster at `link` whose `actor` will sign the token.
    pub fn grant(&self, actor: &Actor, link: Link, path: TCPathBuf, mode: Mode) -> TCResult<Self> {
        let now = NetworkTime::now();
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
            client: self.client.clone(),
            token: Some(Arc::new(token)),
        })
    }

    /// Get the set of permissions authorized by hosts in the `keyring` for the given `resource`.
    pub fn mode<K>(&self, keyring: TxnSetLockIter<K>, resource: &[PathSegment]) -> Mode
    where
        K: Deref<Target = Actor>,
    {
        let mut mode = Mode::new().with_class_perm(umask::OTHERS, umask::ALL);

        // TODO: add user & group permissions

        mode
    }

    pub fn claim(self, actor: &Actor, path: &[PathSegment]) -> TCResult<Self> {
        debug_assert!(self.leader(path).is_none());

        self.grant(
            actor,
            TCPathBuf::from_slice(path).into(),
            self.txn_path(),
            umask::USER_EXEC,
        )
    }

    pub fn has_claims(&self) -> bool {
        self.token.is_some()
    }

    pub fn leader(&self, path: &[PathSegment]) -> Option<VerifyingKey> {
        if let Some(token) = &self.token {
            for (host, actor_id, path) in token.claims() {
                todo!("Txn::leader")
            }
        }

        None
    }

    pub fn owner(&self) -> Option<VerifyingKey> {
        todo!("Txn::owner")
    }

    #[inline]
    fn txn_path(&self) -> TCPathBuf {
        [label("txn").into(), self.id.to_id()].into()
    }
}

#[async_trait]
impl Transaction<CacheBlock> for Txn {
    #[inline]
    fn id(&self) -> &TxnId {
        &self.id
    }

    async fn context(&self) -> TCResult<DirLock<CacheBlock>> {
        self.workspace.get_or_create(&self.id).await
    }

    fn subcontext<I: Into<Id> + Send>(&self, id: I) -> Self {
        Txn {
            id: self.id,
            expires: self.expires,
            workspace: self.workspace.clone().create_dir(id.into()),
            client: self.client.clone(),
            token: self.token.clone(),
        }
    }

    fn subcontext_unique(&self) -> Self {
        Txn {
            id: self.id,
            expires: self.expires,
            workspace: self.workspace.clone().create_dir_unique(),
            client: self.client.clone(),
            token: self.token.clone(),
        }
    }
}

#[async_trait]
impl Gateway<State> for Txn {
    async fn get<'a, L, V>(&'a self, link: L, key: V) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send,
    {
        self.client.get(self, link.into(), key.cast_into()).await
    }

    async fn put<'a, L, K, V>(&'a self, link: L, key: K, value: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        K: CastInto<Value> + Send,
        V: CastInto<State> + Send,
    {
        self.client
            .put(self, link.into(), key.cast_into(), value.cast_into())
            .await
    }

    async fn post<'a, L, P>(&'a self, link: L, params: P) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        P: CastInto<Map<State>> + Send,
    {
        self.client
            .post(self, link.into(), params.cast_into())
            .await
    }

    async fn delete<'a, L, V>(&'a self, link: L, key: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send,
    {
        self.client.delete(self, link.into(), key.cast_into()).await
    }
}
