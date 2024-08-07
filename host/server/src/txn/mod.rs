use std::collections::HashMap;
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
use tc_transact::Gateway;
use tc_value::{Host, Link, ToUrl, Value};
use tcgeneric::{label, Id, Label, Map, NetworkTime, PathSegment, TCPathBuf};

use crate::claim::Claim;
use crate::client::{Client, Egress};
use crate::{Actor, RPCClient, SignedToken, State};

pub use hypothetical::Hypothetical;
pub use server::TxnServer;
pub use tc_transact::{IntoView, Transaction, TxnId};

mod hypothetical;
mod server;

const PREFIX: Label = label("txn");

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
    pub const DEFAULT_MODE: Mode = Mode::new().with_class_perm(umask::OTHERS, umask::ALL);

    #[inline]
    fn validate_token(id: TxnId, token: &SignedToken) -> TCResult<()> {
        let mut owner = None;
        let mut lock = None;

        for (host, actor_id, claim) in token.claims() {
            if host.path().is_empty() {
                return Err(bad_request!(
                    "invalid token: cannot claim reserved path {}",
                    claim.path()
                ));
            } else if !claim.path().is_empty() && claim.path()[0] == PREFIX {
                if claim.path().len() == 2 && id == claim.path()[1] {
                    if claim.mode().has(umask::USER_EXEC) {
                        if owner.is_none() {
                            owner = Some((host, actor_id));
                        } else {
                            return Err(bad_request!("invalid token: multiple owners"));
                        }
                    }

                    if claim.mode().has(umask::USER_WRITE) {
                        if lock.is_none() {
                            lock = Some((host, actor_id));
                        } else {
                            return Err(bad_request!("invalid token: multiple locks"));
                        }
                    }
                } else {
                    return Err(bad_request!(
                        "cannot initialize txn {id} with a token for {}",
                        claim.path()[1]
                    ));
                }
            }
        }

        if let Some(lock) = lock {
            if let Some(owner) = owner {
                if owner == lock {
                    // pass
                } else {
                    return Err(bad_request!("invalid token: lock does not match owner"));
                }
            } else {
                return Err(bad_request!("invalid token: lock with no owner"));
            }
        }

        Ok(())
    }

    pub(super) fn new(
        id: TxnId,
        expires: NetworkTime,
        workspace: LazyDir,
        client: Client,
        token: Option<SignedToken>,
    ) -> TCResult<Self> {
        if let Some(token) = &token {
            Self::validate_token(id, token)?;
        }

        Ok(Self {
            id,
            expires,
            workspace,
            client,
            token: token.map(Arc::new),
        })
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
    pub fn mode<Keyring>(&self, _keyring: Keyring, _resource: &[PathSegment]) -> Mode
    where
        Keyring: Deref<Target = HashMap<Host, VerifyingKey>>,
    {
        let mode = Self::DEFAULT_MODE;

        // TODO: add user & group permissions

        mode
    }

    // TODO: require a write bit for commits and a read bit for rollbacks
    pub fn locked_by(&self) -> TCResult<Option<VerifyingKey>> {
        let token = if let Some(token) = &self.token {
            token
        } else {
            return Ok(None);
        };

        let mut claims = token.claims().into_iter();

        let mut locked_by = None;
        while let Some((_host, actor_id, claim)) = claims.next() {
            if self.is_txn_path(claim.path()) {
                if claim.mode().has(umask::USER_WRITE) {
                    locked_by = Some(actor_id);
                    break;
                }
            }
        }

        let mut owned_by = None;
        while let Some((_host, actor_id, claim)) = claims.next() {
            if self.is_txn_path(claim.path()) {
                if claim.mode().has(umask::USER_EXEC) {
                    owned_by = Some(actor_id);
                }
            }
        }

        if let Some(locked_by) = locked_by {
            let owner = owned_by.ok_or_else(|| internal!("ownerless tranaction"))?;

            if locked_by == owner {
                let locked_by = Arc::<[u8]>::try_from(locked_by.clone())?;

                VerifyingKey::try_from(&*locked_by)
                    .map_err(|cause| bad_request!("invalid public key for txn leader: {cause}"))
                    .map(Some)
            } else {
                Err(internal!("txn locked by non-owner"))
            }
        } else {
            Ok(None)
        }
    }

    pub fn claim(self, link: Link, actor: &Actor) -> TCResult<Self> {
        debug_assert!(self.leader(link.path()).expect("leader").is_none());

        let txn = if self.owner()?.is_none() {
            self.grant(actor, link.clone(), self.txn_path(), umask::USER_EXEC)?
        } else {
            self
        };

        txn.grant(actor, link, TCPathBuf::default(), umask::USER_EXEC)
    }

    pub fn lock(self, actor: &Actor) -> TCResult<Self> {
        let mut owner = None;
        if let Some(token) = &self.token {
            for (host, actor_id, claim) in token.claims() {
                if self.is_txn_path(claim.path()) && claim.mode().has(umask::USER_EXEC) {
                    owner = Some((host, actor_id));
                }
            }
        }

        let link = if let Some((link, public_key)) = owner {
            assert_eq!(
                *public_key,
                Value::Bytes((*actor.public_key().as_bytes()).into())
            );

            link.clone()
        } else {
            panic!("tried to lock a transaction with no owner")
        };

        self.grant(actor, link, self.txn_path(), umask::USER_WRITE)
    }

    pub fn has_claims(&self) -> bool {
        self.token.is_some()
    }

    pub fn leader(&self, path: &[PathSegment]) -> TCResult<Option<(Link, VerifyingKey)>> {
        if let Some(token) = &self.token {
            for (host, actor_id, claim) in token.claims() {
                if host.path() == path
                    && claim.path().is_empty()
                    && claim.mode().has(umask::USER_EXEC)
                {
                    let public_key = Arc::<[u8]>::try_from(actor_id.clone())?;
                    let public_key = VerifyingKey::try_from(&*public_key)
                        .map_err(|cause| bad_request!("invalid leader key: {cause}"))?;

                    return Ok(Some((host.clone(), public_key)));
                }
            }
        }

        Ok(None)
    }

    pub fn owner(&self) -> TCResult<Option<VerifyingKey>> {
        let mut owner = None;

        if let Some(token) = &self.token {
            for (_host, actor_id, claim) in token.claims() {
                if self.is_txn_path(claim.path()) && claim.mode().has(umask::USER_EXEC) {
                    owner = Some(actor_id);
                }
            }
        }

        if let Some(owner) = owner {
            let public_key = Arc::<[u8]>::try_from(owner.clone())?;
            let public_key = VerifyingKey::try_from(&*public_key)
                .map_err(|cause| bad_request!("invalid owner key: {cause}"))?;

            Ok(Some(public_key))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn with_egress(self, egress: Arc<dyn Egress>) -> Self {
        Self {
            id: self.id,
            expires: self.expires,
            workspace: self.workspace,
            client: self.client.with_egress(egress),
            token: self.token,
        }
    }

    #[inline]
    fn txn_path(&self) -> TCPathBuf {
        [PREFIX.into(), self.id.to_id()].into()
    }

    #[inline]
    fn is_txn_path(&self, path: &[PathSegment]) -> bool {
        path.len() == 2 && PREFIX == path[0] && self.id == path[1]
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
