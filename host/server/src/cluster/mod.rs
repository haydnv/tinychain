use std::collections::HashMap;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures::future::{Future, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::debug;
use rjwt::{OsRng, SigningKey, Token, VerifyingKey};
use umask::Mode;

use tc_error::*;
use tc_transact::lock::{TxnLock, TxnMapLockIter};
use tc_transact::{fs, Gateway};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Host, Link, Value};
use tcgeneric::{PathSegment, TCPath, TCPathBuf};

use crate::claim::Claim;
use crate::txn::Txn;
use crate::{default_host, Actor, CacheBlock, SignedToken, State};

pub use class::Class;
pub use dir::{Dir, DirEntry};

pub const DEFAULT_UMASK: Mode = Mode::new()
    .with_class_perm(umask::USER, umask::ALL)
    .with_class_perm(umask::GROUP, umask::ALL)
    .with_class_perm(umask::OTHERS, umask::READ);

mod class;
mod dir;
mod public;

#[derive(Clone, Debug)]
pub struct Schema {
    lead: Host,
    link: Link,
    owner: Option<Link>,
    group: Option<Link>,
}

impl Schema {
    pub fn new(lead: Host, link: Link, owner: Option<Link>, group: Option<Link>) -> Self {
        Self {
            lead,
            link,
            owner,
            group,
        }
    }

    pub fn append(self, suffix: PathSegment) -> Self {
        Self {
            lead: self.lead,
            link: self.link.append(suffix),
            owner: self.owner,
            group: self.group,
        }
    }
}

#[async_trait]
pub trait Replicate: Send + Sync {
    async fn replicate(&self, txn: &Txn, peer: Host) -> TCResult<()>;
}

struct Inner {
    schema: Schema,
    actor: Actor,
}

#[derive(Clone)]
pub struct Cluster<T> {
    inner: Arc<Inner>,
    mode: Mode,
    state: T,
    replicas: TxnLock<HashMap<Host, Actor>>,
}

impl<T> Cluster<T> {
    pub fn new(schema: Schema, state: T, txn_id: TxnId) -> Self {
        let keypair = SigningKey::generate(&mut OsRng);
        let public_key = keypair.verifying_key();
        let actor_id = Value::Bytes(Arc::from(*public_key.as_bytes()));
        let actor = Actor::with_keypair(actor_id, keypair);

        let replicas = [(
            schema.link.host().cloned().unwrap_or_else(default_host),
            actor.clone(),
        )]
        .into_iter()
        .collect();

        let replicas = TxnLock::new(replicas);

        let mode = if schema.owner.is_some() || schema.group.is_some() {
            DEFAULT_UMASK
        } else {
            Mode::all()
        };

        Self {
            inner: Arc::new(Inner { schema, actor }),
            mode,
            state,
            replicas,
        }
    }

    pub fn link(&self) -> &Link {
        &self.inner.schema.link
    }
    pub fn path(&self) -> &TCPathBuf {
        self.inner.schema.link.path()
    }

    pub fn umask(&self, _txn_id: TxnId, path: &[PathSegment]) -> Mode {
        assert!(
            path.is_empty(),
            "TODO: cluster subject umask for {}",
            TCPath::from(path)
        );
        self.mode
    }

    #[inline]
    pub fn claim(&self, txn: Txn) -> TCResult<Txn> {
        if txn.leader(self.path())?.is_none() {
            txn.claim(self.link().clone(), &self.inner.actor)
        } else {
            Ok(txn)
        }
    }

    #[inline]
    pub fn lock(&self, txn: Txn) -> TCResult<Txn> {
        txn.lock(&self.inner.actor)
    }

    pub(crate) fn issue_token(&self, mode: Mode, ttl: Duration) -> TCResult<SignedToken> {
        assert!(self.inner.actor.has_private_key());

        let link = self.link().clone();
        let actor_id = self.inner.actor.id().clone();
        let claim = Claim::new(self.path().clone(), mode);
        let token = Token::new(link, SystemTime::now(), ttl, actor_id, claim);
        self.inner.actor.sign_token(token).map_err(TCError::from)
    }

    pub fn keyring(&self, txn_id: TxnId) -> TCResult<impl Deref<Target = HashMap<Host, Actor>>> {
        self.replicas.try_read(txn_id).map_err(TCError::from)
    }

    #[inline]
    pub fn public_key(&self) -> VerifyingKey {
        self.inner.actor.public_key()
    }

    pub(crate) fn state(&self) -> &T {
        &self.state
    }

    async fn replicate_write<Write, Fut>(
        &self,
        txn_id: TxnId,
        path: &[PathSegment],
        op: Write,
    ) -> TCResult<()>
    where
        Write: Fn(Link) -> Fut,
        Fut: Future<Output = TCResult<()>>,
    {
        let mut uri = self.path().clone();
        uri.extend(path.into_iter().cloned());

        let replicas = self.replicas.write(txn_id).await?;

        let this_host = self.link().host().cloned().unwrap_or_else(default_host);
        let mut writes: FuturesUnordered<_> = replicas
            .keys()
            .filter(|host| !(host.is_localhost() && host.port() == this_host.port()))
            .filter(|host| **host != this_host)
            .map(|host| op(Link::new(host.clone(), uri.clone())))
            .collect();

        let mut failed = 0;

        while let Some(result) = writes.next().await {
            if result.is_ok() {
                // no-op
            } else {
                failed += 1;
            }
        }

        if failed > (replicas.len() / 2) {
            todo!("remove failed replicas")
        } else {
            Ok(())
        }
    }
}

impl<T: Transact + Send + Sync + fmt::Debug> Cluster<T> {
    async fn replicate_commit(&self, txn: &Txn) -> TCResult<()> {
        let leader = txn
            .leader(self.path())?
            .ok_or_else(|| bad_request!("cannot commit a leaderless transaction"))?;

        if leader == self.public_key() {
            if txn.locked_by()? == Some(self.public_key()) {
                self.replicate_commit_inner(txn).await?;
            } else {
                return Err(unauthorized!("commit {:?}", self));
            }
        }

        self.commit(*txn.id()).await;

        Ok(())
    }

    async fn replicate_commit_inner(&self, txn: &Txn) -> TCResult<()> {
        let replicas = self.replicas.read(*txn.id()).await?;

        let this_host = self.link().host().cloned().unwrap_or_else(default_host);

        let mut commits: FuturesUnordered<_> = replicas
            .keys()
            .filter(|host| !(host.is_localhost() && host.port() == this_host.port()))
            .filter(|host| **host != this_host)
            .cloned()
            .map(|replica| async move {
                let link = Link::new(replica.clone(), self.path().clone());
                txn.put(link, Value::default(), State::default())
                    .map(|result| (replica, result))
                    .await
            })
            .collect();

        while let Some((host, result)) = commits.next().await {
            if result.is_ok() {
                debug!("replica at {host} succeeded");
            } else {
                // TODO: enqueue a message to initiate a rebalance of the replica set
                panic!("replica at {host} failed--the replica set is now out of sync");
            }
        }

        Ok(())
    }

    async fn replicate_rollback(&self, txn: &Txn) -> TCResult<()> {
        // TODO: validate that the rollback message came from the txn leader, send rollback messages to replicas, log errors, crash if >= 50% fail
        Err(not_implemented!("Cluster::replicate_rollback"))
    }
}

#[async_trait]
pub(crate) trait ReplicateAndJoin {
    type State;

    async fn replicate_and_join(
        &self,
        txn: &Txn,
        peer: Host,
    ) -> TCResult<TxnMapLockIter<PathSegment, DirEntry<Self::State>>>;
}

#[async_trait]
impl<T: Send + Sync + fmt::Debug> ReplicateAndJoin for Cluster<Dir<T>>
where
    T: Send + Sync + fmt::Debug,
    Cluster<T>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema>,
    Cluster<Dir<T>>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema>,
    DirEntry<T>: Clone,
{
    type State = T;

    async fn replicate_and_join(
        &self,
        txn: &Txn,
        peer: Host,
    ) -> TCResult<TxnMapLockIter<PathSegment, DirEntry<T>>> {
        self.state.replicate(txn, peer).await?;
        self.state.entries(*txn.id()).await
    }
}

impl<T> Cluster<Dir<T>>
where
    T: Clone + fmt::Debug,
{
    pub fn lookup<'a>(
        &self,
        txn_id: TxnId,
        path: &'a [PathSegment],
    ) -> TCResult<(&'a [PathSegment], DirEntry<T>)> {
        match self.state().lookup(txn_id, path)? {
            Some((path, entry)) => Ok((path, entry)),
            None => Ok((path, DirEntry::Dir(self.clone()))),
        }
    }
}

#[async_trait]
impl<T> Transact for Cluster<T>
where
    T: Transact + Send + Sync,
{
    type Commit = T::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        self.replicas.commit(txn_id);
        self.state.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.replicas.rollback(txn_id);
        self.state.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.replicas.finalize(*txn_id);
        self.state.finalize(txn_id).await
    }
}

#[async_trait]
impl fs::Persist<CacheBlock> for Cluster<Class> {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        <Class as fs::Persist<CacheBlock>>::create(txn_id, (), store)
            .map_ok(|subject| Self::new(schema, subject, txn_id))
            .await
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        <Class as fs::Persist<CacheBlock>>::load(txn_id, (), store)
            .map_ok(|subject| Self::new(schema, subject, txn_id))
            .await
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        <Class as fs::Persist<CacheBlock>>::dir(&self.state)
    }
}

#[async_trait]
impl fs::Persist<CacheBlock> for Cluster<Dir<Class>> {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        <Dir<Class> as fs::Persist<CacheBlock>>::create(txn_id, schema.clone(), store)
            .map_ok(|subject| Self::new(schema, subject, txn_id))
            .await
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        <Dir<Class> as fs::Persist<CacheBlock>>::load(txn_id, schema.clone(), store)
            .map_ok(|subject| Self::new(schema, subject, txn_id))
            .await
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        <Dir<Class> as fs::Persist<CacheBlock>>::dir(&self.state)
    }
}

impl<T: fmt::Debug> fmt::Debug for Cluster<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?} cluster", self.state)
    }
}
