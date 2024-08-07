use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use futures::future::{Future, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::*;
use rjwt::{OsRng, SigningKey, Token, VerifyingKey};
use safecast::TryCastInto;
use umask::Mode;

use tc_error::*;
#[cfg(feature = "service")]
use tc_state::chain::Recover;
use tc_transact::hash::{Output, Sha256};
use tc_transact::lock::{TxnLock, TxnLockReadGuard, TxnMapLockIter};
use tc_transact::{fs, Gateway};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Host, Link, ToUrl, Value};
use tcgeneric::{label, Label, PathSegment, TCPathBuf, Tuple};

use crate::claim::Claim;
use crate::client::Egress;
use crate::txn::Txn;
use crate::{default_host, Actor, CacheBlock, SignedToken, State};

pub use class::Class;
pub use dir::{Dir, DirEntry};
pub use library::Library;
#[cfg(feature = "service")]
pub use service::Service;

mod class;
mod dir;
mod library;
mod public;
#[cfg(feature = "service")]
mod service;

pub const DEFAULT_UMASK: Mode = Mode::new()
    .with_class_perm(umask::USER, umask::ALL)
    .with_class_perm(umask::GROUP, umask::ALL)
    .with_class_perm(umask::OTHERS, umask::READ);

const REPLICAS: Label = label("replicas");

pub(crate) struct ClusterEgress {
    lead: Host,
    path: TCPathBuf,
    replicas: TxnLockReadGuard<HashMap<Host, VerifyingKey>>,
}

impl Egress for ClusterEgress {
    fn is_authorized(&self, _link: &ToUrl<'_>, _is_write: bool) -> bool {
        // TODO: implement egress control
        true
    }
}

impl fmt::Debug for ClusterEgress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "cluster at {}{} with replicas {:?}",
            self.lead, self.path, self.replicas
        )
    }
}

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
    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<Output<Sha256>>;
}

#[async_trait]
pub(super) trait IsDir {
    fn is_dir(&self) -> bool {
        false
    }

    async fn get_dir_item_key(
        &self,
        _txn_id: TxnId,
        _name: &PathSegment,
    ) -> TCResult<Option<(Output<Sha256>, VerifyingKey)>> {
        Ok(None)
    }

    async fn get_dir_item_keyring(
        &self,
        _txn_id: TxnId,
        _name: &PathSegment,
    ) -> TCResult<Option<TxnLockReadGuard<HashMap<Host, VerifyingKey>>>> {
        Ok(None)
    }
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
    replicas: TxnLock<HashMap<Host, VerifyingKey>>,
    claimed: Arc<Mutex<BTreeMap<TxnId, Arc<Mutex<HashMap<Link, bool>>>>>>,
}

impl<T> Cluster<T> {
    pub fn new(schema: Schema, state: T) -> Self {
        let keypair = SigningKey::generate(&mut OsRng);
        let public_key = keypair.verifying_key();
        let actor = Actor::with_keypair(Value::Bytes(Arc::new(*public_key.as_bytes())), keypair);

        let replicas = [(
            schema.link.host().cloned().unwrap_or_else(default_host),
            public_key,
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
            claimed: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn link(&self) -> &Link {
        &self.inner.schema.link
    }

    pub fn path(&self) -> &TCPathBuf {
        self.inner.schema.link.path()
    }

    pub fn umask(&self, _txn_id: TxnId, path: &[PathSegment]) -> Mode {
        if path.is_empty() || path == [REPLICAS] {
            self.mode
        } else {
            // TODO: granular permissions
            self.mode
        }
    }

    #[inline]
    fn authorize_egress(&self, txn: Txn) -> TCResult<Txn> {
        let whitelist = Arc::new(Mutex::new(HashMap::new()));

        let egress = ClusterEgress {
            lead: self.inner.schema.lead.clone(),
            path: self.path().clone(),
            replicas: self.replicas.try_read(*txn.id())?,
        };

        let mut claimed = self.claimed.lock().expect("claimed");
        claimed.insert(*txn.id(), whitelist);

        Ok(txn.with_egress(Arc::new(egress)))
    }

    #[inline]
    pub fn claim(&self, txn: Txn) -> TCResult<Txn> {
        if txn.leader(self.path())?.is_none() {
            txn.claim(self.link().clone(), &self.inner.actor)
                .and_then(|txn| self.authorize_egress(txn))
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

    #[inline]
    pub fn public_key(&self) -> VerifyingKey {
        self.inner.actor.public_key()
    }

    fn state(&self) -> &T {
        &self.state
    }
}

impl<T: fmt::Debug> Cluster<T> {
    pub fn keyring(
        &self,
        txn_id: TxnId,
    ) -> TCResult<TxnLockReadGuard<HashMap<Host, VerifyingKey>>> {
        trace!("read {self:?} keyring");
        self.replicas.try_read(txn_id).map_err(TCError::from)
    }

    fn keyring_mut(
        &self,
        txn_id: TxnId,
    ) -> TCResult<impl DerefMut<Target = HashMap<Host, VerifyingKey>>> {
        trace!("write {self:?} keyring");
        self.replicas.try_write(txn_id).map_err(TCError::from)
    }

    async fn replicate_write<Write, Fut>(
        &self,
        txn: &Txn,
        path: &[PathSegment],
        op: Write,
    ) -> TCResult<()>
    where
        Write: Fn(Txn, Link) -> Fut,
        Fut: Future<Output = TCResult<()>>,
    {
        let this_host = self.link().host().cloned().unwrap_or_else(default_host);

        let mut uri = self.path().clone();
        uri.extend(path.into_iter().cloned());

        let (num_replicas, mut failed) = {
            let replicas = self.replicas.read(*txn.id()).await?;
            let failed =
                Self::replicate_write_inner(&txn, &this_host, &*replicas, &uri, op).await?;

            (replicas.len(), failed)
        };

        if failed.is_empty() {
            return Ok(());
        }

        let txn = txn.grant(
            &self.inner.actor,
            self.link().clone(),
            TCPathBuf::default(),
            umask::USER_WRITE,
        )?;

        let uri = self.path().clone().append(REPLICAS);

        trace!("lock replica set of {self:?} in order to replicate a write operation");
        let mut replicas = self.replicas.write(*txn.id()).await?;
        let mut num_failed = failed.len();

        while !failed.is_empty() && num_failed * 2 < num_replicas {
            let mut to_remove = Tuple::<Value>::with_capacity(failed.len());
            for host in failed {
                replicas.remove(&host);
                to_remove.push(host.into());
            }

            failed =
                Self::replicate_write_inner(&txn, &this_host, &*replicas, &uri, |txn, link| {
                    let to_remove = to_remove.clone();
                    async move { txn.delete(link, to_remove).await }
                })
                .await?;

            num_failed += failed.len();
        }

        if num_failed * 2 > num_replicas {
            Err(bad_gateway!("a majority of replicas failed"))
        } else {
            Ok(())
        }
    }

    async fn replicate_write_inner<Write, Fut>(
        txn: &Txn,
        this_host: &Host,
        replicas: &HashMap<Host, VerifyingKey>,
        uri: &TCPathBuf,
        op: Write,
    ) -> TCResult<Vec<Host>>
    where
        Write: Fn(Txn, Link) -> Fut,
        Fut: Future<Output = TCResult<()>>,
    {
        let mut writes: FuturesUnordered<_> = replicas
            .keys()
            .filter(|host| !(host.is_localhost() && host.port() == this_host.port()))
            .filter(|host| *host != this_host)
            .map(|host| {
                op(txn.clone(), Link::new(host.clone(), uri.clone()))
                    .map(move |result| (host, result))
            })
            .collect();

        let mut failed = vec![];

        while let Some((host, result)) = writes.next().await {
            match result {
                Ok(()) => {} // no-op
                Err(cause) if cause.code().is_conflict() => {
                    return Err(cause);
                }
                Err(cause) => {
                    warn!("{host} failed: {cause}");
                    failed.push(host.clone());
                }
            }
        }

        Ok(failed)
    }
}

impl<T: Transact + Send + Sync + fmt::Debug> Cluster<T> {
    async fn replicate_commit(&self, txn: &Txn) -> TCResult<()> {
        debug!("Cluster::replicate_commit");

        // TODO: verify that the txn owner has permission to write to this cluster w/ a smart contract

        let (_leader, leader_pk) = txn
            .leader(self.path())?
            .ok_or_else(|| bad_request!("cannot commit a leaderless transaction"))?;

        if leader_pk == self.public_key() {
            if txn.locked_by()? == txn.owner()? {
                let txn_id = *txn.id();
                let notify = |link| txn.put(link, Value::default(), State::default());
                self.replicate_txn_state(txn_id, notify).await?;
            } else {
                return Err(unauthorized!("commit {:?}", self));
            }
        }

        self.commit(*txn.id()).await;

        debug!("Cluster::replicate_commit complete");

        Ok(())
    }

    pub async fn replicate_rollback(&self, txn: &Txn) -> TCResult<()> {
        debug!("Cluster::replicate_rollback");

        // TODO: verify that the txn owner has permission to write to this cluster w/ a smart contract

        let (_leader, leader_pk) = txn
            .leader(self.path())?
            .ok_or_else(|| bad_request!("cannot roll back a leaderless transaction"))?;

        if leader_pk == self.public_key() {
            if txn.locked_by()? == txn.owner()? {
                let txn_id = *txn.id();
                let notify = move |link| txn.delete(link, Value::default());
                self.replicate_txn_state(txn_id, notify).await?;
            } else {
                return Err(unauthorized!("roll back {:?}", self));
            }
        }

        self.rollback(txn.id()).await;

        debug!("Cluster::replicate_rollback complete");

        Ok(())
    }

    async fn replicate_txn_state<Notify, Fut>(&self, txn_id: TxnId, notify: Notify) -> TCResult<()>
    where
        Notify: Fn(Link) -> Fut + Copy + Send + Sync,
        Fut: Future<Output = TCResult<()>>,
    {
        let mut downstream = {
            let mut external = self.claimed.lock().expect("cluster deps");

            if let Some(deps) = external.remove(&txn_id) {
                let mut deps = deps.lock().expect("txn deps");

                deps.drain()
                    .filter_map(|(link, is_write)| if is_write { Some(link) } else { None })
                    .collect::<Tuple<Link>>()
            } else {
                vec![].into()
            }
        };

        debug!("deps are {}", downstream);

        let downstream = if !downstream.is_empty() {
            // TODO: delete this sort-and-filter logic after implementing an egress whitelist
            downstream.sort_by(|l, r| {
                l.host()
                    .cmp(&r.host())
                    .then(l.path().len().cmp(&r.path().len()))
            });

            let updates = FuturesUnordered::new();

            let mut last = Option::<Link>::None;

            for link in downstream {
                if let Some(last) = &last {
                    if link.path().starts_with(last.path()) {
                        debug!("no need to notify {link} since {last} was already notified");
                        continue;
                    }
                }

                last = Some(link.clone());

                updates.push(notify(link.clone()).map(move |r| (link, r)));
            }

            Some(updates)
        } else {
            None
        };

        let replicas = self.replicas.read(txn_id).await?;

        let this_host = self.link().host().cloned().unwrap_or_else(default_host);
        debug_assert!(replicas.contains_key(&this_host));

        let mut updates: FuturesUnordered<_> = replicas
            .keys()
            .filter(|host| !(host.is_localhost() && host.port() == this_host.port()))
            .filter(|host| **host != this_host)
            .cloned()
            .map(|replica| async move {
                let link = Link::new(replica.clone(), self.path().clone());
                trace!("notify replica at {link}...");
                notify(link).map(|result| (replica, result)).await
            })
            .collect();

        debug_assert_eq!(updates.len(), replicas.len() - 1);

        let mut num_failed = 0;
        while let Some((host, result)) = updates.next().await {
            if result.is_ok() {
                debug!("replica at {host} succeeded");
            } else {
                // TODO: enqueue a message to initiate a rebalance of the replica set
                error!("replica at {host} failed--the replica set is now out of sync!");
                num_failed += 1;
            }
        }

        if let Some(mut updates) = downstream {
            while let Some((link, result)) = updates.next().await {
                match result {
                    Ok(()) => debug!("synchronized dependency at {link}"),
                    Err(cause) => error!("dependency at {link} failed: {cause}"),
                }
            }
        }

        if num_failed * 2 > replicas.len() {
            panic!("most replicas failed--shutting down this replica since it's now out of sync")
        } else {
            Ok(())
        }
    }
}

#[async_trait]
pub(crate) trait ReplicateAndJoin {
    async fn replicate_and_join(&self, txn: Txn, peer: Host) -> TCResult<()>;
}

#[async_trait]
impl<T> ReplicateAndJoin for Cluster<T>
where
    T: Replicate + Transact + fmt::Debug,
{
    async fn replicate_and_join(&self, txn: Txn, peer: Host) -> TCResult<()> {
        info!("replicating {} from {}...", self.path(), peer);

        let peer_link = Link::new(peer, self.path().clone());
        let hash = self.state.replicate(&txn, peer_link.clone()).await?;

        let peer_link = peer_link.append(REPLICAS);
        let host = self
            .inner
            .schema
            .link
            .host()
            .cloned()
            .unwrap_or_else(default_host);

        let public_key = Value::Bytes(Arc::new(*self.public_key().as_bytes()));

        // since the keyring has to be write-locked in the next step,
        // peers will be unable to read-lock it in order to validate a leadership claim
        assert!(
            txn.leader(self.path())?.is_none(),
            "this transaction should not be claimed"
        );

        {
            trace!(
                "lock replica set of {:?} in order to join an active cluster",
                self
            );

            let mut keyring = self.replicas.write(*txn.id()).await?;

            let replica_set = txn.get(peer_link, Value::default()).await?;
            let replica_set: Tuple<(Host, Arc<[u8]>)> = replica_set
                .try_into_tuple(|s| TCError::unexpected(s, "a replica set"))
                .and_then(|tuple: Tuple<State>| tuple.into_iter().map(Value::try_from).collect())
                .and_then(|tuple: Tuple<Value>| {
                    tuple
                        .into_iter()
                        .map(|v| {
                            v.try_cast_into(|v| {
                                TCError::unexpected(v, "a replica host address and key")
                            })
                        })
                        .collect()
                })?;

            info!(
                "{self:?} will join a set of {} replicas...",
                replica_set.len()
            );

            let mut participants = HashMap::with_capacity(replica_set.len());
            for (host, public_key) in replica_set {
                let public_key = VerifyingKey::try_from(&*public_key)
                    .map_err(|v| TCError::unexpected(v, "a public key"))?;

                keyring.insert(host.clone(), public_key);
                participants.insert(Link::new(host, self.path().clone()), true);
            }

            let mut claimed = self.claimed.lock().expect("claimed txns");

            assert!(claimed
                .insert(*txn.id(), Arc::new(Mutex::new(participants)))
                .is_none());

            trace!("drop lock on replica set of {:?}", self);
        }

        self.replicate_write(&txn, &[REPLICAS.into()], |txn, link| {
            let host = host.clone();
            let public_key = public_key.clone();
            let hash = Value::Bytes(hash.as_slice().into());
            async move { txn.put(link, (host, public_key), hash).await }
        })
        .await?;

        let txn = txn.claim(self.link().clone(), &self.inner.actor)?;
        let txn = txn.lock(&self.inner.actor)?;

        self.replicate_commit(&txn).await
    }
}

#[async_trait]
impl<T: IsDir + Send + Sync> IsDir for Cluster<T> {
    fn is_dir(&self) -> bool {
        self.state.is_dir()
    }

    async fn get_dir_item_key(
        &self,
        txn_id: TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<(Output<Sha256>, VerifyingKey)>> {
        self.state.get_dir_item_key(txn_id, name).await
    }

    async fn get_dir_item_keyring(
        &self,
        txn_id: TxnId,
        name: &PathSegment,
    ) -> TCResult<Option<TxnLockReadGuard<HashMap<Host, VerifyingKey>>>> {
        self.state.get_dir_item_keyring(txn_id, name).await
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

impl<T: Clone + fmt::Debug> Cluster<Dir<T>> {
    pub(crate) async fn entries(
        &self,
        txn_id: TxnId,
    ) -> TCResult<TxnMapLockIter<PathSegment, DirEntry<T>>> {
        self.state.entries(txn_id).await
    }
}

#[cfg(feature = "service")]
#[async_trait]
impl<T> Recover<CacheBlock> for Cluster<T>
where
    T: Recover<CacheBlock, Txn = Txn> + Send + Sync,
{
    type Txn = Txn;

    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        self.state.recover(txn).await
    }
}

#[async_trait]
impl<T> Transact for Cluster<T>
where
    T: Transact + Send + Sync,
{
    type Commit = T::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("Cluster::commit {}", self.path());

        self.replicas.commit(txn_id);
        self.state.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("Cluster::rollback");

        self.replicas.rollback(txn_id);
        self.state.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("Cluster::finalize");

        self.replicas.finalize(*txn_id);
        self.state.finalize(txn_id).await
    }
}

macro_rules! impl_persist {
    ($t:ty) => {
        #[async_trait]
        impl fs::Persist<CacheBlock> for Cluster<$t> {
            type Txn = Txn;
            type Schema = Schema;

            async fn create(
                txn_id: TxnId,
                schema: Self::Schema,
                store: fs::Dir<CacheBlock>,
            ) -> TCResult<Self> {
                <$t as fs::Persist<CacheBlock>>::create(txn_id, (), store)
                    .map_ok(|subject| Self::new(schema, subject))
                    .await
            }

            async fn load(
                txn_id: TxnId,
                schema: Self::Schema,
                store: fs::Dir<CacheBlock>,
            ) -> TCResult<Self> {
                <$t as fs::Persist<CacheBlock>>::load(txn_id, (), store)
                    .map_ok(|subject| Self::new(schema, subject))
                    .await
            }

            fn dir(&self) -> fs::Inner<CacheBlock> {
                <$t as fs::Persist<CacheBlock>>::dir(&self.state)
            }
        }

        #[async_trait]
        impl fs::Persist<CacheBlock> for Cluster<Dir<$t>> {
            type Txn = Txn;
            type Schema = Schema;

            async fn create(
                txn_id: TxnId,
                schema: Self::Schema,
                store: fs::Dir<CacheBlock>,
            ) -> TCResult<Self> {
                <Dir<$t> as fs::Persist<CacheBlock>>::create(txn_id, schema.clone(), store)
                    .map_ok(|subject| Self::new(schema, subject))
                    .await
            }

            async fn load(
                txn_id: TxnId,
                schema: Self::Schema,
                store: fs::Dir<CacheBlock>,
            ) -> TCResult<Self> {
                <Dir<$t> as fs::Persist<CacheBlock>>::load(txn_id, schema.clone(), store)
                    .map_ok(|subject| Self::new(schema, subject))
                    .await
            }

            fn dir(&self) -> fs::Inner<CacheBlock> {
                <Dir<$t> as fs::Persist<CacheBlock>>::dir(&self.state)
            }
        }
    };
}

impl_persist!(Class);
impl_persist!(Library);
#[cfg(feature = "service")]
impl_persist!(Service);

impl<T: fmt::Debug> fmt::Debug for Cluster<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cluster at {}", self.link())
    }
}
