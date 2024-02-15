use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{Future, TryFutureExt};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use rjwt::{OsRng, SigningKey, VerifyingKey};
use umask::Mode;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::{TxnLock, TxnSetLock, TxnSetLockIter};
use tc_transact::{Transact, TxnId};
use tc_value::{Host, Link, Value};
use tcgeneric::{PathSegment, TCPathBuf, ThreadSafe};

use crate::txn::Txn;
use crate::Actor;

pub const DEFAULT_UMASK: Mode = Mode::new()
    .with_class_perm(umask::USER, umask::ALL)
    .with_class_perm(umask::GROUP, umask::ALL)
    .with_class_perm(umask::OTHERS, umask::READ);

mod public;

#[derive(Debug)]
pub(crate) struct Key(Actor);

impl Deref for Key {
    type Target = Actor;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Eq for Key {}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.0.id() == other.0.id()
    }
}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id().hash(state)
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .public_key()
            .as_bytes()
            .cmp(other.0.public_key().as_bytes())
    }
}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone)]
pub struct Schema {
    lead: Link,
    owner: Option<Link>,
    group: Option<Link>,
}

impl Schema {
    pub fn new<Lead: Into<Link>>(lead: Lead, owner: Option<Link>, group: Option<Link>) -> Self {
        Self {
            lead: lead.into(),
            owner,
            group,
        }
    }
}

pub struct Cluster<T> {
    schema: Schema,
    actor: Actor,
    mode: Mode,
    subject: T,
    keyring: TxnSetLock<Key>,
    replicas: TxnLock<HashSet<Host>>,
}

impl<T> Cluster<T> {
    pub fn new(schema: Schema, subject: T, txn_id: TxnId) -> Self {
        let keypair = SigningKey::generate(&mut OsRng);
        let public_key = keypair.verifying_key();
        let actor_id = Value::Bytes(Arc::from(*public_key.as_bytes()));
        let actor = Actor::with_keypair(actor_id, keypair);
        let keyring = [Key(actor.clone())];

        let mode = if schema.owner.is_some() || schema.group.is_some() {
            DEFAULT_UMASK
        } else {
            Mode::all()
        };

        Self {
            schema,
            actor,
            mode,
            subject,
            keyring: TxnSetLock::new(txn_id, keyring),
            replicas: TxnLock::new(HashSet::new()),
        }
    }

    pub fn path(&self) -> &TCPathBuf {
        self.schema.lead.path()
    }

    pub fn umask(&self, path: &[PathSegment]) -> Mode {
        assert!(path.is_empty(), "TODO: cluster subject umask");
        self.mode
    }

    #[inline]
    pub fn claim<State, FE>(&self, txn: Txn<State, FE>) -> TCResult<Txn<State, FE>> {
        if txn.leader(self.path()).is_none() {
            txn.claim(&self.actor, self.path())
        } else {
            Ok(txn)
        }
    }

    pub fn keyring(&self, txn_id: TxnId) -> TCResult<TxnSetLockIter<Key>> {
        self.keyring.try_iter(txn_id).map_err(TCError::from)
    }

    #[inline]
    pub fn public_key(&self) -> VerifyingKey {
        self.actor.public_key()
    }

    pub(crate) fn subject(&self) -> &T {
        &self.subject
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

        let mut writes: FuturesUnordered<_> = replicas
            .iter()
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

    async fn replicate_commit(&self, _txn_id: TxnId) -> TCResult<()> {
        // TODO: validate that the commit message came from the txn leader, send commit messages to replicas, log errors, crash if >= 50% fail
        Err(not_implemented!("Cluster::replicate_commit"))
    }

    async fn replicate_rollback(&self, _txn_id: TxnId) -> TCResult<()> {
        // TODO: validate that the rollback message came from the txn leader, send rollback messages to replicas, log errors, crash if >= 50% fail
        Err(not_implemented!("Cluster::replicate_rollback"))
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
        self.subject.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.replicas.rollback(txn_id);
        self.subject.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.replicas.finalize(*txn_id);
        self.subject.finalize(txn_id).await
    }
}

#[async_trait]
impl<State, FE, T> fs::Persist<FE> for Cluster<T>
where
    State: ThreadSafe,
    FE: ThreadSafe + Clone,
    T: fs::Persist<FE, Schema = (), Txn = Txn<State, FE>>,
{
    type Txn = Txn<State, FE>;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        <T as fs::Persist<FE>>::create(txn_id, (), store)
            .map_ok(|subject| Self::new(schema, subject, txn_id))
            .await
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        <T as fs::Persist<FE>>::load(txn_id, (), store)
            .map_ok(|subject| Self::new(schema, subject, txn_id))
            .await
    }

    fn dir(&self) -> fs::Inner<FE> {
        <T as fs::Persist<FE>>::dir(&self.subject)
    }
}
