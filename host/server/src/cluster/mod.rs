use std::collections::HashSet;

use async_trait::async_trait;
use futures::future::{Future, TryFutureExt};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use rjwt::VerifyingKey;
use umask::Mode;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, TxnId};
use tc_value::{Host, Link};
use tcgeneric::{PathSegment, TCPathBuf, ThreadSafe};

use crate::txn::Txn;
use crate::Actor;

pub const CLUSTER_MODE: Mode = Mode::new()
    .with_class_perm(umask::USER, umask::ALL)
    .with_class_perm(umask::GROUP, umask::ALL)
    .with_class_perm(umask::OTHERS, umask::READ);

mod public;

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
    actor: Actor,
    schema: Schema,
    subject: T,
    replicas: TxnLock<HashSet<Host>>,
}

impl<T> Cluster<T> {
    pub fn new(schema: Schema, subject: T) -> Self {
        Self {
            actor: Actor::new(TCPathBuf::default().into()),
            schema,
            subject,
            replicas: TxnLock::new(HashSet::new()),
        }
    }

    pub fn path(&self) -> &TCPathBuf {
        self.schema.lead.path()
    }

    pub fn public_key(&self) -> &VerifyingKey {
        self.actor.public_key()
    }

    pub fn authorization<FE>(&self, path: &[PathSegment], txn: &Txn<FE>) -> Mode {
        assert!(path.is_empty(), "TODO: cluster subject umask");

        let mut mode = CLUSTER_MODE;

        if let Some(actor) = &self.schema.owner {
            if let Some(grant) = txn.mode(actor, &self.schema.lead) {
                mode &= grant;
            }
        }

        if let Some(actor) = &self.schema.group {
            if let Some(grant) = txn.mode(actor, &self.schema.lead) {
                mode &= grant;
            }
        }

        mode
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
impl<FE, T> fs::Persist<FE> for Cluster<T>
where
    FE: ThreadSafe + Clone,
    T: fs::Persist<FE, Schema = (), Txn = Txn<FE>>,
{
    type Txn = Txn<FE>;
    type Schema = Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        <T as fs::Persist<FE>>::create(txn_id, (), store)
            .map_ok(|subject| Self::new(schema, subject))
            .await
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        <T as fs::Persist<FE>>::load(txn_id, (), store)
            .map_ok(|subject| Self::new(schema, subject))
            .await
    }

    fn dir(&self) -> fs::Inner<FE> {
        <T as fs::Persist<FE>>::dir(&self.subject)
    }
}
