use std::collections::HashSet;

use async_trait::async_trait;
use futures::future::{Future, TryFutureExt};
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use rjwt::VerifyingKey;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::TxnLock;
use tc_transact::{Transact, TxnId};
use tc_value::{Host, Link};
use tcgeneric::{PathSegment, TCPathBuf, ThreadSafe};

use crate::txn::Txn;
use crate::Actor;

mod public;

pub struct Cluster<T> {
    actor: Actor,
    path: TCPathBuf,
    subject: T,
    replicas: TxnLock<HashSet<Host>>,
}

impl<T> Cluster<T> {
    pub fn new<Path: Into<TCPathBuf>>(path: Path, subject: T) -> Self {
        Self {
            actor: Actor::new(TCPathBuf::default().into()),
            path: path.into(),
            subject,
            replicas: TxnLock::new(HashSet::new()),
        }
    }

    pub fn path(&self) -> &TCPathBuf {
        &self.path
    }

    pub fn public_key(&self) -> &VerifyingKey {
        self.actor.public_key()
    }

    pub async fn replicate_write<Write, Fut>(
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

    pub(crate) fn subject(&self) -> &T {
        &self.subject
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
    T: fs::Persist<FE, Txn = Txn<FE>>,
{
    type Txn = Txn<FE>;
    type Schema = (TCPathBuf, T::Schema);

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let (path, schema) = schema;

        <T as fs::Persist<FE>>::create(txn_id, schema, store)
            .map_ok(|subject| Self::new(path, subject))
            .await
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let (path, schema) = schema;

        <T as fs::Persist<FE>>::load(txn_id, schema, store)
            .map_ok(|subject| Self::new(path, subject))
            .await
    }

    fn dir(&self) -> fs::Inner<FE> {
        <T as fs::Persist<FE>>::dir(&self.subject)
    }
}
