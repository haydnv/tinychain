use async_trait::async_trait;
use futures::TryFutureExt;

use tc_error::*;
use tc_transact::fs;
use tc_transact::lock::TxnSetLock;
use tc_transact::{Transact, TxnId};
use tc_value::Host as LinkHost;
use tcgeneric::{TCPathBuf, ThreadSafe};

use crate::txn::Txn;

pub struct Cluster<T> {
    path: TCPathBuf,
    subject: T,
    replicas: TxnSetLock<LinkHost>,
}

impl<T> Cluster<T> {
    pub fn new<Path: Into<TCPathBuf>>(path: Path, subject: T, txn_id: TxnId) -> Self {
        Self {
            path: path.into(),
            subject,
            replicas: TxnSetLock::new(txn_id),
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
            .map_ok(|subject| Self::new(path, subject, txn_id))
            .await
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let (path, schema) = schema;

        <T as fs::Persist<FE>>::load(txn_id, schema, store)
            .map_ok(|subject| Self::new(path, subject, txn_id))
            .await
    }

    fn dir(&self) -> fs::Inner<FE> {
        <T as fs::Persist<FE>>::dir(&self.subject)
    }
}
