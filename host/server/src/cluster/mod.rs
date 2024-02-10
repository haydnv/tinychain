use async_trait::async_trait;
use futures::TryFutureExt;

use tc_error::*;
use tc_transact::fs;
use tc_transact::TxnId;
use tcgeneric::ThreadSafe;

use crate::txn::Txn;

pub struct Cluster<T> {
    subject: T,
}

#[async_trait]
impl<FE, T> fs::Persist<FE> for Cluster<T>
where
    FE: ThreadSafe + Clone,
    T: fs::Persist<FE, Txn = Txn<FE>>,
{
    type Txn = Txn<FE>;
    type Schema = T::Schema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        <T as fs::Persist<FE>>::create(txn_id, schema, store)
            .map_ok(|subject| Self { subject })
            .await
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        <T as fs::Persist<FE>>::load(txn_id, schema, store)
            .map_ok(|subject| Self { subject })
            .await
    }

    fn dir(&self) -> fs::Inner<FE> {
        <T as fs::Persist<FE>>::dir(&self.subject)
    }
}
