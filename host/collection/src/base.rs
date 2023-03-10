use std::marker::PhantomData;

use async_trait::async_trait;

use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, Persist, Restore};
use tc_transact::{Transaction, TxnId};
use tcgeneric::ThreadSafe;

use super::{Collection, Schema};

#[derive(Clone)]
pub struct CollectionBase<T, FE> {
    phantom: PhantomData<(T, FE)>,
}

#[async_trait]
impl<T: Transaction<FE>, FE: ThreadSafe> Persist<FE> for CollectionBase<T, FE> {
    type Txn = T;
    type Schema = Schema;

    async fn create(_txn_id: TxnId, _schema: Schema, _store: Dir<FE>) -> TCResult<Self> {
        todo!()
    }

    async fn load(_txn_id: TxnId, _schema: Schema, _store: Dir<FE>) -> TCResult<Self> {
        todo!()
    }

    fn dir(&self) -> &tc_transact::fs::Inner<FE> {
        todo!()
    }
}

#[async_trait]
impl<T, FE> CopyFrom<FE, Collection<T, FE>> for CollectionBase<T, FE>
where
    T: Transaction<FE>,
    FE: ThreadSafe,
{
    async fn copy_from(_txn: &T, _store: Dir<FE>, _instance: Collection<T, FE>) -> TCResult<Self> {
        todo!()
    }
}

#[async_trait]
impl<T: Transaction<FE>, FE: ThreadSafe> Restore<FE> for CollectionBase<T, FE> {
    async fn restore(&self, _txn_id: TxnId, _backup: &Self) -> TCResult<()> {
        todo!()
    }
}
