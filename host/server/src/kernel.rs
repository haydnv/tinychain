use std::marker::PhantomData;

use async_trait::async_trait;

use tc_error::*;
use tc_transact::{fs, Transact, TxnId};
use tcgeneric::ThreadSafe;

use crate::cluster::Cluster;
use crate::txn::{Hypothetical, Txn};

pub struct Kernel<FE> {
    hypothetical: Cluster<Hypothetical>,
    file: PhantomData<FE>,
}

impl<FE> Kernel<FE> {
    pub(crate) async fn commit(&self, txn_id: TxnId) {
        self.hypothetical.rollback(&txn_id).await;
    }

    pub(crate) async fn finalize(&self, txn_id: TxnId) {
        self.hypothetical.finalize(&txn_id).await;
    }
}

#[async_trait]
impl<FE: ThreadSafe + Clone> fs::Persist<FE> for Kernel<FE> {
    type Schema = ();
    type Txn = Txn<FE>;

    async fn create(txn_id: TxnId, _schema: Self::Schema, _store: fs::Dir<FE>) -> TCResult<Self> {
        Ok(Self {
            hypothetical: Cluster::new(Hypothetical::PATH, Hypothetical::new(), txn_id),
            file: PhantomData,
        })
    }

    async fn load(txn_id: TxnId, _schema: Self::Schema, _store: fs::Dir<FE>) -> TCResult<Self> {
        Ok(Self {
            hypothetical: Cluster::new(Hypothetical::PATH, Hypothetical::new(), txn_id),
            file: PhantomData,
        })
    }

    fn dir(&self) -> fs::Inner<FE> {
        unimplemented!("Kernel::inner")
    }
}
