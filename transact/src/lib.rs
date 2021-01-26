use std::collections::HashSet;

use async_trait::async_trait;

use error::*;
use generic::Id;

pub mod lock;
mod txn;

pub use txn::*;

pub trait State: Clone + Sized + Send {
    fn is_ref(&self) -> bool;
}

#[async_trait]
pub trait Refer {
    type State: State;

    fn requires(&self, deps: &mut HashSet<Id>);

    async fn resolve(self, txn: &Txn<Self::State>) -> TCResult<Self::State>;
}

#[async_trait]
pub trait Transact {
    async fn commit(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}
