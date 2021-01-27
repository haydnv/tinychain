use std::collections::HashSet;

use async_trait::async_trait;

use error::*;
use generic::{Id, Map, PathSegment};

pub mod lock;
mod txn;

pub use txn::*;

#[async_trait]
pub trait Public {
    type Key;
    type State: Clone;

    async fn get(
        &self,
        txn: &Txn<Self::State>,
        path: &[PathSegment],
        key: Self::Key,
    ) -> TCResult<Self::State>;

    async fn put(
        &self,
        txn: &Txn<Self::State>,
        path: &[PathSegment],
        key: Self::Key,
        value: Self::State,
    ) -> TCResult<()>;

    async fn post(
        &self,
        txn: &Txn<Self::State>,
        path: &[PathSegment],
        params: Map<Self::State>,
    ) -> TCResult<Self::State>;
}

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
