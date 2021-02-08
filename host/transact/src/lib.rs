use async_trait::async_trait;
use destream::en;
use futures_locks::RwLock;

use error::*;
use generic::Id;

mod id;

pub mod fs;
pub mod lock;

pub use id::TxnId;

pub trait IntoView<'en, D: fs::Dir> {
    type Txn: Transaction<D>;
    type View: en::IntoStream<'en> + Sized;

    fn into_view(self, txn: Self::Txn) -> Self::View;
}

#[async_trait]
pub trait Transact {
    async fn commit(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}

#[async_trait]
pub trait Transaction<D: fs::Dir>: Sized {
    fn id(&'_ self) -> &'_ TxnId;

    fn context(&'_ self) -> &'_ RwLock<D>;

    async fn subcontext(&self, id: Id) -> TCResult<Self>;
}
