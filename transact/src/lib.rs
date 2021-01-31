use std::sync::Arc;

use async_trait::async_trait;
use destream::en;

use error::*;
use generic::Id;

pub mod fs;
mod id;
pub mod lock;

pub use id::TxnId;

pub trait IntoView<'en, F: fs::FileEntry> {
    type Txn: Transaction<F>;
    type View: en::IntoStream<'en> + Sized;

    fn into_view(self, txn: Self::Txn) -> Self::View;
}

#[async_trait]
pub trait Transact {
    async fn commit(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}

#[async_trait]
pub trait Transaction<E: fs::FileEntry>: Sized {
    fn id(&'_ self) -> &'_ TxnId;

    async fn context(&'_ self) -> &'_ Arc<fs::Dir<E>>;

    async fn subcontext(&self, id: &Id) -> TCResult<Self>;
}
