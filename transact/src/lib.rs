use async_trait::async_trait;
use std::convert::TryFrom;

use destream::de::Decoder;
use destream::en::Encoder;

use error::*;

pub mod fs;
mod id;
pub mod lock;

pub use id::TxnId;

#[async_trait]
pub trait FromStream: Sized {
    async fn from_stream<F: fs::FileEntry, T: Transaction<F>, D: Decoder>(
        txn: T,
        decoder: D,
    ) -> Result<Self, D::Error>;
}

#[async_trait]
pub trait IntoStream<'en>: Sized {
    async fn into_stream<F: fs::FileEntry, T: Transaction<F>, E: Encoder<'en>>(
        self,
        txn: T,
        encoder: E,
    ) -> Result<E::Ok, E::Error>;
}

#[async_trait]
pub trait Transact {
    async fn commit(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}

#[async_trait]
pub trait Transaction<E: fs::FileEntry>: Sized {
    type Subcontext: Transaction<E>;

    fn id(&'_ self) -> &'_ TxnId;

    async fn context<B: fs::BlockData>(&self) -> TCResult<fs::File<B>>
    where
        E: From<fs::File<B>>,
        fs::File<B>: TryFrom<E>;

    async fn subcontext(&self) -> TCResult<Self::Subcontext>;
}
