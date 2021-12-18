use async_trait::async_trait;
use bytes::Bytes;
use destream::en;
use futures::future::TryFutureExt;
use futures::stream::TryStreamExt;
use sha2::{Digest, Sha256};

use tc_error::*;
use tcgeneric::TCBoxTryStream;

use crate::fs::Dir;
use crate::Transaction;

/// Defines a standard hash for a mutable collection.
#[async_trait]
pub trait HashCollection<'en, D: Dir> {
    type Item: en::IntoStream<'en> + Send + 'en;
    type Txn: Transaction<D>;

    /// Return the SHA256 hash of this state as a hexadecimal string.
    async fn hash_hex(&'en self, txn: &'en Self::Txn) -> TCResult<String> {
        self.hash(txn).map_ok(|hash| hex::encode(hash)).await
    }

    /// Compute the SHA256 hash of this state.
    async fn hash(&'en self, txn: &'en Self::Txn) -> TCResult<Bytes> {
        let mut data = self.hashable(txn).await?;

        let mut hasher = Sha256::default();
        while let Some(item) = data.try_next().await? {
            hash_chunks(&mut hasher, item).await?;
        }

        let digest = hasher.finalize();
        Ok(Bytes::from(digest.to_vec()))
    }

    /// Return a stream of hashable items which this state comprises, in a consistent order.
    async fn hashable(&'en self, txn: &'en Self::Txn) -> TCResult<TCBoxTryStream<'en, Self::Item>>;
}

async fn hash_chunks<'en, T: en::IntoStream<'en> + 'en>(
    hasher: &mut Sha256,
    data: T,
) -> TCResult<()> {
    let mut data = tbon::en::encode(data).map_err(TCError::internal)?;
    while let Some(chunk) = data.try_next().map_err(TCError::internal).await? {
        hasher.update(&chunk);
    }

    Ok(())
}
