use async_trait::async_trait;
use bytes::Bytes;
use futures::future::TryFutureExt;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use sha2::{Digest, Sha256};

use tc_error::*;
use tc_value::{Number, Value};
use tcgeneric::{Id, Map, TCBoxTryStream, Tuple};

use crate::fs::Dir;
use crate::Transaction;

/// Defines a standard hash for a scalar TinyChain state.
#[async_trait]
pub trait Hash: Send + Sync + Sized {
    /// Compute the SHA256 hash of this state.
    async fn hash(&self) -> TCResult<Bytes>;

    /// Consume this state and compute its SHA256 hash.
    async fn hash_owned(self) -> TCResult<Bytes> {
        self.hash().await
    }

    /// Return the SHA256 hash of this state as a hexadecimal string.
    async fn hash_hex(&self) -> TCResult<String> {
        self.hash().map_ok(|hash| hex::encode(hash)).await
    }
}

#[async_trait]
impl<T> Hash for Map<T>
where
    T: Hash + 'static,
{
    async fn hash(&self) -> TCResult<Bytes> {
        let mut hasher = Sha256::default();
        for (id, state) in self.iter() {
            let (id_hash, state_hash) = try_join!(id.hash(), state.hash())?;
            hasher.update(id_hash);
            hasher.update(state_hash);
        }

        Ok(hasher.finalize().to_vec().into())
    }
}

#[async_trait]
impl<T> Hash for Tuple<T>
where
    T: Hash,
{
    async fn hash(&self) -> TCResult<Bytes> {
        let item_hashes = stream::iter(self.iter()).then(|item| item.hash());
        hash_stream(item_hashes).await
    }
}

#[async_trait]
impl<T> Hash for Vec<T>
where
    T: Hash,
{
    async fn hash(&self) -> TCResult<Bytes> {
        let item_hashes = stream::iter(self.iter()).then(|item| item.hash());
        hash_stream(item_hashes).await
    }
}

#[async_trait]
impl<T1, T2> Hash for (T1, T2)
where
    T1: Hash,
    T2: Hash,
{
    async fn hash(&self) -> TCResult<Bytes> {
        let mut hasher = Sha256::default();
        hasher.update(self.0.hash().await?);
        hasher.update(self.1.hash().await?);
        Ok(Bytes::from(hasher.finalize().to_vec()))
    }
}

/// Implement [`Hash`] for the given type using [`tbon::en::encode`].
#[macro_export]
macro_rules! hash_encode {
    ($ty:ty) => {
        #[async_trait]
        impl Hash for $ty {
            async fn hash(&self) -> TCResult<Bytes> {
                let mut chunks = tbon::en::encode(self).map_err(TCError::internal)?;
                let mut hasher = sha2::Sha256::default();
                while let Some(hash) = chunks.try_next().map_err(TCError::internal).await? {
                    sha2::Digest::update(&mut hasher, &hash);
                }

                Ok(sha2::Digest::finalize(hasher).to_vec().into())
            }
        }
    };
}

#[cfg(feature = "tensor")]
hash_encode!(afarray::Array);
hash_encode!(Id);
hash_encode!(Number);
hash_encode!(Value);
hash_encode!(u64);

/// Defines a standard hash for a mutable collection.
#[async_trait]
pub trait HashCollection<D: Dir> {
    type Item: Hash;
    type Txn: Transaction<D>;

    /// Compute the SHA256 hash of this state.
    async fn hash(&self, txn: &Self::Txn) -> TCResult<Bytes> {
        let items = self.hashable(txn).await?;
        let item_hashes = items
            .map_ok(|item| item.hash_owned())
            .try_buffered(num_cpus::get());

        hash_stream(item_hashes).await
    }

    /// Return the SHA256 hash of this state as a hexadecimal string.
    async fn hash_hex(&self, txn: &Self::Txn) -> TCResult<String> {
        self.hash(txn).map_ok(|hash| hex::encode(hash)).await
    }

    /// Return a stream of hashable items which this state comprises, in a consistent order.
    async fn hashable(&self, txn: &Self::Txn) -> TCResult<TCBoxTryStream<Self::Item>>;
}

async fn hash_stream<S: Stream<Item = TCResult<Bytes>> + Unpin>(mut items: S) -> TCResult<Bytes> {
    let mut hasher = Sha256::default();
    while let Some(hash) = items.try_next().await? {
        hasher.update(&hash);
    }

    Ok(Bytes::from(hasher.finalize().to_vec()))
}
