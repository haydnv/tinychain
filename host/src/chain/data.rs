//! A [`ChainBlock`], the block type of a [`super::Chain`]

use std::collections::btree_map::{BTreeMap, Entry};
use std::fmt;
use std::io;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::{future, TryFutureExt, TryStreamExt};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWrite};
use tokio_util::io::StreamReader;

use tc_error::*;
use tc_transact::fs::BlockData;
use tc_transact::lock::Mutate;
use tc_transact::TxnId;
use tcgeneric::TCPathBuf;

use crate::scalar::{Scalar, Value};

/// A single filesystem block belonging to a [`super::Chain`].
#[derive(Clone)]
pub struct ChainBlock {
    hash: Bytes,
    contents: BTreeMap<TxnId, Vec<(TCPathBuf, Value, Scalar)>>,
}

impl ChainBlock {
    /// Return a new, empty block.
    pub fn new<H: Into<Bytes>>(hash: H) -> Self {
        Self {
            hash: hash.into(),
            contents: BTreeMap::new(),
        }
    }

    /// Append an op to the contents of this `ChainBlock`.
    pub fn append(&mut self, txn_id: TxnId, path: TCPathBuf, key: Value, value: Scalar) {
        match self.contents.entry(txn_id) {
            Entry::Vacant(entry) => {
                entry.insert(vec![(path, key, value)]);
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().push((path, key, value));
            }
        }
    }

    /// The mutations listed in this `ChainBlock`.
    pub fn mutations(&self) -> &BTreeMap<TxnId, Vec<(TCPathBuf, Value, Scalar)>> {
        &self.contents
    }

    /// The hash of the previous block in the chain.
    pub fn last_hash(&self) -> &Bytes {
        &self.hash
    }
}

#[async_trait]
impl Mutate for ChainBlock {
    type Pending = Self;

    fn diverge(&self, _txn_id: &TxnId) -> Self::Pending {
        self.clone()
    }

    async fn converge(&mut self, new_value: Self::Pending) {
        *self = new_value;
    }
}

#[async_trait]
impl de::FromStream for ChainBlock {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(context, decoder)
            .map_ok(|(hash, contents)| Self { hash, contents })
            .await
    }
}

impl<'en> en::IntoStream<'en> for ChainBlock {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let hash = base64::encode(self.hash);
        en::IntoStream::into_stream((hash, self.contents), encoder)
    }
}

impl<'en> en::ToStream<'en> for ChainBlock {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        let hash = base64::encode(&self.hash);
        en::IntoStream::into_stream((hash, &self.contents), encoder)
    }
}

#[async_trait]
// TODO: replace destream_json with tbon
impl BlockData for ChainBlock {
    fn ext() -> &'static str {
        super::EXT
    }

    async fn hash(&self) -> TCResult<Bytes> {
        let mut data = destream_json::encode(self.clone()).map_err(TCError::internal)?;
        let mut hasher = Sha256::default();
        while let Some(chunk) = data.try_next().map_err(TCError::internal).await? {
            hasher.update(&chunk);
        }

        let digest = hasher.finalize();
        Ok(Bytes::from(digest.to_vec()))
    }

    async fn load<S: AsyncReadExt + Send + Unpin>(source: S) -> TCResult<Self> {
        destream_json::read_from((), source)
            .map_ok(|(hash, contents)| Self { hash, contents })
            .map_err(|e| TCError::internal(format!("ChainBlock corrupted! {}", e)))
            .await
    }

    async fn persist<W: AsyncWrite + Send + Unpin>(&self, sink: &mut W) -> TCResult<u64> {
        let encoded = destream_json::encode(self)
            .map_err(|e| TCError::internal(format!("unable to serialize ChainBlock: {}", e)))?;

        let mut reader = StreamReader::new(
            encoded
                .map_ok(Bytes::from)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e)),
        );

        tokio::io::copy(&mut reader, sink)
            .map_err(|e| TCError::bad_gateway(e))
            .await
    }

    async fn size(&self) -> TCResult<u64> {
        let encoded = destream_json::encode(self)
            .map_err(|e| TCError::internal(format!("unable to serialize ChainBlock: {}", e)))?;

        encoded
            .map_err(|e| TCError::bad_request("serialization error", e))
            .try_fold(0, |size, chunk| {
                future::ready(Ok(size + chunk.len() as u64))
            })
            .await
    }
}

impl fmt::Display for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("(chain block)")
    }
}
