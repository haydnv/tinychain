//! A [`ChainBlock`], the block type of a [`super::Chain`]

use std::collections::btree_map::{BTreeMap, Entry};
use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::TryFutureExt;

use tc_transact::fs::BlockData;
use tc_transact::lock::Mutate;
use tc_transact::TxnId;
use tcgeneric::TCPathBuf;

use crate::scalar::{Scalar, Value};

/// A single filesystem block belonging to a [`super::Chain`].
#[derive(Clone, Eq, PartialEq)]
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
impl<'en> BlockData<'en> for ChainBlock {
    fn ext() -> &'static str {
        super::EXT
    }
}

#[async_trait]
impl de::FromStream for ChainBlock {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(context, decoder)
            .map_ok(|(hash, contents)| Self { hash, contents })
            .map_err(|e| de::Error::custom(format!("failed to decode ChainBlock: {}", e)))
            .await
    }
}

impl<'en> en::IntoStream<'en> for ChainBlock {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((self.hash, self.contents), encoder)
    }
}

impl<'en> en::ToStream<'en> for ChainBlock {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.hash, &self.contents), encoder)
    }
}

impl fmt::Debug for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("(chain block)")
    }
}
