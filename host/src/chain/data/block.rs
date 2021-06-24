use std::collections::btree_map::{BTreeMap, Entry};
use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::TryFutureExt;
use log::debug;

use tc_transact::fs::BlockData;
use tc_transact::lock::Mutate;
use tc_transact::TxnId;
use tcgeneric::{TCPathBuf, Tuple};

use crate::chain::{BLOCK_SIZE, EXT};
use crate::scalar::{Scalar, Value};

#[derive(Clone, Eq, PartialEq)]
pub enum Mutation {
    Delete(TCPathBuf, Value),
    Put(TCPathBuf, Value, Scalar),
}

#[async_trait]
impl de::FromStream for Mutation {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(MutationVisitor).await
    }
}

impl<'en> en::ToStream<'en> for Mutation {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::IntoStream;

        match self {
            Self::Delete(path, key) => (path, key).into_stream(encoder),
            Self::Put(path, key, value) => (path, key, value).into_stream(encoder),
        }
    }
}

impl<'en> en::IntoStream<'en> for Mutation {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Delete(path, key) => (path, key).into_stream(encoder),
            Self::Put(path, key, value) => (path, key, value).into_stream(encoder),
        }
    }
}

impl fmt::Debug for Mutation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Delete(path, key) => write!(f, "DELETE {}: {:?}", path, key),
            Self::Put(path, key, value) => write!(f, "PUT {}: {:?} <- {:?}", path, key, value),
        }
    }
}

impl fmt::Display for Mutation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Delete(path, key) => write!(f, "DELETE {}: {}", path, key),
            Self::Put(path, key, value) => write!(f, "PUT {}: {} <- {}", path, key, value),
        }
    }
}

struct MutationVisitor;

#[async_trait]
impl de::Visitor for MutationVisitor {
    type Value = Mutation;

    fn expecting() -> &'static str {
        "a mutation record"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let path = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, Self::expecting()))?;

        let key = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, Self::expecting()))?;

        if let Some(value) = seq.next_element(()).await? {
            Ok(Mutation::Put(path, key, value))
        } else {
            Ok(Mutation::Delete(path, key))
        }
    }
}

/// A single filesystem block belonging to a `Chain`.
#[derive(Clone, Eq, PartialEq)]
pub struct ChainBlock {
    hash: Bytes,
    contents: BTreeMap<TxnId, Vec<Mutation>>,
}

impl ChainBlock {
    /// Return a new, empty block.
    pub fn new<H: Into<Bytes>>(hash: H) -> Self {
        Self {
            hash: hash.into(),
            contents: BTreeMap::new(),
        }
    }

    /// Return a new, empty block with an empty mutation list for the given `TxnId`.
    pub fn with_txn<H: Into<Bytes>>(hash: H, txn_id: TxnId) -> Self {
        let mut contents = BTreeMap::new();
        contents.insert(txn_id, Vec::new());

        Self {
            hash: hash.into(),
            contents,
        }
    }

    /// Return a new, empty block with an empty mutation list for the given `TxnId`.
    pub fn with_mutations(hash: Bytes, contents: BTreeMap<TxnId, Vec<Mutation>>) -> Self {
        Self { hash, contents }
    }

    pub fn append(&mut self, txn_id: TxnId, mutation: Mutation) {
        match self.contents.entry(txn_id) {
            Entry::Vacant(entry) => {
                entry.insert(vec![mutation]);
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().push(mutation);
            }
        }
    }

    /// Append a DELETE op to the contents of this `ChainBlock`.
    pub fn append_delete(&mut self, txn_id: TxnId, path: TCPathBuf, key: Value) {
        self.append(txn_id, Mutation::Delete(path, key))
    }

    /// Append a PUT op to the contents of this `ChainBlock`.
    pub fn append_put(&mut self, txn_id: TxnId, path: TCPathBuf, key: Value, value: Scalar) {
        debug!("ChainBlock::append_put {}: {} <- {}", path, key, value);
        self.append(txn_id, Mutation::Put(path, key, value))
    }

    /// Delete all mutations listed in this `ChainBlock` prior to the given `TxnId`.
    pub fn clear_until(&mut self, txn_id: &TxnId) {
        let old_txn_ids: Vec<TxnId> = self
            .contents
            .keys()
            .filter(|k| k < &txn_id)
            .cloned()
            .collect();

        for old_txn_id in old_txn_ids.into_iter() {
            self.contents.remove(&old_txn_id);
        }
    }

    /// The mutations listed in this `ChainBlock`.
    pub fn mutations(&self) -> &BTreeMap<TxnId, Vec<Mutation>> {
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
impl BlockData for ChainBlock {
    fn ext() -> &'static str {
        EXT
    }

    fn max_size() -> u64 {
        BLOCK_SIZE
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
        writeln!(f, "chain block:")?;
        writeln!(f, "\thash: {}", hex::encode(&self.hash))?;
        writeln!(f, "\tentries: {}", self.contents.len())?;
        for (txn_id, mutations) in &self.contents {
            writeln!(
                f,
                "\t\t{}: {:?}",
                txn_id,
                Tuple::<&Mutation>::from_iter(mutations)
            )?;
        }

        Ok(())
    }
}

impl fmt::Display for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "(a Chain block starting at hash {} with {} entries)",
            hex::encode(&self.hash),
            self.contents.len()
        )
    }
}
