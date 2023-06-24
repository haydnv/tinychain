use std::collections::btree_map::{BTreeMap, Entry};
use std::fmt;
use std::iter::FromIterator;

use async_hash::{Digest, Hash, Output, Sha256};
use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::{future, TryFutureExt, TryStreamExt};
use get_size::GetSize;
use log::debug;

use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::TxnId;
use tc_value::Value;
use tcgeneric::Tuple;

#[derive(Clone, Eq, PartialEq)]
pub enum Mutation {
    Delete(Value),
    Put(Value, Scalar),
}

impl<'a, D: Digest> Hash<D> for &'a Mutation {
    fn hash(self) -> Output<D> {
        match self {
            Mutation::Delete(key) => Hash::<D>::hash(key),
            Mutation::Put(key, value) => Hash::<D>::hash((key, value)),
        }
    }
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
            Self::Delete(key) => (key,).into_stream(encoder),
            Self::Put(key, value) => (key, value).into_stream(encoder),
        }
    }
}

impl<'en> en::IntoStream<'en> for Mutation {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Delete(key) => (key,).into_stream(encoder),
            Self::Put(key, value) => (key, value).into_stream(encoder),
        }
    }
}

impl fmt::Debug for Mutation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Delete(key) => write!(f, "DELETE {:?}", key),
            Self::Put(key, value) => write!(f, "PUT {:?} <- {:?}", key, value),
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
        let key = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, Self::expecting()))?;

        match seq.next_element(()).await? {
            Some(value) => Ok(Mutation::Put(key, value)),
            None => Ok(Mutation::Delete(key)),
        }
    }
}

/// A single filesystem block belonging to a `Chain`.
#[derive(Clone, Eq, PartialEq)]
pub struct ChainBlock {
    last_hash: Bytes,
    pub mutations: BTreeMap<TxnId, Vec<Mutation>>,
}

impl GetSize for ChainBlock {
    fn get_size(&self) -> usize {
        let size = self
            .mutations
            .iter()
            .map(|txn| txn.get_size())
            .sum::<usize>();

        self.last_hash.len() + size
    }
}

impl ChainBlock {
    /// Compute the hash of a [`ChainBlock`] with the given contents
    pub fn hash<'a, M, R>(last_hash: &'a [u8], mutations: M) -> Output<Sha256>
    where
        M: IntoIterator<Item = (&'a TxnId, &'a R)> + 'a,
        &'a R: IntoIterator<Item = &'a Mutation> + 'a,
    {
        let mut hasher = Sha256::new();
        hasher.update(last_hash);

        let mut txn_hasher = Sha256::new();
        for (txn_id, mutations) in mutations {
            let mut mutation_hasher = Sha256::new();
            mutation_hasher.update(Hash::<Sha256>::hash(txn_id));

            for mutation in mutations {
                mutation_hasher.update(Hash::<Sha256>::hash(mutation));
            }

            txn_hasher.update(mutation_hasher.finalize());
        }

        hasher.update(txn_hasher.finalize());
        hasher.finalize()
    }

    /// Return a new, empty block.
    pub fn new<H: Into<Bytes>>(hash: H) -> Self {
        Self {
            last_hash: hash.into(),
            mutations: BTreeMap::new(),
        }
    }

    /// Return a new, empty block with an empty mutation list for the given `TxnId`.
    pub fn with_txn<H: Into<Bytes>>(hash: H, txn_id: TxnId) -> Self {
        let mut mutations = BTreeMap::new();
        mutations.insert(txn_id, Vec::new());

        Self {
            last_hash: hash.into(),
            mutations,
        }
    }

    /// Return a new, empty block with an the given mutation list for the given `TxnId`.
    pub fn with_mutations(hash: Bytes, mutations: BTreeMap<TxnId, Vec<Mutation>>) -> Self {
        Self {
            last_hash: hash,
            mutations,
        }
    }

    /// Append a [`Mutation`] to this [`ChainBlock`]
    pub(super) fn append(&mut self, txn_id: TxnId, mutation: Mutation) {
        match self.mutations.entry(txn_id) {
            Entry::Vacant(entry) => {
                entry.insert(vec![mutation]);
            }
            Entry::Occupied(mut entry) => {
                entry.get_mut().push(mutation);
            }
        }
    }

    /// Append a DELETE op to this `ChainBlock`
    pub fn append_delete(&mut self, txn_id: TxnId, key: Value) {
        self.append(txn_id, Mutation::Delete(key))
    }

    /// Append a PUT op to the this `ChainBlock`
    pub fn append_put(&mut self, txn_id: TxnId, key: Value, value: Scalar) {
        debug!("ChainBlock::append_put {} <- {:?}", key, value);
        self.append(txn_id, Mutation::Put(key, value))
    }

    /// The current hash of this block
    pub fn current_hash(&self) -> Output<Sha256> {
        Self::hash(&self.last_hash, &self.mutations)
    }

    /// The hash of the previous block in the chain
    pub fn last_hash(&self) -> &Bytes {
        &self.last_hash
    }

    /// The current size of this block
    // TODO: delete
    pub async fn size(&self) -> TCResult<usize> {
        let encoded = tbon::en::encode(self)
            .map_err(|cause| unexpected!("TBON encoding error").consume(cause))?;

        encoded
            .map_err(|cause| unexpected!("TBON encoding error").consume(cause))
            .try_fold(0, |size, chunk| future::ready(Ok(size + chunk.len())))
            .await
    }
}

#[async_trait]
impl de::FromStream for ChainBlock {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(context, decoder)
            .map_ok(|(hash, mutations)| Self {
                last_hash: hash,
                mutations,
            })
            .map_err(|e| de::Error::custom(format!("failed to decode ChainBlock: {}", e)))
            .await
    }
}

impl<'en> en::IntoStream<'en> for ChainBlock {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((self.last_hash, self.mutations), encoder)
    }
}

impl<'en> en::ToStream<'en> for ChainBlock {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.last_hash, &self.mutations), encoder)
    }
}

#[cfg(debug_assertions)]
impl fmt::Debug for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "chain block:")?;
        writeln!(f, "\thash: {}", hex::encode(&self.last_hash))?;
        writeln!(f, "\tentries: {}", self.mutations.len())?;

        for (txn_id, mutations) in &self.mutations {
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

#[cfg(not(debug_assertions))]
impl fmt::Debug for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "(a Chain block starting at hash {} with {} entries)",
            hex::encode(&self.last_hash),
            self.mutations.len()
        )
    }
}
