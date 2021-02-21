//! A [`ChainBlock`], the block type of a [`super::Chain`]

use std::convert::TryFrom;
use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::TryFutureExt;

use tc_error::*;
use tc_transact::fs::BlockData;
use tc_transact::lock::Mutate;
use tc_transact::TxnId;

use crate::scalar::OpRef;

/// A single filesystem block belonging to a [`super::Chain`].
#[derive(Clone)]
pub struct ChainBlock {
    hash: Bytes,
    contents: Vec<OpRef>,
}

impl ChainBlock {
    pub fn append(&mut self, op_ref: OpRef) {
        self.contents.push(op_ref);
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

impl TryFrom<Bytes> for ChainBlock {
    type Error = TCError;

    fn try_from(_bytes: Bytes) -> TCResult<Self> {
        unimplemented!()
    }
}

impl BlockData for ChainBlock {}

impl From<ChainBlock> for Bytes {
    fn from(_block: ChainBlock) -> Bytes {
        unimplemented!()
    }
}

impl fmt::Display for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("(chain block)")
    }
}
