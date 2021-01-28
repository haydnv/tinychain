use std::convert::TryFrom;
use std::fmt;

use async_trait::async_trait;
use bytes::Bytes;

use error::*;
use fs::BlockData;
use transact::lock::Mutate;

use crate::state::scalar::OpRef;
use crate::TxnId;

#[derive(Clone)]
pub struct ChainBlock {
    order: u64,
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

impl BlockData for ChainBlock {}

impl TryFrom<Bytes> for ChainBlock {
    type Error = TCError;

    fn try_from(_data: Bytes) -> TCResult<Self> {
        unimplemented!()
    }
}

impl From<ChainBlock> for Bytes {
    fn from(_block: ChainBlock) -> Bytes {
        unimplemented!()
    }
}

impl fmt::Display for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChainBlock {}", self.order)
    }
}
