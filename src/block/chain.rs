use std::convert::TryFrom;
use std::sync::Arc;

use bytes::Bytes;

use crate::class::TCResult;
use crate::collection::Collect;
use crate::error;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::Txn;

use super::file::File;
use super::BlockData;

#[derive(Clone)]
pub struct ChainBlock {}

impl TryFrom<Bytes> for ChainBlock {
    type Error = error::TCError;

    fn try_from(_data: Bytes) -> TCResult<ChainBlock> {
        Err(error::not_implemented())
    }
}

impl From<ChainBlock> for Bytes {
    fn from(_block: ChainBlock) -> Bytes {
        unimplemented!()
    }
}

impl BlockData for ChainBlock {}

pub struct Chain<O: Collect> {
    file: Arc<File<ChainBlock>>,
    latest_block: TxnLock<Mutable<u64>>,
    object: O,
}

impl<O: Collect> Chain<O> {
    pub async fn create(_txn: Arc<Txn>, _object: O) -> TCResult<Chain<O>> {
        Err(error::not_implemented())
    }
}
