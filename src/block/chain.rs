use std::convert::TryFrom;
use std::sync::Arc;

use bytes::Bytes;

use crate::class::TCResult;
use crate::error;
use crate::transaction::lock::{Mutable, TxnLock};

use super::file::File;
use super::BlockData;

#[derive(Clone)]
struct ChainBlock {}

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

pub struct Chain {
    file: Arc<File<ChainBlock>>,
    latest_block: TxnLock<Mutable<u64>>,
}
