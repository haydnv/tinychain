use std::convert::TryFrom;
use std::fmt;

use bytes::Bytes;

use crate::block::BlockData;
use crate::error;
use crate::TCResult;

#[derive(Clone)]
pub struct ChainBlock {}

impl TryFrom<Bytes> for ChainBlock {
    type Error = error::TCError;

    fn try_from(_data: Bytes) -> TCResult<ChainBlock> {
        Err(error::not_implemented("ChainBlock::try_from(Bytes)"))
    }
}

impl From<ChainBlock> for Bytes {
    fn from(_block: ChainBlock) -> Bytes {
        unimplemented!()
    }
}

impl BlockData for ChainBlock {
    fn size(&self) -> usize {
        0
    }
}

impl fmt::Display for ChainBlock {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChainBlock")
    }
}
