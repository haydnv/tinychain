use std::convert::TryFrom;
use std::mem;

use bytes::Bytes;
use error::TCError;

use crate::chain::ChainBlock;

mod cache;

pub trait BlockData: TryFrom<Bytes, Error = TCError> {
    fn size(&self) -> usize {
        mem::size_of::<Self>()
    }
}

pub trait File<B: BlockData> {}

impl BlockData for ChainBlock {}
