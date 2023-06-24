use async_hash::generic_array::GenericArray;
use async_hash::{Output, Sha256};

use tcgeneric::{label, Label};

pub mod block;
mod history;
mod store;

pub const CHAIN: Label = label("chain");

const BLOCK_SIZE: usize = 1_000_000; // TODO: reduce to 4,096

#[inline]
pub(crate) fn null_hash() -> Output<Sha256> {
    GenericArray::default()
}
