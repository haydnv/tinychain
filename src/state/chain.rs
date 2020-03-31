use std::sync::Arc;

use crate::context::TCContext;
use crate::state::block::BlockContext;

#[derive(Hash)]
pub struct Chain {}

impl TCContext for Chain {}

pub struct ChainContext {
    block_context: Arc<BlockContext>,
}

impl ChainContext {
    pub fn new(block_context: Arc<BlockContext>) -> Arc<ChainContext> {
        Arc::new(ChainContext { block_context })
    }
}

impl TCContext for ChainContext {}
