use std::sync::Arc;

use crate::context::TCContext;
use crate::drive::Drive;

#[derive(Hash)]
pub struct Block {}

impl TCContext for Block {}

pub struct BlockContext {
    drive: Arc<Drive>,
}

impl BlockContext {
    pub fn new(drive: Arc<Drive>) -> Arc<BlockContext> {
        Arc::new(BlockContext { drive })
    }
}

impl TCContext for BlockContext {}
