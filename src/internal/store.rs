use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use futures::lock::Mutex;

use crate::error;
use crate::transaction::TxnId;
use crate::value::link::PathSegment;
use crate::value::TCResult;

pub type BlockId = PathSegment;

struct StoreState {
    blocks: HashMap<BlockId, Bytes>,
    txn_cache: HashMap<TxnId, HashMap<BlockId, Bytes>>,
}

impl StoreState {
    fn new() -> StoreState {
        StoreState {
            blocks: HashMap::new(),
            txn_cache: HashMap::new(),
        }
    }
}

pub struct Store {
    state: Mutex<StoreState>,
}

impl Store {
    pub fn new() -> Arc<Store> {
        Arc::new(Store {
            state: Mutex::new(StoreState::new()),
        })
    }

    pub async fn exists(&self, txn_id: &TxnId, block_id: &BlockId) -> bool {
        let state = self.state.lock().await;
        if let Some(txn_data) = state.txn_cache.get(txn_id) {
            if txn_data.get(block_id).is_some() {
                return true;
            }
        } else if state.blocks.get(block_id).is_some() {
            return true;
        }

        false
    }

    pub async fn new_block(
        &self,
        txn_id: &TxnId,
        block_id: BlockId,
        _initial_value: Bytes,
    ) -> TCResult<()> {
        if self.exists(txn_id, &block_id).await {
            return Err(error::bad_request(
                "Tried to create a block that already exists",
                block_id,
            ));
        }

        Err(error::not_implemented())
    }
}
