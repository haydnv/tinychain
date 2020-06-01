use std::collections::HashMap;
use std::sync::Arc;

use bytes::{BufMut, Bytes, BytesMut};
use futures::lock::Mutex;

use crate::error;
use crate::transaction::TxnId;
use crate::value::link::PathSegment;
use crate::value::TCResult;

pub type BlockId = PathSegment;

struct StoreState {
    blocks: HashMap<BlockId, Bytes>,
    txn_cache: HashMap<TxnId, HashMap<BlockId, BytesMut>>,
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

    pub async fn append(&self, txn_id: &TxnId, block_id: &BlockId, data: Bytes) -> TCResult<()> {
        let mut state = self.state.lock().await;
        if let Some(block) = state
            .txn_cache
            .entry(txn_id.clone())
            .or_insert(HashMap::new())
            .get_mut(block_id)
        {
            block.put(data);
            Ok(())
        } else if let Some(block) = state.blocks.get(block_id) {
            let mut block_txn_copy = BytesMut::from(&block[..]);
            block_txn_copy.put(data);
            state
                .txn_cache
                .get_mut(txn_id)
                .unwrap()
                .insert(block_id.clone(), block_txn_copy);
            Ok(())
        } else {
            Err(error::not_found(block_id))
        }
    }

    pub async fn contains_block(&self, txn_id: &TxnId, block_id: &BlockId) -> bool {
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
        txn_id: TxnId,
        block_id: BlockId,
        initial_value: Bytes,
    ) -> TCResult<()> {
        if self.contains_block(&txn_id, &block_id).await {
            return Err(error::bad_request(
                "Tried to create a block that already exists",
                block_id,
            ));
        }

        let mut block = BytesMut::with_capacity(initial_value.len());
        block.put(initial_value);

        self.state
            .lock()
            .await
            .txn_cache
            .entry(txn_id)
            .or_insert(HashMap::new())
            .insert(block_id, block);

        Ok(())
    }

    pub async fn get_block(&'_ self, txn_id: &'_ TxnId, block_id: &'_ BlockId) -> TCResult<Bytes> {
        let state = self.state.lock().await;
        if let Some(Some(block)) = state
            .txn_cache
            .get(txn_id)
            .map(|blocks| blocks.get(block_id))
        {
            Ok(Bytes::copy_from_slice(&block[..]))
        } else if let Some(block) = state.blocks.get(block_id) {
            Ok(Bytes::copy_from_slice(&block[..]))
        } else {
            Err(error::not_found(block_id))
        }
    }
}
