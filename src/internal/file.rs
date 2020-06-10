use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{BufMut, Bytes, BytesMut};
use futures::lock::Mutex;

use crate::error;
use crate::transaction::{Transact, TxnId};
use crate::value::link::PathSegment;
use crate::value::TCResult;

pub type BlockId = PathSegment;

struct FileState {
    blocks: HashMap<BlockId, Bytes>,
    txn_cache: HashMap<TxnId, HashMap<BlockId, BytesMut>>,
}

impl FileState {
    fn new() -> FileState {
        FileState {
            blocks: HashMap::new(),
            txn_cache: HashMap::new(),
        }
    }

    async fn get_block(&self, txn_id: &TxnId, block_id: &BlockId) -> Option<Bytes> {
        if let Some(Some(block)) = self
            .txn_cache
            .get(txn_id)
            .map(|blocks| blocks.get(block_id))
        {
            Some(Bytes::copy_from_slice(&block[..]))
        } else if let Some(block) = self.blocks.get(block_id) {
            Some(Bytes::copy_from_slice(&block[..]))
        } else {
            None
        }
    }
}

pub struct File {
    state: Mutex<FileState>,
}

impl File {
    pub fn new() -> Arc<File> {
        Arc::new(File { state: Mutex::new(FileState::new()) })
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
                println!("File::contains_block {}", block_id);
                return true;
            }
        } else if state.blocks.get(block_id).is_some() {
            println!("File::contains_block {}", block_id);
            return true;
        }

        println!("File::contains_block {} FALSE", block_id);
        false
    }

    pub async fn get_block_owned(
        self: Arc<Self>,
        txn_id: TxnId,
        block_id: BlockId,
    ) -> Option<Bytes> {
        self.get_block(&txn_id, &block_id).await
    }

    pub async fn get_block(&self, txn_id: &TxnId, block_id: &BlockId) -> Option<Bytes> {
        self.state.lock().await.get_block(&txn_id, &block_id).await
    }

    pub async fn insert_into(
        &self,
        txn_id: &TxnId,
        block_id: BlockId,
        data: Bytes,
        offset: usize,
    ) -> TCResult<()> {
        let mut state = self.state.lock().await;
        let block = state
            .get_block(txn_id, &block_id)
            .await
            .ok_or(error::not_found(&block_id))?;
        let mut new_block = BytesMut::with_capacity(block.len() + data.len());
        new_block.put(&block[..offset]);
        new_block.put(data);
        new_block.put(&block[offset..]);

        if let Some(txn_data) = state.txn_cache.get_mut(txn_id) {
            txn_data.insert(block_id, new_block);
        } else {
            let mut txn_data = HashMap::new();
            txn_data.insert(block_id, new_block);
            state.txn_cache.insert(txn_id.clone(), txn_data);
        }

        Ok(())
    }

    pub async fn new_block(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
        initial_value: Bytes,
    ) -> TCResult<()> {
        println!("File::new_block {} {}", &txn_id, &block_id);
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
}

#[async_trait]
impl Transact for File {
    async fn commit(&self, txn_id: &TxnId) {
        let mut state = self.state.lock().await;
        if let Some(mut blocks) = state.txn_cache.remove(txn_id) {
            state
                .blocks
                .extend(blocks.drain().map(|(id, block)| (id, block.freeze())));
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.state.lock().await.txn_cache.remove(txn_id);
    }
}
