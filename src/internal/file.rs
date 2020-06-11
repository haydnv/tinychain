use std::collections::{HashMap, HashSet};
use std::fmt;
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

    fn contains_block(&self, txn_id: &TxnId, block_id: &BlockId) -> bool {
        if let Some(txn_data) = self.txn_cache.get(txn_id) {
            if txn_data.get(block_id).is_some() {
                println!("File::contains_block {}", block_id);
                return true;
            }
        } else if self.blocks.get(block_id).is_some() {
            println!("File::contains_block {}", block_id);
            return true;
        }

        println!("File::contains_block {} FALSE", block_id);
        false
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
        Arc::new(File {
            state: Mutex::new(FileState::new()),
        })
    }

    pub async fn block_ids(&self, txn_id: &TxnId) -> HashSet<BlockId> {
        let mut block_ids = HashSet::new();
        let state = self.state.lock().await;
        block_ids.extend(state.blocks.keys().cloned());
        if let Some(blocks) = state.txn_cache.get(txn_id) {
            block_ids.extend(blocks.keys().cloned());
        }

        block_ids
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> bool {
        let state = self.state.lock().await;
        if let Some(txn_data) = state.txn_cache.get(txn_id) {
            txn_data.is_empty() && state.blocks.is_empty()
        } else {
            state.blocks.is_empty()
        }
    }

    pub async fn contains_block(&self, txn_id: &TxnId, block_id: &BlockId) -> bool {
        self.state.lock().await.contains_block(txn_id, block_id)
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

    pub async fn new_block(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
        initial_value: Bytes,
    ) -> TCResult<()> {
        println!("File::new_block {} {}", &txn_id, &block_id);
        let mut state = self.state.lock().await;

        if state.contains_block(&txn_id, &block_id) {
            return Err(error::bad_request(
                "Tried to create a block that already exists",
                block_id,
            ));
        }

        let mut block = BytesMut::with_capacity(initial_value.len());
        block.put(initial_value);

        state
            .txn_cache
            .entry(txn_id)
            .or_insert_with(HashMap::new)
            .insert(block_id, block);

        Ok(())
    }

    pub async fn put_block(
        &self,
        txn_id: TxnId,
        block_id: BlockId,
        block: BytesMut,
    ) -> TCResult<()> {
        let mut state = self.state.lock().await;

        if !state.contains_block(&txn_id, &block_id) {
            return Err(error::bad_request(
                "Tried to overwrite a nonexistent block",
                block_id,
            ));
        }

        state
            .txn_cache
            .entry(txn_id)
            .or_insert_with(HashMap::new)
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

impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "(file)")
    }
}
