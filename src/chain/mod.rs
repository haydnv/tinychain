use std::sync::Arc;

use async_trait::async_trait;
use futures::join;

use crate::block::BlockData;
use crate::block::File;
use crate::class::TCResult;
use crate::collection::CollectionBase;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};

mod block;

pub type ChainBlock = block::ChainBlock;

#[derive(Clone)]
pub struct Chain {
    file: Arc<File<ChainBlock>>,
    collection: CollectionBase,
    latest_block: TxnLock<Mutable<u64>>,
}

impl Chain {
    pub async fn create(txn: Arc<Txn>, collection: CollectionBase) -> TCResult<Chain> {
        let file = txn.context().await?;
        let latest_block = TxnLock::new(format!("Chain: {}", &collection), 0.into());
        Ok(Chain {
            file,
            collection,
            latest_block,
        })
    }
}

#[async_trait]
impl Transact for Chain {
    async fn commit(&self, txn_id: &TxnId) {
        self.collection.commit(txn_id).await;
        // don't commit the Chain until the actual changes are committed, for crash recovery
        join!(self.file.commit(txn_id), self.latest_block.commit(txn_id));
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.collection.rollback(txn_id).await;
        join!(
            self.file.rollback(txn_id),
            self.latest_block.rollback(txn_id)
        );
    }
}
