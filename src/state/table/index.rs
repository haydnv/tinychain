use std::sync::Arc;

use crate::state::btree::{self, BTree, BTreeRange, Key};
use crate::transaction::TxnId;
use crate::value::{TCResult, TCStream, Value};

use super::Schema;

struct Index {
    btree: Arc<BTree>,
    schema: Schema,
}

impl Index {
    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.btree.is_empty(txn_id).await
    }

    pub async fn len(&self, txn_id: TxnId) -> TCResult<u64> {
        self.btree.clone().len(txn_id, btree::Selector::all()).await
    }

    pub async fn contains(&self, txn_id: TxnId, key: Key) -> TCResult<bool> {
        Ok(self.btree.clone().len(txn_id, key.into()).await? > 0)
    }

    pub async fn reversed(
        &self,
        txn_id: TxnId,
        range: BTreeRange,
    ) -> TCResult<TCStream<Vec<Value>>> {
        self.btree
            .clone()
            .slice(txn_id, btree::Selector::reverse(range))
            .await
    }
}
