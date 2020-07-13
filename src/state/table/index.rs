use std::sync::Arc;

use crate::state::btree::{BTree, Bounds, Key};
use crate::transaction::TxnId;
use crate::value::TCResult;

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
        self.btree.clone().len(txn_id, Bounds::none()).await
    }

    pub async fn contains(&self, txn_id: TxnId, key: Key) -> TCResult<bool> {
        Ok(self.btree.clone().len(txn_id, Bounds::Key(key)).await? > 0)
    }
}
