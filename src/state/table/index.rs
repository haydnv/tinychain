use std::sync::Arc;

use crate::state::btree::BTree;
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
}
