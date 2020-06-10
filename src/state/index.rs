use std::sync::Arc;

use crate::error;
use crate::internal::File;
use crate::value::TCResult;
use crate::transaction::TxnId;

pub struct Index {
    file: Arc<File>,
}

impl Index {
    async fn new(txn_id: &TxnId, file: Arc<File>) -> TCResult<Index> {
        if file.is_empty(txn_id).await {
            Ok(Index { file })
        } else {
            Err(error::bad_request("Tried to create a new Index without a new File", file))
        }
    }
}
