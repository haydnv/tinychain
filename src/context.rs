use std::sync::Arc;

use crate::error;
use crate::transaction::Transaction;

pub type TCResult<T> = Result<T, error::TCError>;

pub trait TCContext {
    fn post(self: Arc<Self>, _method: String, _txn: Transaction) -> TCResult<()> {
        Err(error::method_not_allowed())
    }
}
