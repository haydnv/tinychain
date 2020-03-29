use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error;
use crate::transaction::Transaction;

pub type TCResult<T> = Result<T, error::TCError>;

#[derive(Deserialize, Serialize)]
pub enum TCValue {}

pub trait TCContext {
    fn post(self: Arc<Self>, _method: String, _txn: Transaction) -> TCResult<()> {
        Err(error::method_not_allowed())
    }
}
