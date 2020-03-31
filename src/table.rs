use std::sync::Arc;

use crate::context::{TCContext, TCResult, TCState, TCValue};
use crate::error;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Table {
    schema: TCValue,
}

impl TCContext for Table {}

pub struct TableContext {}

impl TableContext {
    pub fn new() -> Arc<TableContext> {
        Arc::new(TableContext {})
    }
}

impl TCContext for TableContext {
    fn post(self: Arc<Self>, method: String, txn: Arc<Transaction>) -> TCResult<Arc<TCState>> {
        if method != "new" {
            return Err(error::not_found(method));
        }

        if let TCState::Value(schema) = &*txn.require("schema")? {
            Ok(Arc::new(TCState::Table(Arc::new(Table {
                schema: schema.clone(),
            }))))
        } else {
            Err(error::bad_request(
                "TableContext::new takes one parameter",
                "schema",
            ))
        }
    }
}
