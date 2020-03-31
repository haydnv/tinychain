use std::sync::Arc;

use crate::context::{TCContext, TCResult, TCState, TCValue};
use crate::error;
use crate::state::chain::ChainContext;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Table {
    schema: TCValue,
}

impl TCContext for Table {}

pub struct TableContext {
    chain_context: Arc<ChainContext>,
}

impl TableContext {
    pub fn new(chain_context: Arc<ChainContext>) -> Arc<TableContext> {
        Arc::new(TableContext { chain_context })
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
