use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;

use crate::context::{TCExecutable, TCResult};
use crate::error;
use crate::state::TCState;
use crate::transaction::Transaction;
use crate::value::{Link, TCValue};

#[derive(Debug)]
pub struct ValueContext;

impl ValueContext {
    pub fn new() -> Arc<ValueContext> {
        Arc::new(ValueContext)
    }
}

#[async_trait]
impl TCExecutable for ValueContext {
    async fn post(self: &Arc<Self>, txn: Arc<Transaction>, method: &Link) -> TCResult<TCState> {
        if method.len() != 1 {
            return Err(error::bad_request(
                "ValueContext has no such method",
                method,
            ));
        }

        match method.as_str(0) {
            "string" => Ok(TCValue::r#String(txn.require("from")?.try_into()?).into()),
            _ => Err(error::bad_request(
                "ValueContext has no such method",
                method,
            )),
        }
    }
}
