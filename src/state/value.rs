use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::dir::Dir;
use crate::error;
use crate::transaction::Transaction;

struct StringContext {}

#[async_trait]
impl TCExecutable for StringContext {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: Link) -> TCResult<Arc<TCState>> {
        match method.as_str() {
            "/new" => Ok(TCState::from_value(TCValue::from_string(""))),
            "/from" => {
                let source = txn.require("value")?;
                source.clone().to_value()?.to_string()?; // Return an error if it's not a string
                Ok(source)
            }
            _ => Err(error::bad_request(
                "StringContext has no such method",
                method,
            )),
        }
    }
}

pub struct ValueContext {}

impl ValueContext {
    pub fn init() -> TCResult<Arc<Dir>> {
        let dir = Dir::new();
        dir.clone()
            .put_exe(Link::to("/string")?, Arc::new(StringContext {}));
        Ok(dir)
    }
}
