use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;
use crate::transaction::Transaction;

struct StringContext {}

impl StringContext {
    fn new() -> Arc<StringContext> {
        Arc::new(StringContext {})
    }
}

#[async_trait]
impl TCContext for StringContext {
    async fn get(self: Arc<Self>, _txn: Arc<Transaction>, path: Link) -> TCResult<Arc<TCState>> {
        if path.as_str() != "/new" {
            return Err(error::not_found(path));
        }

        Ok(TCState::from_string(String::new()))
    }
}

pub struct ValueContext {
    string_context: Arc<StringContext>,
}

impl ValueContext {
    pub fn new() -> Arc<ValueContext> {
        Arc::new(ValueContext {
            string_context: StringContext::new(),
        })
    }
}

#[async_trait]
impl TCContext for ValueContext {
    async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<Arc<TCState>> {
        let segments = path.segments();

        match segments[0] {
            "string" => Ok(self
                .string_context
                .clone()
                .get(txn, path.from("/string")?)
                .await?),
            _ => Err(error::not_found(path)),
        }
    }
}
