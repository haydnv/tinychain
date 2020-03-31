use std::sync::Arc;

use async_trait::async_trait;

use crate::context::*;
use crate::error;

struct StringContext {}

impl StringContext {
    fn new() -> Arc<StringContext> {
        Arc::new(StringContext {})
    }
}

#[async_trait]
impl TCContext for StringContext {
    async fn get(self: Arc<Self>, path: Link) -> TCResult<Arc<TCState>> {
        let segments = path.segments();
        let segments: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();

        match segments[..] {
            ["new"] => Ok(TCState::from_string(String::new())),
            _ => Err(error::not_found(path)),
        }
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
    async fn get(self: Arc<Self>, path: Link) -> TCResult<Arc<TCState>> {
        let segments = path.segments();
        let segments: Vec<&str> = segments.iter().map(|s| s.as_str()).collect();

        match segments[0] {
            "string" => Ok(self
                .string_context
                .clone()
                .get(path.from("/string")?)
                .await?),
            _ => Err(error::not_found(path)),
        }
    }
}
