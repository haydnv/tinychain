use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::transaction::Transaction;

pub type TCResult<T> = Result<T, error::TCError>;

#[derive(Deserialize, Serialize)]
pub struct TCOp {
    method: String,

    #[serde(default="HashMap::new")]
    args: HashMap<String, TCValue>,
}

#[derive(Deserialize, Serialize, Hash)]
pub enum TCValue {
    Int32(i32),
    r#String(String),
    Vector(Vec<TCValue>),
}

#[derive(Deserialize, Serialize)]
pub enum TCRequest {
    Op(String, TCOp),
    Value(TCValue),
}

#[derive(Hash)]
pub enum TCState {
    Value(TCValue),
}

#[async_trait]
pub trait TCContext: Send + Sync {
    async fn get(self: Arc<Self>, _path: String) -> TCResult<TCState> {
        Err(error::method_not_allowed())
    }

    fn post(
        self: Arc<Self>,
        _method: String,
        _args: HashMap<String, TCValue>,
    ) -> TCResult<Arc<Transaction>> {
        Err(error::method_not_allowed())
    }
}
