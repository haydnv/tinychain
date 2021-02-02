use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Future;

use error::*;
use generic::{PathSegment, TCPath};
use value::Value;

use crate::state::State;
use crate::txn::Txn;

mod op_def;
mod state;

pub type GetFuture<'a> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
pub type GetHandler<'a> = Box<dyn FnOnce(&'a Txn, Value) -> GetFuture + Send + 'a>;

pub trait Handler<'a>: Send {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        None
    }
}

pub trait Route: Send + Sync {
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>;
}

#[async_trait]
pub trait Public {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State>;
}

#[async_trait]
impl<T: Route + fmt::Display> Public for T {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;
        if let Some(get_handler) = handler.get() {
            get_handler(txn, key).await
        } else {
            Err(TCError::method_not_allowed(format!(
                "{} {}",
                self,
                TCPath::from(path)
            )))
        }
    }
}
