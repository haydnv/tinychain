use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Future;

use tc_error::*;
use tcgeneric::{Map, PathSegment, TCPath};

use crate::scalar::Value;
use crate::state::State;
use crate::txn::Txn;

mod chain;
mod cluster;
mod collection;
mod generic;
mod object;
mod scalar;
mod state;

pub type GetFuture<'a> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
pub type GetHandler<'a> = Box<dyn FnOnce(Txn, Value) -> GetFuture<'a> + Send + 'a>;

pub type PutFuture<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;
pub type PutHandler<'a> = Box<dyn FnOnce(Txn, Value, State) -> PutFuture<'a> + Send + 'a>;

pub type PostFuture<'a> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
pub type PostHandler<'a> = Box<dyn FnOnce(Txn, Map<State>) -> PostFuture<'a> + Send + 'a>;

pub type DeleteFuture<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;
pub type DeleteHandler<'a> = Box<dyn FnOnce(Txn, Value) -> DeleteFuture<'a> + Send + 'a>;

pub trait Handler<'a>: Send {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        None
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        None
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        None
    }

    fn delete(self: Box<Self>) -> Option<DeleteHandler<'a>> {
        None
    }
}

pub trait Route: Send + Sync {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>;
}

#[async_trait]
pub trait Public {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State>;

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()>;

    async fn post(&self, txn: &Txn, path: &[PathSegment], params: Map<State>) -> TCResult<State>;

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()>;
}

#[async_trait]
impl<T: Route + fmt::Display> Public for T {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(get_handler) = handler.get() {
            get_handler(txn.clone(), key).await
        } else {
            Err(TCError::method_not_allowed(format!(
                "{} {}",
                self,
                TCPath::from(path)
            )))
        }
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(put_handler) = handler.put() {
            put_handler(txn.clone(), key, value).await
        } else {
            Err(TCError::method_not_allowed(format!(
                "{} {}",
                self,
                TCPath::from(path)
            )))
        }
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], params: Map<State>) -> TCResult<State> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(post_handler) = handler.post() {
            post_handler(txn.clone(), params).await
        } else {
            Err(TCError::method_not_allowed(format!(
                "{} {}",
                self,
                TCPath::from(path)
            )))
        }
    }

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(delete_handler) = handler.delete() {
            delete_handler(txn.clone(), key).await
        } else {
            Err(TCError::method_not_allowed(format!(
                "{} {}",
                self,
                TCPath::from(path)
            )))
        }
    }
}
