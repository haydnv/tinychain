use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Future;

use tc_error::*;
use tc_value::Value;
use tcgeneric::{Map, PathSegment, TCPath, ThreadSafe};

use super::Transaction;

pub mod generic;
pub mod helpers;

pub type GetFuture<'a, State> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
pub type GetHandler<'a, 'b, Txn, State> =
    Box<dyn FnOnce(&'b Txn, Value) -> GetFuture<'a, State> + Send + 'a>;

pub type PutFuture<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;
pub type PutHandler<'a, 'b, Txn, State> =
    Box<dyn FnOnce(&'b Txn, Value, State) -> PutFuture<'a> + Send + 'a>;

pub type PostFuture<'a, State> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
pub type PostHandler<'a, 'b, Txn, State> =
    Box<dyn FnOnce(&'b Txn, Map<State>) -> PostFuture<'a, State> + Send + 'a>;

pub type DeleteFuture<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;
pub type DeleteHandler<'a, 'b, Txn> =
    Box<dyn FnOnce(&'b Txn, Value) -> DeleteFuture<'a> + Send + 'a>;

#[derive(Debug)]
pub enum HandlerType {
    Get,
    Put,
    Post,
    Delete,
}

pub trait StateInstance: From<Value> + ThreadSafe + fmt::Debug {}

#[async_trait]
pub trait Handler<'a, Txn, State>: Send {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        None
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        None
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        None
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        None
    }
}

pub trait Route<FE>: Send + Sync {
    type Txn: Transaction<FE>;
    type State: StateInstance;

    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, Self::Txn, Self::State> + 'a>>;
}

#[async_trait]
pub trait Public<FE> {
    type Txn: Transaction<FE>;
    type State: StateInstance;

    async fn get(&self, txn: &Self::Txn, path: &[PathSegment], key: Value)
        -> TCResult<Self::State>;

    async fn put(
        &self,
        txn: &Self::Txn,
        path: &[PathSegment],
        key: Value,
        value: Self::State,
    ) -> TCResult<()>;

    async fn post(
        &self,
        txn: &Self::Txn,
        path: &[PathSegment],
        params: Map<Self::State>,
    ) -> TCResult<Self::State>;

    async fn delete(&self, txn: &Self::Txn, path: &[PathSegment], key: Value) -> TCResult<()>;
}

#[async_trait]
impl<FE, T: Route<FE> + fmt::Debug> Public<FE> for T {
    type Txn = T::Txn;
    type State = T::State;

    async fn get(
        &self,
        txn: &Self::Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<Self::State> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(get_handler) = handler.get() {
            get_handler(txn, key).await
        } else {
            Err(TCError::method_not_allowed(
                HandlerType::Get,
                self,
                TCPath::from(path),
            ))
        }
    }

    async fn put(
        &self,
        txn: &Self::Txn,
        path: &[PathSegment],
        key: Value,
        value: Self::State,
    ) -> TCResult<()> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(put_handler) = handler.put() {
            put_handler(txn, key, value).await
        } else {
            Err(TCError::method_not_allowed(
                HandlerType::Put,
                self,
                TCPath::from(path),
            ))
        }
    }

    async fn post(
        &self,
        txn: &Self::Txn,
        path: &[PathSegment],
        params: Map<Self::State>,
    ) -> TCResult<Self::State> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(post_handler) = handler.post() {
            post_handler(txn, params).await
        } else {
            Err(TCError::method_not_allowed(
                HandlerType::Post,
                self,
                TCPath::from(path),
            ))
        }
    }

    async fn delete(&self, txn: &Self::Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(delete_handler) = handler.delete() {
            delete_handler(txn, key).await
        } else {
            Err(TCError::method_not_allowed(
                HandlerType::Delete,
                self,
                TCPath::from(path),
            ))
        }
    }
}
