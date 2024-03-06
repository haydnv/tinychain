use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Future;

use tc_error::*;
use tc_value::{Number, Value};
use tcgeneric::{Instance, Map, PathSegment, TCPath, ThreadSafe, Tuple};

use super::{Gateway, Transaction};

pub mod generic;
pub mod helpers;
pub mod number;
pub mod string;
pub mod value;

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

#[async_trait]
pub trait ClosureInstance<State: StateInstance>: Send + Sync {
    /// Execute this `ClosureInstance` with the given `args`
    async fn call(self: Box<Self>, txn: State::Txn, args: State) -> TCResult<State>;
}

pub trait StateInstance:
    Default
    + Instance
    + Route<Self>
    + ToState<Self>
    + From<bool>
    + From<Number>
    + From<Value>
    + From<Map<Self>>
    + From<Tuple<Self>>
    + From<Self::Class>
    + From<Self::Closure>
    + Clone
    + fmt::Debug
    + 'static
{
    type FE: ThreadSafe + Clone;
    type Txn: Transaction<Self::FE> + Gateway<Self>;
    type Closure: ClosureInstance<Self>;

    /// Return `true` if this is a `Map` of states.
    fn is_map(&self) -> bool;

    /// Return `true` if this is a `Tuple` of states.
    fn is_tuple(&self) -> bool;
}

/// Trait to define a [`StateInstance`] representation of a (possibly non-[`StateInstance`]) value
pub trait ToState<State: StateInstance> {
    fn to_state(&self) -> State;
}

impl<State: StateInstance, T: Clone + Into<State>> ToState<State> for T {
    fn to_state(&self) -> State {
        self.clone().into()
    }
}

#[async_trait]
pub trait Handler<'a, State: StateInstance>: Send {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        None
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        None
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        None
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, State::Txn>>
    where
        'b: 'a,
    {
        None
    }
}

pub trait Route<State>: Send + Sync {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>>;
}

#[async_trait]
pub trait Public<State: StateInstance> {
    async fn get(&self, txn: &State::Txn, path: &[PathSegment], key: Value) -> TCResult<State>;

    async fn put(
        &self,
        txn: &State::Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()>;

    async fn post(
        &self,
        txn: &State::Txn,
        path: &[PathSegment],
        params: Map<State>,
    ) -> TCResult<State>;

    async fn delete(&self, txn: &State::Txn, path: &[PathSegment], key: Value) -> TCResult<()>;
}

#[async_trait]
impl<State: StateInstance, T: Route<State> + fmt::Debug> Public<State> for T {
    async fn get(&self, txn: &State::Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(get_handler) = handler.get() {
            get_handler(txn, key).await
        } else {
            Err(TCError::method_not_allowed(
                HandlerType::Get,
                TCPath::from(path),
            ))
        }
    }

    async fn put(
        &self,
        txn: &State::Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        let handler = self
            .route(path)
            .ok_or_else(|| not_found!("{} in {:?}", TCPath::from(path), self))?;

        if let Some(put_handler) = handler.put() {
            put_handler(txn, key, value).await
        } else {
            Err(TCError::method_not_allowed(
                HandlerType::Put,
                TCPath::from(path),
            ))
        }
    }

    async fn post(
        &self,
        txn: &State::Txn,
        path: &[PathSegment],
        params: Map<State>,
    ) -> TCResult<State> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(post_handler) = handler.post() {
            post_handler(txn, params).await
        } else {
            Err(TCError::method_not_allowed(
                HandlerType::Post,
                TCPath::from(path),
            ))
        }
    }

    async fn delete(&self, txn: &State::Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(delete_handler) = handler.delete() {
            delete_handler(txn, key).await
        } else {
            Err(TCError::method_not_allowed(
                HandlerType::Delete,
                TCPath::from(path),
            ))
        }
    }
}
