use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::Future;
use safecast::TryCastFrom;
use tc_error::*;
use tcgeneric::{Id, Map, PathSegment, TCPath};

use crate::scalar::{OpRefType as ORT, Value};
use crate::state::State;
use crate::txn::Txn;

mod chain;
mod cluster;
mod collection;
mod generic;
mod object;
mod scalar;
mod state;
mod stream;

pub type GetFuture<'a> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
pub type GetHandler<'a, 'b> = Box<dyn FnOnce(&'b Txn, Value) -> GetFuture<'a> + Send + 'a>;

pub type PutFuture<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;
pub type PutHandler<'a, 'b> = Box<dyn FnOnce(&'b Txn, Value, State) -> PutFuture<'a> + Send + 'a>;

pub type PostFuture<'a> = Pin<Box<dyn Future<Output = TCResult<State>> + Send + 'a>>;
pub type PostHandler<'a, 'b> = Box<dyn FnOnce(&'b Txn, Map<State>) -> PostFuture<'a> + Send + 'a>;

pub type DeleteFuture<'a> = Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>>;
pub type DeleteHandler<'a, 'b> = Box<dyn FnOnce(&'b Txn, Value) -> DeleteFuture<'a> + Send + 'a>;

#[async_trait]
pub trait Handler<'a>: Send {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        None
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        None
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        None
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
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
            get_handler(txn, key).await
        } else {
            Err(TCError::method_not_allowed(
                ORT::Get,
                self,
                TCPath::from(path),
            ))
        }
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(put_handler) = handler.put() {
            put_handler(txn, key, value).await
        } else {
            Err(TCError::method_not_allowed(
                ORT::Put,
                self,
                TCPath::from(path),
            ))
        }
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], params: Map<State>) -> TCResult<State> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(post_handler) = handler.post() {
            post_handler(txn, params).await
        } else {
            Err(TCError::method_not_allowed(
                ORT::Post,
                self,
                TCPath::from(path),
            ))
        }
    }

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        let handler = self
            .route(path)
            .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

        if let Some(delete_handler) = handler.delete() {
            delete_handler(txn, key).await
        } else {
            Err(TCError::method_not_allowed(
                ORT::Delete,
                self,
                TCPath::from(path),
            ))
        }
    }
}

struct EchoHandler;

impl<'a> Handler<'a> for EchoHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move { Ok(key.into()) })
        }))
    }
}

struct ErrorHandler<'a> {
    code: &'a Id,
}

impl<'a> Handler<'a> for ErrorHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let message = String::try_cast_from(key, |v| {
                    TCError::bad_request("cannot cast into error message string from", v)
                })?;

                if let Some(err_type) = error_type(self.code) {
                    Err(TCError::new(err_type, message))
                } else {
                    Err(TCError::not_found(self.code))
                }
            })
        }))
    }
}

struct AttributeHandler<T> {
    attribute: T,
}

impl<'a, T: Clone + Send + Sync + 'a> Handler<'a> for AttributeHandler<T>
where
    State: From<T>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(self.attribute.into())
                } else {
                    Err(TCError::not_found(key))
                }
            })
        }))
    }
}

impl<T> From<T> for AttributeHandler<T> {
    fn from(attribute: T) -> Self {
        Self { attribute }
    }
}

struct SelfHandler<'a, T> {
    subject: &'a T,
}

impl<'a, T: Clone + Send + Sync> Handler<'a> for SelfHandler<'a, T>
where
    State: From<T>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(self.subject.clone().into())
                } else {
                    Err(TCError::not_found(key))
                }
            })
        }))
    }
}

impl<'a, T> From<&'a T> for SelfHandler<'a, T> {
    fn from(subject: &'a T) -> Self {
        Self { subject }
    }
}

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return None;
        }

        if path[0] == state::PREFIX {
            state::Static.route(&path[1..])
        } else if path[0].as_str() == "error" {
            if path.len() == 2 {
                let code = &path[1];
                Some(Box::new(ErrorHandler { code }))
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl fmt::Display for Static {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("static context")
    }
}

fn error_type(err_type: &Id) -> Option<ErrorType> {
    match err_type.as_str() {
        "bad_gateway" => Some(ErrorType::BadGateway),
        "bad_request" => Some(ErrorType::BadRequest),
        "conflict" => Some(ErrorType::Conflict),
        "forbidden" => Some(ErrorType::Forbidden),
        "internal" => Some(ErrorType::Internal),
        "method_not_allowed" => Some(ErrorType::MethodNotAllowed),
        "not_found" => Some(ErrorType::NotFound),
        "not_implemented" => Some(ErrorType::NotImplemented),
        "timeout" => Some(ErrorType::Timeout),
        "unauthorized" => Some(ErrorType::Unauthorized),
        _ => None,
    }
}
