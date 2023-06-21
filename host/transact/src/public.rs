use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use futures::{future, Future};
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_value::{TCString, Value};
use tcgeneric::{label, Id, Map, PathSegment, TCPath, ThreadSafe, Tuple};

use super::Transaction;

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

pub struct EchoHandler;

impl<'a, Txn, State> Handler<'a, Txn, State> for EchoHandler
where
    State: StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move { Ok(key.into()) })
        }))
    }
}

pub struct ErrorHandler<'a> {
    code: &'a Id,
}

impl<'a, Txn, State> Handler<'a, Txn, State> for ErrorHandler<'a>
where
    State: StateInstance,
    TCString: TryCastFrom<State>,
    Tuple<TCString>: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let message: TCString =
                    key.try_cast_into(|v| TCError::unexpected(v, "an error message"))?;

                if let Some(err_type) = error_type(self.code) {
                    Err(TCError::new(err_type, message.to_string()))
                } else {
                    Err(TCError::not_found(self.code))
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let message: TCString = params.require(&label("message").into())?;
                let stack: Tuple<TCString> = params.require(&label("stack").into())?;
                params.expect_empty()?;

                if let Some(err_type) = error_type(self.code) {
                    Err(TCError::with_stack(err_type, message, stack))
                } else {
                    Err(TCError::not_found(self.code))
                }
            })
        }))
    }
}

pub struct AttributeHandler<T> {
    attribute: T,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for AttributeHandler<T>
where
    State: From<T>,
    T: Clone + Send + Sync + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(self.attribute.into())
                } else {
                    Err(TCError::not_found(format!("attribute {}", key)))
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

pub struct MethodNotAllowedHandler<'a, T> {
    subject: &'a T,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for MethodNotAllowedHandler<'a, T>
where
    State: StateInstance,
    T: Clone + Send + Sync + fmt::Debug,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(move |_txn, _key| {
            Box::pin(future::ready(Err(TCError::method_not_allowed(
                HandlerType::Get,
                self.subject,
                TCPath::default(),
            ))))
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(move |_txn, _key, _value| {
            Box::pin(future::ready(Err(TCError::method_not_allowed(
                HandlerType::Put,
                self.subject,
                TCPath::default(),
            ))))
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(move |_txn, _key| {
            Box::pin(future::ready(Err(TCError::method_not_allowed(
                HandlerType::Post,
                self.subject,
                TCPath::default(),
            ))))
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(move |_txn, _key| {
            Box::pin(future::ready(Err(TCError::method_not_allowed(
                HandlerType::Delete,
                self.subject,
                TCPath::default(),
            ))))
        }))
    }
}

impl<'a, T> From<&'a T> for MethodNotAllowedHandler<'a, T> {
    fn from(subject: &'a T) -> Self {
        Self { subject }
    }
}
struct SelfHandler<'a, T> {
    subject: &'a T,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for SelfHandler<'a, T>
where
    State: From<T>,
    T: Clone + Send + Sync + fmt::Debug,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(self.subject.clone().into())
                } else {
                    Err(TCError::not_found(format!(
                        "attribute {} of {:?}",
                        key, self.subject
                    )))
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

struct SelfHandlerOwned<T> {
    subject: T,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for SelfHandlerOwned<T>
where
    State: From<T>,
    T: Send + Sync + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(self.subject.into())
                } else {
                    Err(TCError::not_found(format!(
                        "attribute {} of {:?}",
                        key, self.subject
                    )))
                }
            })
        }))
    }
}

impl<'a, T> From<T> for SelfHandlerOwned<T> {
    fn from(subject: T) -> Self {
        Self { subject }
    }
}

fn error_type(err_type: &Id) -> Option<ErrorKind> {
    match err_type.as_str() {
        "bad_gateway" => Some(ErrorKind::BadGateway),
        "bad_request" => Some(ErrorKind::BadRequest),
        "conflict" => Some(ErrorKind::Conflict),
        "forbidden" => Some(ErrorKind::Forbidden),
        "internal" => Some(ErrorKind::Internal),
        "method_not_allowed" => Some(ErrorKind::MethodNotAllowed),
        "not_found" => Some(ErrorKind::NotFound),
        "not_implemented" => Some(ErrorKind::NotImplemented),
        "timeout" => Some(ErrorKind::Timeout),
        "unauthorized" => Some(ErrorKind::Unauthorized),
        _ => None,
    }
}
