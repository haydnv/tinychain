use std::fmt;

use futures::future;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_value::TCString;
use tcgeneric::{label, Id, TCPath, Tuple};

use super::{
    DeleteHandler, GetHandler, Handler, HandlerType, PostHandler, PutHandler, StateInstance,
};

pub struct EchoHandler;

impl<'a, State> Handler<'a, State> for EchoHandler
where
    State: StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<'a, State> Handler<'a, State> for ErrorHandler<'a>
where
    State: StateInstance,
    TCString: TryCastFrom<State>,
    Tuple<TCString>: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
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

impl<'a, State, T> Handler<'a, State> for AttributeHandler<T>
where
    State: StateInstance + From<T>,
    T: Clone + Send + Sync + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<'a, State, T> Handler<'a, State> for MethodNotAllowedHandler<'a, T>
where
    State: StateInstance,
    T: Clone + Send + Sync + fmt::Debug,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, State::Txn, State>>
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
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

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, State::Txn>>
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

impl<'a, State, T> Handler<'a, State> for SelfHandler<'a, T>
where
    State: StateInstance + From<T>,
    T: Clone + Send + Sync + fmt::Debug,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<'a, State, T> Handler<'a, State> for SelfHandlerOwned<T>
where
    State: StateInstance + From<T>,
    T: Send + Sync + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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
