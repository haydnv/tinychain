use bytes::Bytes;
use safecast::TryCastFrom;
use uuid::Uuid;

use tc_error::TCError;
use tcgeneric::{label, Label, PathSegment};

use crate::route::{GetHandler, Handler, Route, SelfHandler};
use crate::scalar::Value;
use crate::state::State;

mod number;

pub const PREFIX: Label = label("value");

struct EqHandler<F> {
    call: F,
}

impl<'a, F: FnOnce(Value) -> bool + Send + 'a> Handler<'a> for EqHandler<F> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move { Ok(Value::from((self.call)(key)).into()) })
        }))
    }
}

impl<F> From<F> for EqHandler<F> {
    fn from(call: F) -> Self {
        Self { call }
    }
}

impl Route for Value {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        let child_handler = match self {
            Self::Number(number) => number.route(path),
            Self::Tuple(tuple) => tuple.route(path),
            _ => None,
        };

        if child_handler.is_some() {
            child_handler
        } else if path.len() == 1 {
            match path[0].as_str() {
                "eq" => Some(Box::new(EqHandler::from(move |other| self == &other))),
                "ne" => Some(Box::new(EqHandler::from(move |other| self != &other))),
                _ => None,
            }
        } else if path.is_empty() {
            Some(Box::new(SelfHandler::from(self)))
        } else {
            None
        }
    }
}

struct UuidHandler<'a> {
    dtype: &'a str,
}

impl<'a> Handler<'a> for UuidHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let uuid = if key.is_some() {
                    Uuid::try_cast_from(key, |v| TCError::bad_request("invalid UUID", v))?
                } else {
                    return Err(TCError::bad_request(
                        "missing UUID to cast into",
                        self.dtype,
                    ));
                };

                let value = match self.dtype {
                    "bytes" => Value::Bytes(Bytes::copy_from_slice(uuid.as_bytes())),
                    "id" => Value::Id(uuid.into()),
                    "string" => Value::String(uuid.to_string()),
                    other => {
                        return Err(TCError::not_found(format!("{} in {}", other, self.dtype)))
                    }
                };

                Ok(State::from(value))
            })
        }))
    }
}

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return None;
        }

        match path[0].as_str() {
            "bytes" | "id" | "string" if path.len() == 2 => match path[1].as_str() {
                "uuid" => Some(Box::new(UuidHandler {
                    dtype: path[0].as_str(),
                })),
                _ => None,
            },
            _ => None,
        }
    }
}
