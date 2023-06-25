use safecast::TryCastFrom;

use tc_error::*;
use tc_value::uuid::Uuid;
use tc_value::{Number, TCString, Value};
use tcgeneric::{label, Id, Label, Map, PathSegment, Tuple};

use super::helpers::SelfHandler;
use super::{ClosureInstance, GetHandler, Handler, Route, StateInstance};

pub const PREFIX: Label = label("value");

struct EqHandler<F> {
    call: F,
}

impl<'a, State, F> Handler<'a, State> for EqHandler<F>
where
    State: StateInstance,
    F: FnOnce(Value) -> bool + Send + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<State> Route<State> for Value
where
    State: StateInstance + From<Value> + From<Tuple<Value>>,
    Box<dyn ClosureInstance<State>>: TryCastFrom<State>,
    Id: TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError>,
    Number: TryCastFrom<State>,
    TCString: TryCastFrom<State>,
    Tuple<State>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        let child_handler = match self {
            Self::Number(number) => number.route(path),
            Self::String(s) => s.route(path),
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

impl<'a, State: StateInstance> Handler<'a, State> for UuidHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let uuid = if key.is_some() {
                    Uuid::try_cast_from(key, |v| TCError::unexpected(v, "a UUID"))?
                } else {
                    return Err(bad_request!("missing UUID to cast into {}", self.dtype));
                };

                let value = match self.dtype {
                    "bytes" => Value::Bytes(uuid.into_bytes().into()),
                    "id" => Value::Id(uuid.into()),
                    "string" => Value::String(uuid.to_string().into()),
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

impl<State: StateInstance> Route<State> for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
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
