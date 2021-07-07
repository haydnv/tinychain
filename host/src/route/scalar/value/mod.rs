use tcgeneric::PathSegment;

use crate::route::{GetHandler, Handler, Route};
use crate::scalar::Value;

mod number;

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
        } else {
            None
        }
    }
}
