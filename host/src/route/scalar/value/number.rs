use safecast::TryCastInto;

use tc_error::*;
use tc_value::{Number, NumberInstance, Value};
use tcgeneric::PathSegment;

use crate::route::{GetHandler, Handler, Route};

struct Dual<'a> {
    handler: GetHandler<'a>,
}

impl<'a, F: FnOnce(Number) -> Number + Send + 'a> From<F> for Dual<'a> {
    fn from(handler: F) -> Self {
        Self {
            handler: Box::new(|_txn, other| {
                Box::pin(async move {
                    let other = other.try_cast_into(|v| {
                        TCError::bad_request("cannot cast into Number from", v)
                    })?;

                    Ok(Value::from(handler(other)).into())
                })
            }),
        }
    }
}

impl<'a> Handler<'a> for Dual<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(self.handler)
    }
}

struct Unary<'a> {
    handler: GetHandler<'a>,
}

impl<'a, F: FnOnce() -> Number + Send + 'a> From<F> for Unary<'a> {
    fn from(handler: F) -> Self {
        Self {
            handler: Box::new(|_txn, other| {
                Box::pin(async move {
                    if other.is_some() {
                        return Err(TCError::bad_request(
                            "unary operation takes no arguments, but got",
                            other,
                        ));
                    }

                    Ok(Value::from(handler()).into())
                })
            }),
        }
    }
}

impl<'a> Handler<'a> for Unary<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(self.handler)
    }
}

impl Route for Number {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        let handler: Box<dyn Handler<'a> + 'a> = match path[0].as_str() {
            "abs" => Box::new(Unary::from(move || self.abs())),
            "add" => Box::new(Dual::from(move |other| *self + other)),
            "div" => Box::new(Dual::from(move |other| *self / other)),
            "mul" => Box::new(Dual::from(move |other| *self * other)),
            "sub" => Box::new(Dual::from(move |other| *self - other)),
            "pow" => Box::new(Dual::from(move |other| self.pow(other))),
            _ => return None,
        };

        Some(handler)
    }
}
