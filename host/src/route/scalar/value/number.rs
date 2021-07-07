use safecast::TryCastInto;

use tc_error::*;
use tc_value::{Number, NumberInstance, Value};
use tcgeneric::PathSegment;

use crate::route::{GetHandler, Handler, Route};
use crate::state::State;

struct Dual<F> {
    op: F,
}

impl<F> Dual<F> {
    fn new(op: F) -> Self {
        Self { op }
    }
}

impl<'a, F> Handler<'a> for Dual<F>
where
    F: Fn(Number) -> Number + Send + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, value| {
            Box::pin(async move {
                let value = value.try_cast_into(|v| TCError::bad_request("not a Number", v))?;
                Ok(State::from(Value::from((self.op)(value))))
            })
        }))
    }
}

struct Unary<F> {
    name: &'static str,
    op: F,
}

impl<F> Unary<F> {
    fn new(name: &'static str, op: F) -> Self {
        Self { name, op }
    }
}

impl<'a, F> Handler<'a> for Unary<F>
where
    F: Fn() -> Number + Send + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, value| {
            Box::pin(async move {
                if value.is_some() {
                    return Err(TCError::unsupported(format!(
                        "{} does not have any parameters (found {})",
                        self.name, value
                    )));
                }

                Ok(State::from(Value::from((self.op)())))
            })
        }))
    }
}

impl Route for Number {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        let handler: Box<dyn Handler<'a> + 'a> = match path[0].as_str() {
            "abs" => Box::new(Unary::new("abs", move || self.abs())),
            "add" => Box::new(Dual::new(move |other| *self + other)),
            "div" => Box::new(Dual::new(move |other| *self / other)),
            "mul" => Box::new(Dual::new(move |other| *self * other)),
            "sub" => Box::new(Dual::new(move |other| *self - other)),
            "pow" => Box::new(Dual::new(move |other| self.pow(other))),
            "gt" => Box::new(Dual::new(move |other| (*self > other).into())),
            "gte" => Box::new(Dual::new(move |other| (*self >= other).into())),
            "lt" => Box::new(Dual::new(move |other| (*self < other).into())),
            "lte" => Box::new(Dual::new(move |other| (*self <= other).into())),
            _ => return None,
        };

        Some(handler)
    }
}
