use safecast::{CastFrom, TryCastFrom, TryCastInto};
use std::marker::PhantomData;

use tc_error::*;
use tc_value::{Float, Number, NumberClass, NumberInstance, Trigonometry, Value};
use tcgeneric::{label, PathSegment};

use super::{GetHandler, Handler, PostHandler, Route, StateInstance};

struct Dual<F> {
    op: F,
}

impl<F> Dual<F> {
    fn new(op: F) -> Self {
        Self { op }
    }
}

impl<'a, State, F> Handler<'a, State> for Dual<F>
where
    State: StateInstance,
    F: Fn(Number) -> TCResult<Number> + Send + 'a,
    Number: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, value| {
            Box::pin(async move {
                let value = value.try_cast_into(|v| TCError::unexpected(v, "a Number"))?;
                (self.op)(value).map(Value::Number).map(State::from)
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let value: Number = params.require(&label("r").into())?;
                params.expect_empty()?;

                (self.op)(value).map(Value::Number).map(State::from)
            })
        }))
    }
}

// TODO: should this be more general, like `DualWithDefaultArg`?
struct Log {
    n: Number,
}

impl Log {
    fn new(n: Number) -> Self {
        Self { n }
    }
}

impl<'a, State: StateInstance> Handler<'a, State> for Log
where
    Value: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, value| {
            Box::pin(async move {
                if self.n == Number::from(0) {
                    return Err(bad_request!("the logarithm of zero is undefined"));
                }

                let log = if value.is_none() {
                    Ok(self.n.ln())
                } else {
                    let base: Number =
                        value.try_cast_into(|v| TCError::unexpected(v, "a Number"))?;

                    if base.class().is_complex() {
                        Err(bad_request!("invalid base {} for log", base))
                    } else {
                        let base = Float::cast_from(base);
                        Ok(self.n.log(base))
                    }
                }?;

                Ok(Value::Number(log).into())
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let base: Value = params.or_default(&label("r").into())?;
                params.expect_empty()?;

                let log = if base.is_none() {
                    self.n.ln()
                } else {
                    let base: Number =
                        base.try_cast_into(|v| bad_request!("invalid base {} for log", v))?;

                    if base.class().is_complex() {
                        return Err(bad_request!("log does not support a complex base {}", base));
                    }

                    let base = Float::cast_from(base);
                    self.n.log(base)
                };

                Ok(Value::Number(log).into())
            })
        }))
    }
}

struct Unary<State, F> {
    name: &'static str,
    op: F,
    state: PhantomData<State>,
}

impl<State, F> Unary<State, F> {
    fn new(name: &'static str, op: F) -> Self {
        Self {
            name,
            op,
            state: PhantomData,
        }
    }
}

impl<'a, State, F> Handler<'a, State> for Unary<State, F>
where
    State: StateInstance,
    F: Fn() -> Number + Send + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, value| {
            Box::pin(async move {
                if value.is_some() {
                    return Err(bad_request!(
                        "{} does not have any parameters (found {})",
                        self.name,
                        value
                    ));
                }

                Ok(State::from(Value::from((self.op)())))
            })
        }))
    }
}

impl<State: StateInstance> Route<State> for Number
where
    Number: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        let handler: Box<dyn Handler<'a, State> + 'a> = match path[0].as_str() {
            // basic math
            "abs" => Box::new(Unary::new("abs", move || self.abs())),
            "add" => Box::new(Dual::new(move |other| Ok(*self + other))),
            "and" => Box::new(Dual::new(move |other| Ok(self.and(other)))),
            "div" => Box::new(Dual::new(move |other: Number| {
                if other == other.class().zero() {
                    Err(bad_request!("cannot divide by zero"))
                } else {
                    Ok(*self / other)
                }
            })),
            "exp" => Box::new(Unary::new("exp", move || self.exp())),
            "ln" => Box::new(Unary::new("ln", move || self.ln())),
            "log" => Box::new(Log::new(*self)),
            "mod" => Box::new(Dual::new(move |other| Ok(*self % other))),
            "mul" => Box::new(Dual::new(move |other| Ok(*self * other))),
            "round" => Box::new(Unary::new("round", move || self.round())),
            "sub" => Box::new(Dual::new(move |other| Ok(*self - other))),
            "pow" => Box::new(Dual::new(move |other| Ok(self.pow(other)))),

            // comparison
            "gt" => Box::new(Dual::new(move |other| Ok((*self > other).into()))),
            "gte" => Box::new(Dual::new(move |other| Ok((*self >= other).into()))),
            "lt" => Box::new(Dual::new(move |other| Ok((*self < other).into()))),
            "lte" => Box::new(Dual::new(move |other| Ok((*self <= other).into()))),
            "not" => Box::new(Unary::new("not", move || self.not())),
            "or" => Box::new(Dual::new(move |other| Ok(self.or(other)))),
            "xor" => Box::new(Dual::new(move |other| Ok(self.xor(other)))),

            // trigonometry
            "asin" => Box::new(Unary::new("abs", move || self.asin())),
            "sin" => Box::new(Unary::new("sin", move || self.sin())),
            "asinh" => Box::new(Unary::new("asinh", move || self.asinh())),
            "sinh" => Box::new(Unary::new("sinh", move || self.sinh())),

            "acos" => Box::new(Unary::new("acos", move || self.acos())),
            "cos" => Box::new(Unary::new("cos", move || self.cos())),
            "acosh" => Box::new(Unary::new("acosh", move || self.acosh())),
            "cosh" => Box::new(Unary::new("cosh", move || self.cosh())),

            "atan" => Box::new(Unary::new("atan", move || self.atan())),
            "tan" => Box::new(Unary::new("tan", move || self.tan())),
            "atanh" => Box::new(Unary::new("atanh", move || self.atanh())),
            "tanh" => Box::new(Unary::new("tanh", move || self.tanh())),

            // complex
            "imag" => match self {
                Number::Complex(c) => Box::new(Unary::new("imag", move || c.im().into())),
                _real => Box::new(Unary::new("imag", || Number::from(0.0f32))),
            },
            "real" => match self {
                Number::Complex(c) => Box::new(Unary::new("real", move || c.re().into())),
                real => Box::new(Unary::new("real", move || *real)),
            },

            _ => return None,
        };

        Some(handler)
    }
}
