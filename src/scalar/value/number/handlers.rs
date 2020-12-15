use std::convert::TryFrom;
use std::fmt;
use std::ops::{Add, Mul, Sub};

use async_trait::async_trait;

use crate::class::{Instance, State, TCResult, TCType};
use crate::error;
use crate::transaction::Txn;

use crate::handler::*;
use crate::scalar::{CastFrom, MethodType, PathSegment, Value};

use super::{Boolean, Number, NumberInstance};

struct GetHandler<'a, T: NumberInstance, R: NumberInstance, F: Fn(&T, T) -> R + Send + Sync>
where
    <T as Instance>::Class: Into<TCType>,
{
    number: &'a T,
    call: F,
}

#[async_trait]
impl<
        'a,
        T: NumberInstance + CastFrom<Number>,
        R: NumberInstance,
        F: Fn(&T, T) -> R + Send + Sync,
    > Handler for GetHandler<'a, T, R, F>
where
    <T as Instance>::Class: Into<TCType>,
{
    fn subject(&self) -> TCType {
        self.number.class().into()
    }

    async fn handle_get(&self, _txn: &Txn, key: Value) -> TCResult<State> {
        let that = T::cast_from(Number::try_from(key)?);

        let result: Number = (self.call)(self.number, that).into();
        Ok(Value::Number(result).into())
    }
}

struct SelfHandler<'a, T: NumberInstance> {
    number: &'a T,
}

#[async_trait]
impl<'a, T: NumberInstance> Handler for SelfHandler<'a, T>
where
    <T as Instance>::Class: Into<TCType> + fmt::Display,
{
    fn subject(&self) -> TCType {
        self.number.class().into()
    }

    async fn handle_get(&self, _txn: &Txn, key: Value) -> TCResult<State> {
        if key.is_none() {
            Ok(State::from(Value::Number((*self.number).into())))
        } else {
            Err(error::not_found(format!(
                "{} has no field {}",
                self.number.class(),
                key
            )))
        }
    }
}

pub fn route<
    'a,
    T: NumberInstance + CastFrom<Number> + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
>(
    number: &'a T,
    method: MethodType,
    path: &[PathSegment],
) -> Option<Box<dyn Handler + 'a>>
where
    <T as Instance>::Class: Into<TCType>,
{
    if method != MethodType::Get || path.len() > 1 {
        return None;
    }

    if path.is_empty() {
        return Some(Box::new(SelfHandler { number }));
    }

    let handler: Box<dyn Handler> = match path[0].as_str() {
        "add" => Box::new(GetHandler {
            number,
            call: |this, that| *this + that,
        }),
        "mul" => Box::new(GetHandler {
            number,
            call: |this, that| *this * that,
        }),
        "sub" => Box::new(GetHandler {
            number,
            call: |this, that| *this - that,
        }),

        "gt" => Box::new(GetHandler {
            number,
            call: |this, that| Boolean::from(this > &that),
        }),
        "gte" => Box::new(GetHandler {
            number,
            call: |this, that| Boolean::from(this >= &that),
        }),
        "lt" => Box::new(GetHandler {
            number,
            call: |this, that| Boolean::from(this < &that),
        }),
        "lte" => Box::new(GetHandler {
            number,
            call: |this, that| Boolean::from(this <= &that),
        }),

        _ => return None,
    };

    Some(handler)
}
