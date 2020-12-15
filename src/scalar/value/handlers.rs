use async_trait::async_trait;

use crate::class::{Instance, State, TCResult, TCType};
use crate::error;
use crate::handler::Handler;
use crate::transaction::Txn;

use super::{Boolean, Number, Value};
use crate::scalar::TryCastInto;

pub struct EqHandler<'a> {
    value: &'a Value,
}

#[async_trait]
impl<'a> Handler for EqHandler<'a> {
    fn subject(&self) -> TCType {
        self.value.class().into()
    }

    async fn handle_get(&self, _txn: &Txn, other: Value) -> TCResult<State> {
        Ok(State::from(Value::Number(Number::Bool(Boolean::from(
            self.value == &other,
        )))))
    }
}

impl<'a> From<&'a Value> for EqHandler<'a> {
    fn from(value: &'a Value) -> Self {
        Self { value }
    }
}

pub struct SelfHandler<'a> {
    value: &'a Value,
}

#[async_trait]
impl<'a> Handler for SelfHandler<'a> {
    fn subject(&self) -> TCType {
        self.value.class().into()
    }

    async fn handle_get(&self, _txn: &Txn, key: Value) -> TCResult<State> {
        if key.is_none() {
            Ok(State::from(self.value.clone()))
        } else if let Value::Tuple(tuple) = self.value {
            let i: usize =
                key.try_cast_into(|v| error::bad_request("Invalid index for tuple", v))?;
            tuple.get(i).cloned().map(State::from).ok_or_else(|| {
                error::not_found(format!("Index {} in tuple of size {}", i, tuple.len()))
            })
        } else {
            Err(error::not_found(format!(
                "{} has no field {}",
                self.value.class(),
                key
            )))
        }
    }
}

impl<'a> From<&'a Value> for SelfHandler<'a> {
    fn from(value: &'a Value) -> Self {
        Self { value }
    }
}
