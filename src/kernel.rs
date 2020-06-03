use futures::Stream;
use std::collections::HashSet;

use crate::auth::Token;
use crate::error;
use crate::gateway::op;
use crate::state::State;
use crate::value::link::TCPath;
use crate::value::{TCResult, Value, ValueId};

pub struct Kernel;

impl Kernel {
    pub fn get(_endpoint: TCPath, _id: Value) -> TCResult<State> {
        Err(error::not_implemented())
    }

    fn get_state(_endpoint: TCPath, _id: Value) -> TCResult<State> {
        Err(error::not_implemented())
    }

    fn get_value(_endpoint: TCPath, _id: Value) -> TCResult<Value> {
        Err(error::not_implemented())
    }

    fn get_string(_endpoint: TCPath, _id: Value) -> TCResult<Value> {
        Err(error::not_implemented())
    }

    fn get_link(_endpoint: TCPath, _id: Value) -> TCResult<Value> {
        Err(error::not_implemented())
    }

    // /transact/execute
    pub async fn execute<I: Stream<Item = (ValueId, Value)>>(
        &self,
        _auth: Option<Token>,
        _capture: HashSet<ValueId>,
        _request: op::Post<I>,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }

    // TODO: /transact/hypothetical, /transact/explain, /transact/background
}
