use crate::error;
use crate::state::State;
use crate::value::link::TCPath;
use crate::value::{TCResult, Value};

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
}
