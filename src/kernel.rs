use futures::Stream;

use crate::auth::Auth;
use crate::class::{State, TCResult};
use crate::collection::Graph;
use crate::error;
use crate::value::link::TCPath;
use crate::value::{Value, ValueId};

pub fn get(endpoint: &TCPath, id: Value) -> TCResult<State> {
    match endpoint[0].as_str() {
        "value" if endpoint.len() > 1 => Value::get(&endpoint.slice_from(1), id).map(State::Value),
        other => Err(error::not_found(other)),
    }
}

pub async fn post<S: Stream<Item = (ValueId, Value)>>(
    _endpoint: &TCPath,
    _op: S,
    _auth: &Auth,
) -> TCResult<Graph> {
    Err(error::not_implemented())
}
