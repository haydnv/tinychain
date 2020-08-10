use futures::Stream;

use crate::auth::Auth;
use crate::class::{State, TCResult};
use crate::collection::Graph;
use crate::error;
use crate::value::link::TCPath;
use crate::value::{Value, ValueId};

pub fn get(endpoint: &TCPath, _id: Value) -> TCResult<State> {
    Err(error::not_found(endpoint))
}

pub async fn post<S: Stream<Item = (ValueId, Value)>>(
    _endpoint: &TCPath,
    _op: S,
    _auth: &Auth,
) -> TCResult<Graph> {
    Err(error::not_implemented())
}
