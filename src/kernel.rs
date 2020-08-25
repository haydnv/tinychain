use std::collections::HashMap;
use std::sync::Arc;

use futures::Stream;

use crate::auth::Auth;
use crate::class::{State, TCResult};
use crate::error;
use crate::transaction::Txn;
use crate::value::link::TCPath;
use crate::value::{Value, ValueId};

pub fn get(path: &TCPath, id: Value) -> TCResult<State> {
    match path[0].as_str() {
        "value" if path.len() > 1 => Value::get(&path.slice_from(1), id).map(State::Value),
        other => Err(error::not_found(other)),
    }
}

pub async fn post<S: Stream<Item = (ValueId, Value)> + Unpin>(
    txn: Arc<Txn>,
    path: &TCPath,
    values: S,
    auth: &Auth,
) -> TCResult<HashMap<ValueId, State>> {
    if path[0] == "transact" && path.len() == 2 {
        match path[1].as_str() {
            "execute" => txn.execute(auth, values).await,
            other => Err(error::not_found(other)),
        }
    } else {
        Err(error::not_found(path))
    }
}
