use std::collections::HashMap;
use std::sync::Arc;

use futures::Stream;

use crate::auth::Auth;
use crate::class::{State, TCResult};
use crate::collection::class::{CollectionClass, CollectionType};
use crate::error;
use crate::transaction::Txn;
use crate::value::class::ValueClass;
use crate::value::link::TCPath;
use crate::value::{Value, ValueId, ValueType};

pub async fn get(path: &TCPath, id: Value, txn: Option<Arc<Txn>>) -> TCResult<State> {
    match path[0].as_str() {
        "collection" if path.len() > 1 => {
            let txn = txn.ok_or(error::unsupported(
                "Collection requires a transaction context",
            ))?;
            CollectionType::get(txn, &path.slice_from(1), id)
                .await
                .map(State::Collection)
        }
        "value" if path.len() > 1 => ValueType::get(&path.slice_from(1), id).map(State::Value),
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
            "execute" => transact(txn, values, auth).await,
            other => Err(error::not_found(other)),
        }
    } else {
        Err(error::not_found(path))
    }
}

async fn transact<S: Stream<Item = (ValueId, Value)> + Unpin>(
    txn: Arc<Txn>,
    values: S,
    auth: &Auth,
) -> TCResult<HashMap<ValueId, State>> {
    match txn.clone().execute(auth, values).await {
        Ok(result) => {
            txn.commit().await;
            Ok(result)
        }
        Err(cause) => {
            txn.rollback().await;
            Err(cause)
        }
    }
}
