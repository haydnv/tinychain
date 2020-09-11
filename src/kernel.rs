use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use futures::{Stream, TryFutureExt};

use crate::auth::Auth;
use crate::chain::{ChainClass, ChainType};
use crate::class::{State, TCResult, TCStream};
use crate::collection::class::{CollectionClass, CollectionType};
use crate::error;
use crate::transaction::Txn;
use crate::value::class::ValueClass;
use crate::value::link::TCPath;
use crate::value::op::{Capture, GetOp};
use crate::value::{label, Value, ValueId, ValueType};

const ERR_TXN_REQUIRED: &str = "Collection requires a transaction context";

pub async fn get(path: &TCPath, id: Value, txn: Option<Arc<Txn>>) -> TCResult<State> {
    println!("kernel::get {}", path);

    let suffix = path.from_path(&label("sbin").into())?;
    if suffix.is_empty() {
        return Err(error::unsupported("Cannot access /sbin directly"));
    }

    match suffix[0].as_str() {
        "chain" => {
            let txn = txn.ok_or_else(|| error::unsupported(ERR_TXN_REQUIRED))?;
            let ((ctype, schema), ops): ((TCPath, Value), Vec<(ValueId, GetOp)>) = id.try_into()?;
            let ops: HashMap<ValueId, GetOp> = ops.into_iter().collect();
            ChainType::get(txn, path, ctype, schema, ops)
                .map_ok(State::Chain)
                .await
        }
        "collection" if path.len() > 1 => {
            let txn = txn.ok_or_else(|| error::unsupported(ERR_TXN_REQUIRED))?;
            CollectionType::get(txn, path, id)
                .map_ok(State::Collection)
                .await
        }
        "value" if path.len() > 1 => ValueType::get(path, id).map(State::Value),
        other => Err(error::not_found(other)),
    }
}

pub async fn post<S: Stream<Item = (ValueId, Value)> + Unpin>(
    txn: Arc<Txn>,
    path: &TCPath,
    values: S,
    capture: &Capture,
    auth: Auth,
) -> TCResult<Vec<TCStream<Value>>> {
    if path[0] == "transact" && path.len() == 2 {
        match path[1].as_str() {
            "execute" => transact(txn, values, capture, auth.clone()).await,
            other => Err(error::not_found(other)),
        }
    } else {
        Err(error::not_found(path))
    }
}

async fn transact<S: Stream<Item = (ValueId, Value)> + Unpin>(
    txn: Arc<Txn>,
    values: S,
    capture: &Capture,
    auth: Auth,
) -> TCResult<Vec<TCStream<Value>>> {
    match txn.clone().execute_and_stream(values, capture, auth).await {
        Ok(response) => {
            txn.commit().await;
            Ok(response)
        }
        Err(cause) => {
            txn.rollback().await;
            Err(cause)
        }
    }
}
