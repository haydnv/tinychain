use std::convert::TryInto;

use futures::stream;
use futures::TryFutureExt;

use crate::class::{NativeClass, State, TCResult};
use crate::collection::class::{CollectionClass, CollectionType};
use crate::error;
use crate::object::ObjectType;
use crate::request::Request;
use crate::scalar::*;
use crate::transaction::Txn;

pub async fn get(txn: &Txn, path: &TCPath, id: Value) -> TCResult<State> {
    let suffix = path.from_path(&label("sbin").into())?;
    if suffix.is_empty() {
        return Err(error::unsupported("Cannot access /sbin directly"));
    }

    println!("kernel::get /sbin{}", suffix);

    match suffix[0].as_str() {
        "chain" => Err(error::not_implemented("Instantiate Chain")),
        "collection" => {
            let ctype = CollectionType::from_path(path)?;
            ctype.get(txn, id).map_ok(State::Collection).await
        }
        "error" => Err(error::get(path, id.try_into()?)),
        "value" => {
            let dtype = ValueType::from_path(path)?;
            dtype.try_cast(id).map(Scalar::Value).map(State::Scalar)
        }
        "transact" => Err(error::method_not_allowed(suffix)),
        other => Err(error::not_found(other)),
    }
}

pub async fn post(request: &Request, txn: &Txn, path: TCPath, data: Scalar) -> TCResult<State> {
    println!("kernel::post {}", path);

    if &path == "/sbin/transact" {
        if data.matches::<Vec<(ValueId, Scalar)>>() {
            let values: Vec<(ValueId, Scalar)> = data.opt_cast_into().unwrap();
            txn.execute(request, stream::iter(values)).await
        } else if data.matches::<OpRef>() {
            Err(error::not_implemented("Resolve OpRef"))
        } else {
            Ok(State::Scalar(data))
        }
    } else if path.starts_with(&ObjectType::prefix()) {
        let data = data.try_into()?;
        ObjectType::post(path, data).map(State::Object)
    } else {
        Err(error::not_found(path))
    }
}
