use std::convert::TryInto;

use futures::stream;
use futures::TryFutureExt;
use log::debug;

use crate::class::{NativeClass, State, TCResult, TCType};
use crate::collection::class::{CollectionClass, CollectionType};
use crate::error;
use crate::object::ObjectType;
use crate::request::Request;
use crate::scalar::*;
use crate::transaction::Txn;

pub async fn get(txn: &Txn, path: &[PathSegment], id: Value) -> TCResult<State> {
    let suffix = TCType::prefix().try_suffix(path)?;
    if suffix.is_empty() {
        return Err(error::unsupported("Cannot access /sbin directly"));
    }

    debug!("kernel::get /sbin{}", TCPath::from(suffix));

    match suffix[0].as_str() {
        "chain" => Err(error::not_implemented("Instantiate Chain")),
        "collection" => {
            let ctype = CollectionType::from_path(path)?;
            debug!("new Collection of type {} with schema {}", ctype, id);
            ctype.get(txn, id).map_ok(State::Collection).await
        }
        "value" => {
            let dtype = ValueType::from_path(path)?;
            dtype.try_cast(id).map(Scalar::Value).map(State::Scalar)
        }
        "transact" => Err(error::method_not_allowed(TCPath::from(suffix))),
        other => Err(error::not_found(other)),
    }
}

pub async fn post(
    request: &Request,
    txn: &Txn,
    path: &[PathSegment],
    data: Scalar,
) -> TCResult<State> {
    debug!("kernel::post {}", TCPath::from(path));

    if &path[0] == "sbin" && &path[1] == "transact" {
        if data.matches::<Vec<(Id, Scalar)>>() {
            let values: Vec<(Id, Scalar)> = data.opt_cast_into().unwrap();
            txn.execute(request, stream::iter(values)).await
        } else if data.matches::<OpRef>() {
            Err(error::not_implemented("Resolve OpRef"))
        } else {
            Ok(State::Scalar(data))
        }
    } else if path.starts_with(ObjectType::prefix().as_slice()) {
        let data = data.try_into()?;
        ObjectType::post(path, data).map(State::Object)
    } else {
        Err(error::path_not_found(path))
    }
}
