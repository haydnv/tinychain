use std::convert::TryInto;

use futures::stream;
use futures::TryFutureExt;
use log::debug;

use crate::chain::{ChainClass, ChainType};
use crate::class::{NativeClass, State, TCResult, TCType};
use crate::collection::class::{CollectionClass, CollectionType};
use crate::error::{self, ErrorType};
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
        "chain" => {
            let ctype = ChainType::from_path(path)?;
            let (dtype, schema): (TCPathBuf, Value) =
                id.try_cast_into(|v| error::bad_request("Expected (Class, Schema) but found", v))?;

            ctype
                .get(txn, TCType::from_path(&dtype)?, schema)
                .map_ok(State::Chain)
                .await
        }
        "collection" => {
            let ctype = CollectionType::from_path(path)?;
            ctype.get(txn, id).map_ok(State::Collection).await
        }
        "error" => {
            let etype = ErrorType::from_path(path)?;
            Err(etype.get(id))
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

    if path.is_empty() {
        Err(error::method_not_allowed("/"))
    } else if &path[0] == "sbin" && &path[1] == "transact" {
        if data.matches::<Vec<(Id, Scalar)>>() {
            let values: Vec<(Id, Scalar)> = data.opt_cast_into().unwrap();
            txn.execute(request, stream::iter(values), None).await
        } else {
            Ok(State::Scalar(data))
        }
    } else if path.starts_with(ObjectType::prefix().as_slice()) {
        let data = data.try_into()?;
        ObjectType::post(path, data).map(State::Object)
    } else {
        match path[0].as_str() {
            "sbin" if path.len() > 1 => match path[1].as_str() {
                "chain" | "collection" | "error" | "value" => {
                    Err(error::method_not_allowed(&path[1]))
                }
                other => Err(error::not_found(other)),
            },
            other => Err(error::not_found(other)),
        }
    }
}
