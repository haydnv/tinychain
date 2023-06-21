use std::convert::TryInto;
use std::fmt;

use async_trait::async_trait;
use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{Link, Value};
use tcgeneric::{Map, NativeClass, PathSegment, TCPath, TCPathBuf};

// use crate::object::InstanceClass;
use crate::route::{Public, Static};
use crate::scalar::{OpRefType, Scalar, ScalarType};
use crate::state::{State, StateType};
use crate::txn::Txn;

use super::Dispatch;

/// The host kernel, responsible for dispatching requests to the local host
pub struct System;

#[async_trait]
impl Dispatch for System {
    async fn get(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<State> {
        if path.is_empty() {
            if key.is_some() {
                Err(TCError::not_found(format!(
                    "{} at {}",
                    key,
                    TCPath::from(path)
                )))
            } else {
                Err(forbidden!("access to /"))
            }
        } else if let Some(class) = ScalarType::from_path(path) {
            debug!("cast {} into {:?}", key, class);

            Scalar::from(key)
                .into_type(class)
                .map(State::Scalar)
                .ok_or_else(|| bad_request!("cannot case into an instance of {:?}", class))
        } else {
            Static.get(txn, path, key).await
        }
    }

    async fn put(&self, txn: &Txn, path: &[PathSegment], key: Value, value: State) -> TCResult<()> {
        if path.is_empty() {
            if key.is_none() {
                if Link::can_cast_from(&value) {
                    // It's a synchronization message for a hypothetical transaction
                    return Ok(());
                }
            }

            Err(TCError::method_not_allowed(
                OpRefType::Put,
                self,
                TCPath::from(path),
            ))
        } else if let Some(class) = StateType::from_path(path) {
            Err(TCError::method_not_allowed(
                OpRefType::Put,
                class,
                TCPath::from(path),
            ))
        } else {
            Static.put(txn, path, key, value).await
        }
    }

    async fn post(&self, txn: &Txn, path: &[PathSegment], data: State) -> TCResult<State> {
        if path.is_empty() {
            match data {
                // State::Map(map) if map.is_empty() => {
                //     // it's a "commit" instruction for a hypothetical transaction
                //     Ok(State::default())
                // }
                _ => Err(TCError::method_not_allowed(
                    OpRefType::Post,
                    self,
                    TCPath::from(path),
                )),
            }
        } else if StateType::from_path(path).is_some() {
            // let extends = Link::from(TCPathBuf::from(path.to_vec()));
            //
            // let proto =
            //     data.try_into_map(|state| TCError::unexpected(state, "a class prototype"))?;
            //
            // let proto = proto
            //     .into_iter()
            //     .map(|(key, state)| {
            //         Scalar::try_cast_from(state, |s| {
            //             TCError::unexpected(s, "a Scalar Class attribute")
            //         })
            //         .map(|scalar| (key, scalar))
            //     })
            //     .collect::<TCResult<Map<Scalar>>>()?;
            //
            // Ok(State::Object(InstanceClass::extend(extends, proto).into()))
            Err(not_implemented!("construct class definition"))
        } else {
            let params = data.try_into()?;
            Static.post(txn, path, params).await
        }
    }

    async fn delete(&self, txn: &Txn, path: &[PathSegment], key: Value) -> TCResult<()> {
        if path.is_empty() {
            Err(TCError::method_not_allowed(
                OpRefType::Delete,
                self,
                TCPath::from(path),
            ))
        } else if let Some(class) = StateType::from_path(path) {
            Err(TCError::method_not_allowed(
                OpRefType::Post,
                class,
                TCPath::default(),
            ))
        } else {
            Static.delete(txn, path, key).await
        }
    }

    async fn finalize(&self, _txn_id: TxnId) {
        // no-op
    }
}

impl fmt::Debug for System {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("the system dispatcher")
    }
}
