use futures::TryFutureExt;
use safecast::{Match, TryCastInto};

use tc_error::*;
use tc_tensor::{Bounds, Coord, TensorIO, TensorTransform};
use tc_transact::Transaction;
use tcgeneric::PathSegment;

use crate::collection::{Collection, Tensor};
use crate::route::{GetHandler, PutHandler};
use crate::scalar::Value;
use crate::state::State;

use super::{Handler, Route};

struct TensorHandler<'a> {
    tensor: &'a Tensor,
}

impl<'a> Handler<'a> for TensorHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<Coord>() {
                    let coord = key.opt_cast_into().unwrap();
                    self.tensor
                        .read_value(&txn, coord)
                        .map_ok(Value::from)
                        .map_ok(State::from)
                        .await
                } else if key.matches::<Bounds>() {
                    let bounds = key.opt_cast_into().unwrap();
                    self.tensor.slice(bounds).map(Collection::from).map(State::from)
                } else {
                    Err(TCError::bad_request("invalid range for tensor", key))
                }
            })
        }))
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.matches::<Coord>() {
                    let coord = key.opt_cast_into().unwrap();
                    let value = value
                        .try_cast_into(|v| TCError::bad_request("invalid tensor element", v))?;

                    self.tensor.write_value_at(*txn.id(), coord, value).await
                } else if key.matches::<Bounds>() {
                    let bounds = key.opt_cast_into().unwrap();
                    let value = value
                        .try_cast_into(|v| TCError::bad_request("invalid tensor element", v))?;

                    self.tensor.write_value(*txn.id(), bounds, value).await
                } else {
                    Err(TCError::bad_request("invalid range for tensor", key))
                }
            })
        }))
    }
}

impl<'a> From<&'a Tensor> for TensorHandler<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        Self { tensor }
    }
}

impl Route for Tensor {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(TensorHandler::from(self)))
        } else {
            None
        }
    }
}
