use std::convert::TryFrom;

use futures::TryFutureExt;
use safecast::{Match, TryCastInto};

use tc_error::*;
use tc_tensor::{Bounds, Coord, DenseTensor, TensorIO, TensorTransform, TensorType};
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tcgeneric::PathSegment;

use crate::collection::{Collection, Tensor};
use crate::fs;
use crate::route::{GetHandler, PutHandler};
use crate::scalar::{Number, NumberClass, NumberType, Value, ValueType};
use crate::state::State;
use crate::txn::Txn;

use super::{Handler, Route};

struct ConstantHandler;

impl<'a> Handler<'a> for ConstantHandler {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<(Vec<u64>, Number)>() {
                    let (shape, value): (Vec<u64>, Number) = key.opt_cast_into().unwrap();
                    constant(&txn, shape, value).await
                } else {
                    Err(TCError::bad_request("invalid tensor schema", key))
                }
            })
        }))
    }
}

struct CreateHandler {
    class: TensorType,
}

impl<'a> Handler<'a> for CreateHandler {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<(Vec<u64>, ValueType)>() {
                    let (shape, dtype): (Vec<u64>, ValueType) = key.opt_cast_into().unwrap();
                    let dtype = NumberType::try_from(dtype)?;

                    match self.class {
                        TensorType::Dense => constant(&txn, shape.into(), dtype.zero()).await,
                    }
                } else {
                    Err(TCError::bad_request(
                        "invalid schema for constant tensor",
                        key,
                    ))
                }
            })
        }))
    }
}

impl Route for TensorType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateHandler { class: *self }))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "constant" if self == &Self::Dense => Some(Box::new(ConstantHandler)),
                _ => None,
            }
        } else {
            None
        }
    }
}

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
                    self.tensor
                        .slice(bounds)
                        .map(Collection::from)
                        .map(State::from)
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

async fn constant(txn: &Txn, shape: Vec<u64>, value: Number) -> TCResult<State> {
    let file = create_file(txn).await?;

    DenseTensor::constant(file, *txn.id(), shape, value)
        .map_ok(Tensor::from)
        .map_ok(Collection::from)
        .map_ok(State::from)
        .await
}

async fn create_file(txn: &Txn) -> TCResult<fs::File<afarray::Array>> {
    txn.context()
        .create_file_tmp(*txn.id(), TensorType::Dense)
        .await
}
