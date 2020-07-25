use futures::stream::{self, StreamExt, TryStreamExt};

use crate::error;
use crate::value::class::{NumberImpl, NumberType};
use crate::value::{Number, TCTryStream};

use super::bounds::{Bounds, Shape};
use super::transform::Broadcast;
use super::*;

pub struct SparseTensor {
    dtype: NumberType,
    source: TCTryStream<(Vec<u64>, Number)>,
    shape: Shape,
}

impl TensorView for SparseTensor {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.shape.size()
    }
}

impl TensorTransform for SparseTensor {
    fn into_type(self, dtype: NumberType) -> TCResult<Self> {
        let shape = self.shape().clone();
        let source = Box::pin(
            self.source
                .map_ok(move |(coord, value)| (coord, value.into_type(dtype))),
        );
        Ok(SparseTensor {
            dtype,
            source,
            shape,
        })
    }

    fn broadcast(self, shape: Shape) -> TCResult<Self> {
        let dtype = self.dtype;
        let broadcast = Broadcast::new(self.shape, shape.clone())?;

        let source = Box::pin(
            self.source
                .map_ok(move |(coord, value)| {
                    let bounds = broadcast.map_bounds(coord.into());
                    stream::iter(
                        bounds
                            .affected()
                            .map(move |coord| TCResult::Ok((coord, value.clone()))),
                    )
                })
                .try_flatten(),
        );

        Ok(SparseTensor {
            dtype,
            source,
            shape,
        })
    }

    fn expand_dims(self, _axis: usize) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn slice(self, _bounds: Bounds) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn transpose(self, _permutation: Option<Vec<usize>>) -> TCResult<Self> {
        Err(error::not_implemented())
    }
}
