use futures::stream::TryStreamExt;

use crate::error;
use crate::value::class::{NumberImpl, NumberType};
use crate::value::{Number, TCTryStream};

use super::bounds::Shape;
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

    fn broadcast(self, _shape: bounds::Shape) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn expand_dims(self, _axis: usize) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn slice(self, _bounds: bounds::Bounds) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn transpose(self, _permutation: Option<Vec<usize>>) -> TCResult<Self> {
        Err(error::not_implemented())
    }
}
