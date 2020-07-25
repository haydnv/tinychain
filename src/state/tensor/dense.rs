use crate::value::class::NumberType;
use crate::value::TCTryStream;

use super::array::Array;
use super::bounds::Shape;
use super::TensorView;

pub struct DenseTensor {
    dtype: NumberType,
    source: TCTryStream<Array>,
    shape: Shape,
}

impl TensorView for DenseTensor {
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
