use crate::value::class::NumberType;
use crate::value::{Number, TCTryStream};

use super::bounds::Shape;
use super::TensorView;

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
