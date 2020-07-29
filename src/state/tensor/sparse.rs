use crate::value::class::NumberType;

use super::bounds::Shape;
use super::TensorView;

trait SparseAccessor: TensorView {}

pub struct SparseTensor {
    accessor: Box<dyn SparseAccessor>,
}

impl TensorView for SparseTensor {
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn ndim(&self) -> usize {
        self.accessor.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.accessor.shape()
    }

    fn size(&self) -> u64 {
        self.accessor.size()
    }
}
