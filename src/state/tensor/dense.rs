use crate::value::class::NumberType;

use super::bounds::Shape;
use super::TensorView;

trait BlockList: TensorView {}

pub struct DenseTensor {
    blocks: Box<dyn BlockList>,
}

impl TensorView for DenseTensor {
    fn dtype(&self) -> NumberType {
        self.blocks.dtype()
    }

    fn ndim(&self) -> usize {
        self.blocks.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.blocks.shape()
    }

    fn size(&self) -> u64 {
        self.blocks.size()
    }
}
