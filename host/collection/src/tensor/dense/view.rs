use std::fmt;

use tc_value::{ComplexType, NumberType};
use tcgeneric::ThreadSafe;

use crate::tensor::dense::DenseAccess;
use crate::tensor::{Shape, TensorInstance};

use super::DenseTensor;

pub enum DenseView<FE> {
    Bool(DenseTensor<FE, DenseAccess<FE, u8>>),
    C32(
        (
            DenseTensor<FE, DenseAccess<FE, f32>>,
            DenseTensor<FE, DenseAccess<FE, f32>>,
        ),
    ),
    C64(
        (
            DenseTensor<FE, DenseAccess<FE, f64>>,
            DenseTensor<FE, DenseAccess<FE, f64>>,
        ),
    ),
    F32(DenseTensor<FE, DenseAccess<FE, f32>>),
    F64(DenseTensor<FE, DenseAccess<FE, f64>>),
    I16(DenseTensor<FE, DenseAccess<FE, i16>>),
    I32(DenseTensor<FE, DenseAccess<FE, i32>>),
    I64(DenseTensor<FE, DenseAccess<FE, i64>>),
    U8(DenseTensor<FE, DenseAccess<FE, u8>>),
    U16(DenseTensor<FE, DenseAccess<FE, u16>>),
    U32(DenseTensor<FE, DenseAccess<FE, u32>>),
    U64(DenseTensor<FE, DenseAccess<FE, u64>>),
}

macro_rules! view_dispatch {
    ($this:ident, $var:ident, $bool:expr, $complex:expr, $general:expr) => {
        match $this {
            DenseView::Bool($var) => $bool,
            DenseView::C32($var) => $complex,
            DenseView::C64($var) => $complex,
            DenseView::F32($var) => $general,
            DenseView::F64($var) => $general,
            DenseView::I16($var) => $general,
            DenseView::I32($var) => $general,
            DenseView::I64($var) => $general,
            DenseView::U8($var) => $general,
            DenseView::U16($var) => $general,
            DenseView::U32($var) => $general,
            DenseView::U64($var) => $general,
        }
    };
}

impl<FE: ThreadSafe> TensorInstance for DenseView<FE> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Bool(_) => NumberType::Bool,
            Self::C32(_) => NumberType::Complex(ComplexType::C32),
            Self::C64(_) => NumberType::Complex(ComplexType::C32),
            Self::F32(this) => this.dtype(),
            Self::F64(this) => this.dtype(),
            Self::I16(this) => this.dtype(),
            Self::I32(this) => this.dtype(),
            Self::I64(this) => this.dtype(),
            Self::U8(this) => this.dtype(),
            Self::U16(this) => this.dtype(),
            Self::U32(this) => this.dtype(),
            Self::U64(this) => this.dtype(),
        }
    }

    fn shape(&self) -> &Shape {
        view_dispatch!(
            self,
            this,
            this.shape(),
            {
                debug_assert_eq!(this.0.shape(), this.1.shape());
                this.0.shape()
            },
            this.shape()
        )
    }
}

impl<FE: ThreadSafe> fmt::Debug for DenseView<FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "dense tensor of type {:?} with shape {:?}",
            self.dtype(),
            self.shape()
        )
    }
}
