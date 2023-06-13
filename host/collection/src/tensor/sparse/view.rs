use std::fmt;

use tc_value::{ComplexType, NumberType};
use tcgeneric::ThreadSafe;

use crate::tensor::{Shape, TensorInstance};

use super::{SparseAccess, SparseTensor};

pub enum SparseView<FE> {
    Bool(SparseTensor<FE, SparseAccess<FE, u8>>),
    C32(
        (
            SparseTensor<FE, SparseAccess<FE, f32>>,
            SparseTensor<FE, SparseAccess<FE, f32>>,
        ),
    ),
    C64(
        (
            SparseTensor<FE, SparseAccess<FE, f64>>,
            SparseTensor<FE, SparseAccess<FE, f64>>,
        ),
    ),
    F32(SparseTensor<FE, SparseAccess<FE, f32>>),
    F64(SparseTensor<FE, SparseAccess<FE, f64>>),
    I16(SparseTensor<FE, SparseAccess<FE, i16>>),
    I32(SparseTensor<FE, SparseAccess<FE, i32>>),
    I64(SparseTensor<FE, SparseAccess<FE, i64>>),
    U8(SparseTensor<FE, SparseAccess<FE, u8>>),
    U16(SparseTensor<FE, SparseAccess<FE, u16>>),
    U32(SparseTensor<FE, SparseAccess<FE, u32>>),
    U64(SparseTensor<FE, SparseAccess<FE, u64>>),
}

macro_rules! view_dispatch {
    ($this:ident, $var:ident, $bool:expr, $complex:expr, $general:expr) => {
        match $this {
            SparseView::Bool($var) => $bool,
            SparseView::C32($var) => $complex,
            SparseView::C64($var) => $complex,
            SparseView::F32($var) => $general,
            SparseView::F64($var) => $general,
            SparseView::I16($var) => $general,
            SparseView::I32($var) => $general,
            SparseView::I64($var) => $general,
            SparseView::U8($var) => $general,
            SparseView::U16($var) => $general,
            SparseView::U32($var) => $general,
            SparseView::U64($var) => $general,
        }
    };
}

impl<FE: ThreadSafe> TensorInstance for SparseView<FE> {
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

impl<FE: ThreadSafe> fmt::Debug for SparseView<FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "sparse tensor of type {:?} with shape {:?}",
            self.dtype(),
            self.shape()
        )
    }
}
