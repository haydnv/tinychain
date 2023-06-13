use std::fmt;

use tc_error::*;
use tc_value::{ComplexType, FloatType, IntType, Number, NumberType, UIntType};
use tcgeneric::ThreadSafe;

use crate::tensor::dense::{DenseAccess, DenseUnaryCast};
use crate::tensor::{Shape, TensorBoolean, TensorCast, TensorCompareConst, TensorInstance};

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

macro_rules! dense_view_from {
    ($t:ty, $var:ident) => {
        impl<FE> From<DenseTensor<FE, DenseAccess<FE, $t>>> for DenseView<FE> {
            fn from(tensor: DenseTensor<FE, DenseAccess<FE, $t>>) -> Self {
                Self::$var(tensor)
            }
        }
    };
}

dense_view_from!(f32, F32);
dense_view_from!(f64, F64);

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
            Self::C64(_) => NumberType::Complex(ComplexType::C64),
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

impl<FE: ThreadSafe> TensorBoolean<Self> for DenseView<FE> {
    type Combine = Self;
    type LeftCombine = Self;

    fn and(self, other: Self) -> TCResult<Self::LeftCombine> {
        todo!()
    }

    fn or(self, other: Self) -> TCResult<Self::Combine> {
        todo!()
    }

    fn xor(self, other: Self) -> TCResult<Self::Combine> {
        todo!()
    }
}

impl<FE: ThreadSafe> TensorCompareConst for DenseView<FE> {
    type Compare = Self;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        todo!()
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        todo!()
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        todo!()
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        todo!()
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        todo!()
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        todo!()
    }
}

impl<FE: ThreadSafe> TensorCast for DenseView<FE> {
    type Cast = Self;

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        const ERR_COMPLEX: &str = "cannot cast a real tensor into a complex tensor";

        macro_rules! view_dispatch_cast {
            ($var:ident) => {
                view_dispatch!(
                    self,
                    this,
                    Self::U8(this)
                        .ne_const(0u8.into())
                        .and_then(|tensor| tensor.cast_into(dtype)),
                    {
                        let real = Self::from(this.0).cast_into(dtype)?;

                        match real {
                            Self::$var(real) => Ok(Self::$var(real)),
                            real => unreachable!("cast resulted in {real:?}"),
                        }
                    },
                    {
                        let cast = DenseUnaryCast::new(this.accessor, |block| block.cast());
                        let tensor = DenseTensor::from(DenseAccess::from(cast));
                        Ok(Self::$var(tensor))
                    }
                )
            };
        }

        match dtype {
            NumberType::Number => Ok(self),
            NumberType::Bool => view_dispatch!(
                self,
                this,
                Ok(Self::Bool(this)),
                {
                    let real = Self::from(this.0);
                    let imag = Self::from(this.1);
                    real.or(imag)
                },
                {
                    let cast = DenseUnaryCast::new(this.accessor, |block| block.cast());
                    let tensor = DenseTensor::from(DenseAccess::from(cast));
                    Ok(Self::Bool(tensor))
                }
            ),
            NumberType::Complex(ComplexType::Complex) => self.cast_into(ComplexType::C32.into()),
            NumberType::Complex(ComplexType::C32) => {
                view_dispatch!(
                    self,
                    this,
                    Err(TCError::unsupported(ERR_COMPLEX)),
                    {
                        let ftype = NumberType::Float(FloatType::F32);
                        let real = Self::from(this.0).cast_into(ftype)?;
                        let imag = Self::from(this.1).cast_into(ftype)?;

                        match (real, imag) {
                            (Self::F32(real), Self::F32(imag)) => Ok(Self::C32((real, imag))),
                            (real, imag) => {
                                unreachable!("cast to f32 resulted in {real:?} and {imag:?}")
                            }
                        }
                    },
                    Err(TCError::unsupported(ERR_COMPLEX))
                )
            }
            NumberType::Complex(ComplexType::C64) => {
                view_dispatch!(
                    self,
                    this,
                    Err(TCError::unsupported(ERR_COMPLEX)),
                    {
                        let ftype = NumberType::Float(FloatType::F64);
                        let real = Self::from(this.0).cast_into(ftype)?;
                        let imag = Self::from(this.1).cast_into(ftype)?;

                        match (real, imag) {
                            (Self::F64(real), Self::F64(imag)) => Ok(Self::C64((real, imag))),
                            (real, imag) => {
                                unreachable!("cast to f64 resulted in {real:?} and {imag:?}")
                            }
                        }
                    },
                    Err(TCError::unsupported(ERR_COMPLEX))
                )
            }
            NumberType::Float(FloatType::Float) => self.cast_into(FloatType::F32.into()),
            NumberType::Float(FloatType::F32) => view_dispatch_cast!(F32),
            NumberType::Float(FloatType::F64) => view_dispatch_cast!(F64),
            NumberType::Int(IntType::Int) => self.cast_into(IntType::I32.into()),
            NumberType::Int(IntType::I16) => view_dispatch_cast!(I16),
            NumberType::Int(IntType::I8) => Err(TCError::unsupported(IntType::I8)),
            NumberType::Int(IntType::I32) => view_dispatch_cast!(I32),
            NumberType::Int(IntType::I64) => view_dispatch_cast!(I64),
            NumberType::UInt(UIntType::UInt) => self.cast_into(UIntType::U32.into()),
            NumberType::UInt(UIntType::U8) => view_dispatch_cast!(U8),
            NumberType::UInt(UIntType::U16) => view_dispatch_cast!(U16),
            NumberType::UInt(UIntType::U32) => view_dispatch_cast!(U32),
            NumberType::UInt(UIntType::U64) => view_dispatch_cast!(U64),
        }
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
