use std::fmt;

use ha_ndarray::{Array, NDArrayBoolean};
use safecast::AsType;

use tc_error::*;
use tc_value::{ComplexType, FloatType, IntType, Number, NumberType, UIntType};
use tcgeneric::ThreadSafe;

use crate::tensor::complex::ComplexCompare;
use crate::tensor::{
    Shape, TensorBoolean, TensorCast, TensorCompare, TensorInstance, TensorMath, TensorMathConst,
};

use super::{sparse_from, Node, SparseAccess, SparseCombine, SparseTensor, SparseUnaryCast};

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

impl<FE: ThreadSafe + AsType<Node>> TensorBoolean<Self> for SparseView<FE> {
    type Combine = Self;
    type LeftCombine = Self;

    fn and(self, other: Self) -> TCResult<Self::LeftCombine> {
        let this = expect_bool(self)?;
        let that = expect_bool(other)?;

        SparseCombine::new(
            this.accessor,
            that.accessor,
            |l, r| l.and(r).map(Array::from).map_err(TCError::from),
            |l, r| if l != 0 && r != 0 { 1 } else { 0 },
        )
        .map(SparseTensor::from)
        .map(sparse_from)
        .map(Self::Bool)
    }

    fn or(self, other: Self) -> TCResult<Self::Combine> {
        let this = expect_bool(self)?;
        let that = expect_bool(other)?;

        SparseCombine::new(
            this.accessor,
            that.accessor,
            |l, r| l.or(r).map(Array::from).map_err(TCError::from),
            |l, r| if l != 0 || r != 0 { 1 } else { 0 },
        )
        .map(SparseTensor::from)
        .map(sparse_from)
        .map(Self::Bool)
    }

    fn xor(self, other: Self) -> TCResult<Self::Combine> {
        let this = expect_bool(self)?;
        let that = expect_bool(other)?;

        SparseCombine::new(
            this.accessor,
            that.accessor,
            |l, r| l.xor(r).map(Array::from).map_err(TCError::from),
            |l, r| if (l != 0) ^ (r != 0) { 1 } else { 0 },
        )
        .map(SparseTensor::from)
        .map(sparse_from)
        .map(Self::Bool)
    }
}

impl<FE: ThreadSafe + AsType<Node>> TensorCast for SparseView<FE> {
    type Cast = Self;

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        const ERR_COMPLEX: &str = "cannot cast a real tensor into a complex tensor";

        macro_rules! view_dispatch_cast {
            ($var:ident) => {
                view_dispatch!(
                    self,
                    this,
                    TensorCast::cast_into(Self::U8(this), dtype),
                    {
                        let real = TensorCast::cast_into(Self::from(this.0), dtype)?;

                        match real {
                            Self::$var(real) => Ok(Self::$var(real)),
                            real => unreachable!("cast resulted in {real:?}"),
                        }
                    },
                    {
                        let cast = SparseUnaryCast::new(
                            this.accessor,
                            |block| block.cast(),
                            |n| safecast::CastInto::cast_into(Number::from(n)),
                        );

                        Ok(Self::$var(sparse_from(cast.into())))
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
                    let cast = SparseUnaryCast::new(
                        this.accessor,
                        |block| block.cast(),
                        |n| safecast::CastInto::cast_into(Number::from(n)),
                    );

                    Ok(Self::Bool(sparse_from(cast.into())))
                }
            ),
            NumberType::Complex(ComplexType::Complex) => {
                TensorCast::cast_into(self, ComplexType::C32.into())
            }
            NumberType::Complex(ComplexType::C32) => {
                view_dispatch!(
                    self,
                    this,
                    Err(TCError::unsupported(ERR_COMPLEX)),
                    {
                        let ftype = NumberType::Float(FloatType::F32);
                        let real = TensorCast::cast_into(Self::from(this.0), ftype)?;
                        let imag = TensorCast::cast_into(Self::from(this.1), ftype)?;

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
                        let real = TensorCast::cast_into(Self::from(this.0), ftype)?;
                        let imag = TensorCast::cast_into(Self::from(this.1), ftype)?;

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
            NumberType::Float(FloatType::Float) => {
                TensorCast::cast_into(self, FloatType::F32.into())
            }
            NumberType::Float(FloatType::F32) => view_dispatch_cast!(F32),
            NumberType::Float(FloatType::F64) => view_dispatch_cast!(F64),
            NumberType::Int(IntType::Int) => TensorCast::cast_into(self, IntType::I32.into()),
            NumberType::Int(IntType::I16) => view_dispatch_cast!(I16),
            NumberType::Int(IntType::I8) => Err(TCError::unsupported(IntType::I8)),
            NumberType::Int(IntType::I32) => view_dispatch_cast!(I32),
            NumberType::Int(IntType::I64) => view_dispatch_cast!(I64),
            NumberType::UInt(UIntType::UInt) => TensorCast::cast_into(self, UIntType::U32.into()),
            NumberType::UInt(UIntType::U8) => view_dispatch_cast!(U8),
            NumberType::UInt(UIntType::U16) => view_dispatch_cast!(U16),
            NumberType::UInt(UIntType::U32) => view_dispatch_cast!(U32),
            NumberType::UInt(UIntType::U64) => view_dispatch_cast!(U64),
        }
    }
}

macro_rules! sparse_compare {
    ($this:ident, $that:ident, $complex:expr, $general:expr) => {
        match ($this, $that) {
            (Self::Bool(this), Self::Bool(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::C32((lr, li)), Self::C32((rr, ri))) => {
                $complex((lr.into(), li.into()), (rr.into(), ri.into()))
            }
            (Self::C64((lr, li)), Self::C64((rr, ri))) => {
                $complex((lr.into(), li.into()), (rr.into(), ri.into()))
            }
            (Self::F32(this), Self::F32(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::F64(this), Self::F64(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::I16(this), Self::I16(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::I32(this), Self::I32(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::I64(this), Self::I64(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::U8(this), Self::U8(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::U16(this), Self::U16(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::U32(this), Self::U32(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (Self::U64(this), Self::U64(that)) => {
                $general(this, that).map(sparse_from).map(Self::Bool)
            }
            (this, that) => {
                let dtype = Ord::max(this.dtype(), that.dtype());
                let this = TensorCast::cast_into(this, dtype)?;
                let that = TensorCast::cast_into(that, dtype)?;
                $general(this, that)
            }
        }
    };
}

impl<FE: ThreadSafe + AsType<Node>> TensorCompare<Self> for SparseView<FE> {
    type Compare = Self;

    fn eq(self, other: Self) -> TCResult<Self::Compare> {
        sparse_compare!(self, other, ComplexCompare::eq, TensorCompare::eq)
    }

    fn gt(self, other: Self) -> TCResult<Self::Compare> {
        sparse_compare!(self, other, ComplexCompare::gt, TensorCompare::gt)
    }

    fn ge(self, other: Self) -> TCResult<Self::Compare> {
        sparse_compare!(self, other, ComplexCompare::ge, TensorCompare::ge)
    }

    fn lt(self, other: Self) -> TCResult<Self::Compare> {
        sparse_compare!(self, other, ComplexCompare::lt, TensorCompare::lt)
    }

    fn le(self, other: Self) -> TCResult<Self::Compare> {
        sparse_compare!(self, other, ComplexCompare::le, TensorCompare::le)
    }

    fn ne(self, other: Self) -> TCResult<Self::Compare> {
        sparse_compare!(self, other, ComplexCompare::ne, TensorCompare::ne)
    }
}

impl<FE: ThreadSafe> TensorMath<Self> for SparseView<FE> {
    type Combine = Self;
    type LeftCombine = Self;

    fn add(self, other: Self) -> TCResult<Self::Combine> {
        todo!()
    }

    fn div(self, other: Self) -> TCResult<Self::LeftCombine> {
        todo!()
    }

    fn log(self, base: Self) -> TCResult<Self::LeftCombine> {
        todo!()
    }

    fn mul(self, other: Self) -> TCResult<Self::LeftCombine> {
        todo!()
    }

    fn pow(self, other: Self) -> TCResult<Self::LeftCombine> {
        todo!()
    }

    fn sub(self, other: Self) -> TCResult<Self::Combine> {
        todo!()
    }
}

impl<FE: ThreadSafe> TensorMathConst for SparseView<FE> {
    type Combine = Self;
    type DenseCombine = Self;

    fn add_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        todo!()
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        todo!()
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        todo!()
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        todo!()
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        todo!()
    }

    fn sub_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        todo!()
    }
}

impl<FE> From<SparseTensor<FE, SparseAccess<FE, f32>>> for SparseView<FE> {
    fn from(tensor: SparseTensor<FE, SparseAccess<FE, f32>>) -> Self {
        Self::F32(tensor)
    }
}

impl<FE> From<SparseTensor<FE, SparseAccess<FE, f64>>> for SparseView<FE> {
    fn from(tensor: SparseTensor<FE, SparseAccess<FE, f64>>) -> Self {
        Self::F64(tensor)
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

#[inline]
fn expect_bool<FE>(view: SparseView<FE>) -> TCResult<SparseTensor<FE, SparseAccess<FE, u8>>>
where
    FE: ThreadSafe + AsType<Node>,
{
    match TensorCast::cast_into(view, NumberType::Bool)? {
        SparseView::Bool(that) => Ok(that),
        _ => unreachable!("failed to cast sparse tensor into boolean"),
    }
}
