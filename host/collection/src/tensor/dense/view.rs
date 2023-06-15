use std::fmt;

use ha_ndarray::CDatatype;
use safecast::{AsType, CastFrom};

use tc_error::*;
use tc_value::{
    Complex, ComplexType, FloatType, IntType, Number, NumberClass, NumberInstance, NumberType,
    UIntType,
};
use tcgeneric::ThreadSafe;

use crate::tensor::complex::{ComplexCompare, ComplexMath};
use crate::tensor::sparse::Node;
use crate::tensor::{
    Shape, TensorBoolean, TensorBooleanConst, TensorCast, TensorCompare, TensorCompareConst,
    TensorInstance, TensorMath, TensorMathConst, TensorTrig, TensorUnary,
};

use super::{dense_from, DenseAccess, DenseCacheFile, DenseTensor, DenseUnaryCast};

type DenseComplex<FE, T> = (
    DenseTensor<FE, DenseAccess<FE, T>>,
    DenseTensor<FE, DenseAccess<FE, T>>,
);

pub enum DenseView<FE> {
    Bool(DenseTensor<FE, DenseAccess<FE, u8>>),
    C32(DenseComplex<FE, f32>),
    C64(DenseComplex<FE, f64>),
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

impl<FE> Clone for DenseView<FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Bool(this) => Self::Bool(this.clone()),
            Self::C32((re, im)) => Self::C32((re.clone(), im.clone())),
            Self::C64((re, im)) => Self::C64((re.clone(), im.clone())),
            Self::F32(this) => Self::F32(this.clone()),
            Self::F64(this) => Self::F64(this.clone()),
            Self::I16(this) => Self::I16(this.clone()),
            Self::I32(this) => Self::I32(this.clone()),
            Self::I64(this) => Self::I64(this.clone()),
            Self::U8(this) => Self::U8(this.clone()),
            Self::U16(this) => Self::U16(this.clone()),
            Self::U32(this) => Self::U32(this.clone()),
            Self::U64(this) => Self::U64(this.clone()),
        }
    }
}

impl<FE: ThreadSafe> DenseView<FE> {
    fn complex_from(complex: (Self, Self)) -> TCResult<Self> {
        match complex {
            (Self::F32(real), Self::F32(imag)) => Ok(Self::C32((real, imag))),
            (Self::F64(real), Self::F64(imag)) => Ok(Self::C64((real, imag))),
            (real, imag) => Err(bad_request!(
                "cannot construct a complex tensor from {real:?} and {imag:?}"
            )),
        }
    }
}

impl<FE> From<DenseTensor<FE, DenseAccess<FE, f32>>> for DenseView<FE> {
    fn from(tensor: DenseTensor<FE, DenseAccess<FE, f32>>) -> Self {
        Self::F32(tensor)
    }
}

impl<FE> From<DenseTensor<FE, DenseAccess<FE, f64>>> for DenseView<FE> {
    fn from(tensor: DenseTensor<FE, DenseAccess<FE, f64>>) -> Self {
        Self::F64(tensor)
    }
}

impl<FE> From<DenseComplex<FE, f32>> for DenseView<FE> {
    fn from(tensors: DenseComplex<FE, f32>) -> Self {
        Self::C32(tensors)
    }
}

impl<FE> From<DenseComplex<FE, f64>> for DenseView<FE> {
    fn from(tensors: DenseComplex<FE, f64>) -> Self {
        Self::C64(tensors)
    }
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

macro_rules! view_dispatch_compare {
    ($this:ident, $that:ident, $left:ident, $right:ident, $bool:expr, $complex:expr, $general:expr, $mismatch:expr) => {
        match ($this, $that) {
            (DenseView::Bool($left), DenseView::Bool($right)) => $bool,
            (DenseView::C32($left), DenseView::C32($right)) => $complex,
            (DenseView::C64($left), DenseView::C64($right)) => $complex,
            (DenseView::F32($left), DenseView::F32($right)) => $general,
            (DenseView::F64($left), DenseView::F64($right)) => $general,
            (DenseView::I16($left), DenseView::I16($right)) => $general,
            (DenseView::I32($left), DenseView::I32($right)) => $general,
            (DenseView::I64($left), DenseView::I64($right)) => $general,
            (DenseView::U8($left), DenseView::U8($right)) => $general,
            (DenseView::U16($left), DenseView::U16($right)) => $general,
            (DenseView::U32($left), DenseView::U32($right)) => $general,
            (DenseView::U64($left), DenseView::U64($right)) => $general,
            ($left, $right) => $mismatch,
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

impl<FE> TensorBoolean<Self> for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;
    type LeftCombine = Self;

    fn and(self, other: Self) -> TCResult<Self::LeftCombine> {
        view_dispatch_compare!(
            self,
            other,
            left,
            right,
            { left.and(right).map(dense_from).map(Self::Bool) },
            {
                let (lr, li) = left;
                let (rr, ri) = right;
                let left = lr.or(li)?;
                let right = rr.or(ri)?;

                left.and(right).map(dense_from).map(Self::Bool)
            },
            { left.and(right).map(dense_from).map(Self::Bool) },
            {
                let left = left.cast_into(NumberType::Bool)?;
                let right = right.cast_into(NumberType::Bool)?;
                left.and(right)
            }
        )
    }

    fn or(self, other: Self) -> TCResult<Self::Combine> {
        view_dispatch_compare!(
            self,
            other,
            left,
            right,
            { left.or(right).map(dense_from).map(Self::Bool) },
            {
                let (lr, li) = left;
                let (rr, ri) = right;
                let left = lr.or(li)?;
                let right = rr.or(ri)?;

                left.or(right).map(dense_from).map(Self::Bool)
            },
            { left.or(right).map(dense_from).map(Self::Bool) },
            {
                let left = left.cast_into(NumberType::Bool)?;
                let right = right.cast_into(NumberType::Bool)?;
                left.or(right)
            }
        )
    }

    fn xor(self, other: Self) -> TCResult<Self::Combine> {
        view_dispatch_compare!(
            self,
            other,
            left,
            right,
            { left.xor(right).map(dense_from).map(Self::Bool) },
            {
                let (lr, li) = left;
                let (rr, ri) = right;
                let left = lr.or(li)?;
                let right = rr.or(ri)?;

                left.xor(right).map(dense_from).map(Self::Bool)
            },
            { left.xor(right).map(dense_from).map(Self::Bool) },
            {
                let left = left.cast_into(NumberType::Bool)?;
                let right = right.cast_into(NumberType::Bool)?;
                left.xor(right)
            }
        )
    }
}

impl<FE> TensorBooleanConst for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        view_dispatch!(
            self,
            this,
            this.and_const(other).map(dense_from).map(Self::Bool),
            Self::from(this)
                .cast_into(NumberType::Bool)
                .and_then(|this| this.and_const(other)),
            this.and_const(other).map(dense_from).map(Self::Bool)
        )
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        view_dispatch!(
            self,
            this,
            this.or_const(other).map(dense_from).map(Self::Bool),
            Self::from(this)
                .cast_into(NumberType::Bool)
                .and_then(|this| this.or_const(other)),
            this.or_const(other).map(dense_from).map(Self::Bool)
        )
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        view_dispatch!(
            self,
            this,
            this.xor_const(other).map(dense_from).map(Self::Bool),
            Self::from(this)
                .cast_into(NumberType::Bool)
                .and_then(|this| this.xor_const(other)),
            this.xor_const(other).map(dense_from).map(Self::Bool)
        )
    }
}

macro_rules! view_compare {
    ($this:ident, $that:ident, $complex:expr, $general:expr) => {
        match ($this, $that) {
            (Self::Bool(this), Self::Bool(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::C32((lr, li)), Self::C32((rr, ri))) => {
                $complex((lr.into(), li.into()), (rr.into(), ri.into()))
            }
            (Self::C64((lr, li)), Self::C64((rr, ri))) => {
                $complex((lr.into(), li.into()), (rr.into(), ri.into()))
            }
            (Self::F32(this), Self::F32(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::F64(this), Self::F64(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::I16(this), Self::I16(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::I32(this), Self::I32(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::I64(this), Self::I64(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::U8(this), Self::U8(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::U16(this), Self::U16(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::U32(this), Self::U32(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
            }
            (Self::U64(this), Self::U64(that)) => {
                $general(this, that).map(dense_from).map(Self::Bool)
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

impl<FE> TensorCompare<Self> for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Compare = Self;

    fn eq(self, other: Self) -> TCResult<Self::Compare> {
        view_compare!(self, other, ComplexCompare::eq, TensorCompare::eq)
    }

    fn gt(self, other: Self) -> TCResult<Self::Compare> {
        view_compare!(self, other, ComplexCompare::gt, TensorCompare::gt)
    }

    fn ge(self, other: Self) -> TCResult<Self::Compare> {
        view_compare!(self, other, ComplexCompare::ge, TensorCompare::ge)
    }

    fn lt(self, other: Self) -> TCResult<Self::Compare> {
        view_compare!(self, other, ComplexCompare::lt, TensorCompare::lt)
    }

    fn le(self, other: Self) -> TCResult<Self::Compare> {
        view_compare!(self, other, ComplexCompare::le, TensorCompare::le)
    }

    fn ne(self, other: Self) -> TCResult<Self::Compare> {
        view_compare!(self, other, ComplexCompare::ne, TensorCompare::ne)
    }
}

impl<FE> TensorCompareConst for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Compare = Self;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.eq_const(other).map(dense_from).map(Self::Bool) },
            {
                let (real, imag) = this;

                let number = Complex::cast_from(other);
                let n_real = number.re();
                let n_imag = number.im();

                let real = real.eq_const(n_real.into())?;
                let imag = imag.eq_const(n_imag.into())?;
                real.and(imag).map(dense_from).map(Self::Bool)
            },
            { this.eq_const(other).map(dense_from).map(Self::Bool) }
        )
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Bool(this) => this.gt_const(other).map(dense_from).map(Self::Bool),

            Self::C32(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C32(this).abs().and_then(|abs| abs.gt_const(other))
            }

            Self::C64(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C64(this).abs().and_then(|abs| abs.gt_const(other))
            }

            Self::F32(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
            Self::F64(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
            Self::I16(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
            Self::I32(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
            Self::I64(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
            Self::U8(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
            Self::U16(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
            Self::U32(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
            Self::U64(this) => this.gt_const(other).map(dense_from).map(Self::Bool),
        }
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Bool(this) => this.ge_const(other).map(dense_from).map(Self::Bool),

            Self::C32(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C32(this).abs().and_then(|abs| abs.ge_const(other))
            }

            Self::C64(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C64(this).abs().and_then(|abs| abs.ge_const(other))
            }

            Self::F32(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
            Self::F64(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
            Self::I16(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
            Self::I32(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
            Self::I64(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
            Self::U8(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
            Self::U16(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
            Self::U32(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
            Self::U64(this) => this.ge_const(other).map(dense_from).map(Self::Bool),
        }
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Bool(this) => this.lt_const(other).map(dense_from).map(Self::Bool),

            Self::C32(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C32(this).abs().and_then(|abs| abs.lt_const(other))
            }

            Self::C64(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C64(this).abs().and_then(|abs| abs.lt_const(other))
            }

            Self::F32(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
            Self::F64(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
            Self::I16(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
            Self::I32(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
            Self::I64(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
            Self::U8(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
            Self::U16(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
            Self::U32(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
            Self::U64(this) => this.lt_const(other).map(dense_from).map(Self::Bool),
        }
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Bool(this) => this.le_const(other).map(dense_from).map(Self::Bool),

            Self::C32(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C32(this).abs().and_then(|abs| abs.le_const(other))
            }

            Self::C64(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C64(this).abs().and_then(|abs| abs.le_const(other))
            }

            Self::F32(this) => this.le_const(other).map(dense_from).map(Self::Bool),
            Self::F64(this) => this.le_const(other).map(dense_from).map(Self::Bool),
            Self::I16(this) => this.le_const(other).map(dense_from).map(Self::Bool),
            Self::I32(this) => this.le_const(other).map(dense_from).map(Self::Bool),
            Self::I64(this) => this.le_const(other).map(dense_from).map(Self::Bool),
            Self::U8(this) => this.le_const(other).map(dense_from).map(Self::Bool),
            Self::U16(this) => this.le_const(other).map(dense_from).map(Self::Bool),
            Self::U32(this) => this.le_const(other).map(dense_from).map(Self::Bool),
            Self::U64(this) => this.le_const(other).map(dense_from).map(Self::Bool),
        }
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.ne_const(other).map(dense_from).map(Self::Bool) },
            {
                let (real, imag) = this;

                let number = Complex::cast_from(other);
                let n_real = number.re();
                let n_imag = number.im();

                let real = real.ne_const(n_real.into())?;
                let imag = imag.ne_const(n_imag.into())?;
                let ne = real.and(imag)?;

                Ok(Self::Bool(DenseTensor::from_access(ne.into_inner())))
            },
            { this.ne_const(other).map(dense_from).map(Self::Bool) }
        )
    }
}

impl<FE> TensorCast for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node> + Clone,
{
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
                        let cast = DenseUnaryCast::new(this.accessor, |block| block.cast());
                        Ok(Self::$var(dense_from(cast.into())))
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
                    Ok(Self::Bool(dense_from(cast.into())))
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

impl<FE> TensorMath<Self> for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;
    type LeftCombine = Self;

    fn add(self, other: Self) -> TCResult<Self::Combine> {
        match (self, other) {
            (Self::Bool(this), Self::Bool(that)) => this.or(that).map(dense_from).map(Self::Bool),
            (Self::C32((a, b)), Self::C32((c, d))) => {
                ComplexMath::add((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C32((a, b)), Self::F32(that)) => {
                let real = a.add(that.clone()).map(dense_from)?;
                let imag = b.add(that).map(dense_from)?;
                Ok(Self::C32((real, imag)))
            }
            (Self::C32(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C32(this).add(that)
            }
            (Self::C64((a, b)), Self::C64((c, d))) => {
                ComplexMath::add((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C64((a, b)), Self::F64(that)) => {
                let real = a.add(that.clone()).map(dense_from)?;
                let imag = b.add(that).map(dense_from)?;
                Ok(Self::C64((real, imag)))
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C64(this).add(that)
            }
            (Self::F32(this), Self::F32(that)) => this.add(that).map(dense_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.add(that).map(dense_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.add(that).map(dense_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.add(that).map(dense_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.add(that).map(dense_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.add(that).map(dense_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.add(that).map(dense_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.add(that).map(dense_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.add(that).map(dense_from).map(Self::U64),
            (this, that) if this.dtype().is_real() && that.dtype().is_complex() => that.add(this),
            (this, that) => {
                let dtype = Ord::max(this.dtype(), that.dtype());
                let this = this.cast_into(dtype)?;
                let that = that.cast_into(dtype)?;
                this.add(that)
            }
        }
    }

    fn div(self, other: Self) -> TCResult<Self::LeftCombine> {
        match (self, other) {
            (Self::Bool(this), Self::Bool(that)) => this.div(that).map(dense_from).map(Self::Bool),
            (Self::C32((a, b)), Self::C32((c, d))) => {
                ComplexMath::div((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C32((a, b)), Self::F32(that)) => {
                let real = a.div(that.clone())?;
                let imag = b.div(that)?;
                Ok(Self::C32((dense_from(real), dense_from(imag))))
            }
            (Self::C32(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C32(this).div(that)
            }
            (Self::C64((a, b)), Self::C64((c, d))) => {
                ComplexMath::div((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C64((a, b)), Self::F64(that)) => {
                let real = a.div(that.clone())?;
                let imag = b.div(that)?;
                Ok(Self::C64((dense_from(real), dense_from(imag))))
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C64(this).div(that)
            }
            (Self::F32(this), Self::F32(that)) => this.div(that).map(dense_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.div(that).map(dense_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.div(that).map(dense_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.div(that).map(dense_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.div(that).map(dense_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.div(that).map(dense_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.div(that).map(dense_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.div(that).map(dense_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.div(that).map(dense_from).map(Self::U64),
            (this, that) => {
                let dtype = Ord::max(this.dtype(), that.dtype());
                let this = this.cast_into(dtype)?;
                let that = that.cast_into(dtype)?;
                this.div(that)
            }
        }
    }

    fn log(self, base: Self) -> TCResult<Self::LeftCombine> {
        match (self, base) {
            (Self::Bool(_), _) => Err(bad_request!("a boolean value has no logarithm")),
            (Self::C32(this), that) => Self::C32(this).ln()?.div(that.ln()?),
            (Self::C64(this), that) => Self::C64(this).ln()?.div(that.ln()?),
            (Self::F32(this), Self::F32(that)) => this.log(that).map(dense_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.log(that).map(dense_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.log(that).map(dense_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.log(that).map(dense_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.log(that).map(dense_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.log(that).map(dense_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.log(that).map(dense_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.log(that).map(dense_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.log(that).map(dense_from).map(Self::U64),
            (this, that) => {
                let dtype = Ord::max(this.dtype(), that.dtype());
                let this = this.cast_into(dtype)?;
                let that = that.cast_into(dtype)?;
                this.log(that)
            }
        }
    }

    fn mul(self, other: Self) -> TCResult<Self::LeftCombine> {
        match (self, other) {
            (Self::Bool(this), Self::Bool(that)) => this.mul(that).map(dense_from).map(Self::Bool),
            (Self::C32((a, b)), Self::C32((c, d))) => {
                ComplexMath::mul((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C32((a, b)), Self::F32(that)) => {
                let real = a.mul(that.clone())?;
                let imag = b.mul(that)?;
                Ok(Self::C32((dense_from(real), dense_from(imag))))
            }
            (Self::C32(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C32(this).mul(that)
            }
            (Self::C64((a, b)), Self::C64((c, d))) => {
                ComplexMath::mul((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C64((a, b)), Self::F64(that)) => {
                let real = a.mul(that.clone())?;
                let imag = b.mul(that)?;
                Ok(Self::C64((dense_from(real), dense_from(imag))))
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C64(this).mul(that)
            }
            (Self::F32(this), Self::F32(that)) => this.mul(that).map(dense_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.mul(that).map(dense_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.mul(that).map(dense_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.mul(that).map(dense_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.mul(that).map(dense_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.mul(that).map(dense_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.mul(that).map(dense_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.mul(that).map(dense_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.mul(that).map(dense_from).map(Self::U64),
            (this, that) if this.dtype().is_real() && that.dtype().is_complex() => that.mul(this),
            (this, that) => {
                let dtype = Ord::max(this.dtype(), that.dtype());
                let this = this.cast_into(dtype)?;
                let that = that.cast_into(dtype)?;
                this.mul(that)
            }
        }
    }

    fn pow(self, other: Self) -> TCResult<Self::LeftCombine> {
        match (self, other) {
            (Self::Bool(this), Self::Bool(that)) => this.pow(that).map(dense_from).map(Self::Bool),
            (Self::C32((x, y)), Self::F32(that)) => {
                ComplexMath::pow((x.into(), y.into()), that.into()).and_then(Self::complex_from)
            }
            (Self::C32(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C32(this).pow(that)
            }
            (Self::C64((x, y)), Self::F64(that)) => {
                ComplexMath::pow((x.into(), y.into()), that.into()).and_then(Self::complex_from)
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C64(this).pow(that)
            }
            (Self::F32(this), Self::F32(that)) => this.pow(that).map(dense_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.pow(that).map(dense_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.pow(that).map(dense_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.pow(that).map(dense_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.pow(that).map(dense_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.pow(that).map(dense_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.pow(that).map(dense_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.pow(that).map(dense_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.pow(that).map(dense_from).map(Self::U64),
            (this, that) if that.dtype().is_complex() => Err(not_implemented!(
                "raise {:?} to a complex power {:?}",
                this,
                that,
            )),
            (this, that) => {
                let dtype = Ord::max(this.dtype(), that.dtype());
                let this = this.cast_into(dtype)?;
                let that = that.cast_into(dtype)?;
                this.pow(that)
            }
        }
    }

    fn sub(self, other: Self) -> TCResult<Self::Combine> {
        match (self, other) {
            (Self::Bool(this), Self::Bool(that)) => this.or(that).map(dense_from).map(Self::Bool),
            (Self::C32((a, b)), Self::C32((c, d))) => {
                ComplexMath::sub((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C32((a, b)), Self::F32(that)) => {
                let real = a.sub(that.clone()).map(dense_from)?;
                let imag = b.sub(that).map(dense_from)?;
                Ok(Self::C32((real, imag)))
            }
            (Self::C32(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C32(this).sub(that)
            }
            (Self::C64((a, b)), Self::C64((c, d))) => {
                ComplexMath::sub((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C64((a, b)), Self::F64(that)) => {
                let real = a.sub(that.clone()).map(dense_from)?;
                let imag = b.sub(that).map(dense_from)?;
                Ok(Self::C64((real, imag)))
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = that.cast_into(this.0.dtype())?;
                Self::C64(this).sub(that)
            }
            (Self::F32(this), Self::F32(that)) => this.sub(that).map(dense_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.sub(that).map(dense_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.sub(that).map(dense_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.sub(that).map(dense_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.sub(that).map(dense_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.sub(that).map(dense_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.sub(that).map(dense_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.sub(that).map(dense_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.sub(that).map(dense_from).map(Self::U64),
            (this, that) if this.dtype().is_real() && that.dtype().is_complex() => that.sub(this),
            (this, that) => {
                let dtype = Ord::max(this.dtype(), that.dtype());
                let this = this.cast_into(dtype)?;
                let that = that.cast_into(dtype)?;
                this.sub(that)
            }
        }
    }
}

impl<FE: ThreadSafe> TensorMathConst for DenseView<FE> {
    type Combine = Self;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
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

    fn sub_const(self, other: Number) -> TCResult<Self::Combine> {
        todo!()
    }
}

macro_rules! view_trig {
    ($this:ident, $general32:ident, $general64:ident, $r:ident, $i:ident, $complex:expr) => {
        match $this {
            Self::Bool(this) => {
                let accessor = DenseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(DenseAccess::from(accessor).into()))
            }
            Self::C32(($r, $i)) => {
                let $r = Self::F32($r);
                let $i = Self::F32($i);
                $complex
            }
            Self::C64(($r, $i)) => {
                let $r = Self::F64($r);
                let $i = Self::F64($i);
                $complex
            }
            Self::F32(this) => {
                let accessor = DenseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(DenseAccess::from(accessor).into()))
            }
            Self::F64(this) => {
                let accessor = DenseUnaryCast::$general64(this.into_inner());
                Ok(Self::F64(DenseAccess::from(accessor).into()))
            }
            Self::I16(this) => {
                let accessor = DenseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(DenseAccess::from(accessor).into()))
            }
            Self::I32(this) => {
                let accessor = DenseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(DenseAccess::from(accessor).into()))
            }
            Self::I64(this) => {
                let accessor = DenseUnaryCast::$general64(this.into_inner());
                Ok(Self::F64(DenseAccess::from(accessor).into()))
            }
            Self::U8(this) => {
                let accessor = DenseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(DenseAccess::from(accessor).into()))
            }
            Self::U16(this) => {
                let accessor = DenseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(DenseAccess::from(accessor).into()))
            }
            Self::U32(this) => {
                let accessor = DenseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(DenseAccess::from(accessor).into()))
            }
            Self::U64(this) => {
                let accessor = DenseUnaryCast::$general64(this.into_inner());
                Ok(Self::F64(DenseAccess::from(accessor).into()))
            }
        }
    };
}

impl<FE> TensorTrig for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Unary = Self;

    fn asin(self) -> TCResult<Self::Unary> {
        view_trig!(
            self,
            asin_f32,
            asin_f64,
            x,
            y,
            Err(not_implemented!("arcsine of a complex number"))
        )
    }

    fn sin(self) -> TCResult<Self::Unary> {
        view_trig!(self, sin_f32, sin_f64, x, y, {
            let real = x.clone().sin()?.add(y.clone().cosh()?)?;
            let imag = x.cos()?.add(y.sinh()?)?;
            Self::complex_from((real, imag))
        })
    }

    fn sinh(self) -> TCResult<Self::Unary> {
        view_trig!(self, sinh_f32, sinh_f64, x, y, {
            let real = x.clone().sinh()?.mul(y.clone().cos()?)?;
            let imag = x.cosh()?.mul(y.sin()?)?;
            Self::complex_from((real, imag))
        })
    }

    fn acos(self) -> TCResult<Self::Unary> {
        view_trig!(
            self,
            acos_f32,
            acos_f64,
            x,
            y,
            Err(not_implemented!("arccosine of a complex number"))
        )
    }

    fn cos(self) -> TCResult<Self::Unary> {
        view_trig!(self, cos_f32, cos_f64, x, y, {
            let real = x.clone().cos()?.mul(y.clone().cosh()?)?;
            let imag = x.sin()?.mul(y.sinh()?)?;
            Self::complex_from((real, imag))
        })
    }

    fn cosh(self) -> TCResult<Self::Unary> {
        view_trig!(self, cosh_f32, cosh_f64, x, y, {
            let real = x.clone().cosh()?.mul(y.clone().cos()?)?;
            let imag = x.sinh()?.mul(y.sin()?)?;
            Self::complex_from((real, imag))
        })
    }

    fn atan(self) -> TCResult<Self::Unary> {
        view_trig!(
            self,
            atan_f32,
            atan_f64,
            x,
            y,
            Err(not_implemented!("arctangent of a complex number"))
        )
    }

    fn tan(self) -> TCResult<Self::Unary> {
        view_trig!(self, tan_f32, tan_f64, x, y, {
            let num_real = x.clone().sin()?.mul(y.clone().cosh()?)?;
            let num_imag = x.clone().cos()?.mul(y.clone().sinh()?)?;
            let num = Self::complex_from((num_real, num_imag))?;

            let denom_real = x.clone().cos()?.mul(y.clone().cosh()?)?;
            let denom_imag = x.clone().sin()?.mul(y.clone().sinh()?)?;
            let denom = Self::complex_from((denom_real, denom_imag))?;

            num.div(denom)
        })
    }

    fn tanh(self) -> TCResult<Self::Unary> {
        view_trig!(self, tanh_f32, tanh_f64, x, y, {
            let num_real = x.clone().sinh()?.mul(y.clone().cos()?)?;
            let num_imag = x.clone().cosh()?.mul(y.clone().sin()?)?;
            let num = Self::complex_from((num_real, num_imag))?;

            let denom_real = x.clone().cosh()?.mul(y.clone().cos()?)?;
            let denom_imag = x.sinh()?.mul(y.sin()?)?;
            let denom = Self::complex_from((denom_real, denom_imag))?;

            num.div(denom)
        })
    }
}

impl<FE> TensorUnary for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Unary = Self;

    fn abs(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.abs().map(dense_from).map(Self::Bool),

            Self::C32(this) => {
                let (real, imag) = this;

                let abs = real
                    .pow_const(2.into())?
                    .add(imag.pow_const(2.into())?)?
                    .pow_const((0.5).into())?;

                Ok(Self::F32(dense_from(abs)))
            }

            Self::C64(this) => {
                let (real, imag) = this;

                let abs = real
                    .pow_const(2.into())?
                    .add(imag.pow_const(2.into())?)?
                    .pow_const((0.5).into())?;

                Ok(Self::F64(dense_from(abs)))
            }

            Self::F32(this) => this.abs().map(dense_from).map(Self::F32),
            Self::F64(this) => this.abs().map(dense_from).map(Self::F64),
            Self::I16(this) => this.abs().map(dense_from).map(Self::I16),
            Self::I32(this) => this.abs().map(dense_from).map(Self::I32),
            Self::I64(this) => this.abs().map(dense_from).map(Self::I64),
            Self::U8(this) => this.abs().map(dense_from).map(Self::U8),
            Self::U16(this) => this.abs().map(dense_from).map(Self::U16),
            Self::U32(this) => this.abs().map(dense_from).map(Self::U32),
            Self::U64(this) => this.abs().map(dense_from).map(Self::U64),
        }
    }

    fn exp(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => Ok(Self::Bool(this)),

            Self::C32((x, y)) => {
                let r = Self::C32((x.clone(), y.clone()));
                let r = r.abs()?.exp()?;

                let y = Self::F32(y);
                let e_i_theta = Self::complex_from((y.clone().cos()?, y.sin()?))?;

                r.mul(e_i_theta)
            }

            Self::C64((x, y)) => {
                let r = Self::C64((x.clone(), y.clone()));
                let r = r.abs()?.exp()?;

                let y = Self::F64(y);
                let e_i_theta = Self::complex_from((y.clone().cos()?, y.sin()?))?;

                r.mul(e_i_theta)
            }

            Self::F32(this) => this.exp().map(dense_from).map(Self::F32),
            Self::F64(this) => this.exp().map(dense_from).map(Self::F64),
            Self::I16(this) => this.exp().map(dense_from).map(Self::I16),
            Self::I32(this) => this.exp().map(dense_from).map(Self::I32),
            Self::I64(this) => this.exp().map(dense_from).map(Self::I64),
            Self::U8(this) => this.exp().map(dense_from).map(Self::U8),
            Self::U16(this) => this.exp().map(dense_from).map(Self::U16),
            Self::U32(this) => this.exp().map(dense_from).map(Self::U32),
            Self::U64(this) => this.exp().map(dense_from).map(Self::U64),
        }
    }

    fn ln(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(_) => Err(bad_request!("a boolean value has no logarithm")),
            Self::C32((x, y)) => {
                let r = Self::C32((x.clone(), y.clone())).abs()?;
                let real = r.ln()?;
                let imag = atan2(y, x)?;
                Self::complex_from((real, imag.into()))
            }
            Self::C64((x, y)) => {
                let r = Self::C64((x.clone(), y.clone())).abs()?;
                let real = r.ln()?;
                let imag = atan2(y, x)?;
                Self::complex_from((real, imag.into()))
            }
            Self::F32(this) => this.ln().map(dense_from).map(Self::F32),
            Self::F64(this) => this.ln().map(dense_from).map(Self::F64),
            Self::I16(this) => this.ln().map(dense_from).map(Self::I16),
            Self::I32(this) => this.ln().map(dense_from).map(Self::I32),
            Self::I64(this) => this.ln().map(dense_from).map(Self::I64),
            Self::U8(this) => this.ln().map(dense_from).map(Self::U8),
            Self::U16(this) => this.ln().map(dense_from).map(Self::U16),
            Self::U32(this) => this.ln().map(dense_from).map(Self::U32),
            Self::U64(this) => this.ln().map(dense_from).map(Self::U64),
        }
    }

    fn round(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.round().map(dense_from).map(Self::Bool),
            Self::C32((real, imag)) => {
                let real = real.round().map(dense_from)?;
                let imag = imag.round().map(dense_from)?;
                Ok(Self::C32((real, imag)))
            }
            Self::C64((real, imag)) => {
                let real = real.round().map(dense_from)?;
                let imag = imag.round().map(dense_from)?;
                Ok(Self::C64((real, imag)))
            }
            Self::F32(this) => this.round().map(dense_from).map(Self::F32),
            Self::F64(this) => this.round().map(dense_from).map(Self::F64),
            Self::I16(this) => this.round().map(dense_from).map(Self::I16),
            Self::I32(this) => this.round().map(dense_from).map(Self::I32),
            Self::I64(this) => this.round().map(dense_from).map(Self::I64),
            Self::U8(this) => this.round().map(dense_from).map(Self::U8),
            Self::U16(this) => this.round().map(dense_from).map(Self::U16),
            Self::U32(this) => this.round().map(dense_from).map(Self::U32),
            Self::U64(this) => this.round().map(dense_from).map(Self::U64),
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

#[inline]
fn atan2<FE, T: CDatatype>(
    _y: DenseTensor<FE, DenseAccess<FE, T>>,
    _x: DenseTensor<FE, DenseAccess<FE, T>>,
) -> TCResult<DenseTensor<FE, DenseAccess<FE, T>>> {
    Err(not_implemented!("atan2"))
}
