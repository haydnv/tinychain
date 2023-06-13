use std::fmt;

use ha_ndarray::CDatatype;
use safecast::{AsType, CastFrom};

use tc_error::*;
use tc_value::{
    Complex, ComplexType, FloatType, IntType, Number, NumberInstance, NumberType, UIntType,
};
use tcgeneric::ThreadSafe;

use crate::tensor::sparse::Node;
use crate::tensor::{
    Shape, TensorBoolean, TensorCast, TensorCompareConst, TensorInstance, TensorMath,
    TensorMathConst, TensorTrig, TensorUnary,
};

use super::{DenseAccess, DenseCacheFile, DenseTensor, DenseUnaryCast};

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

macro_rules! view_dispatch_dual {
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
        view_dispatch_dual!(
            self,
            other,
            left,
            right,
            { left.and(right).map(from_access).map(Self::Bool) },
            {
                let (lr, li) = left;
                let (rr, ri) = right;
                let left = lr.or(li)?;
                let right = rr.or(ri)?;

                left.and(right).map(from_access).map(Self::Bool)
            },
            { left.and(right).map(from_access).map(Self::Bool) },
            {
                let left = left.cast_into(NumberType::Bool)?;
                let right = right.cast_into(NumberType::Bool)?;
                left.and(right)
            }
        )
    }

    fn or(self, other: Self) -> TCResult<Self::Combine> {
        view_dispatch_dual!(
            self,
            other,
            left,
            right,
            { left.or(right).map(from_access).map(Self::Bool) },
            {
                let (lr, li) = left;
                let (rr, ri) = right;
                let left = lr.or(li)?;
                let right = rr.or(ri)?;

                left.or(right).map(from_access).map(Self::Bool)
            },
            { left.or(right).map(from_access).map(Self::Bool) },
            {
                let left = left.cast_into(NumberType::Bool)?;
                let right = right.cast_into(NumberType::Bool)?;
                left.or(right)
            }
        )
    }

    fn xor(self, other: Self) -> TCResult<Self::Combine> {
        view_dispatch_dual!(
            self,
            other,
            left,
            right,
            { left.xor(right).map(from_access).map(Self::Bool) },
            {
                let (lr, li) = left;
                let (rr, ri) = right;
                let left = lr.or(li)?;
                let right = rr.or(ri)?;

                left.xor(right).map(from_access).map(Self::Bool)
            },
            { left.xor(right).map(from_access).map(Self::Bool) },
            {
                let left = left.cast_into(NumberType::Bool)?;
                let right = right.cast_into(NumberType::Bool)?;
                left.xor(right)
            }
        )
    }
}

impl<FE> TensorCompareConst for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node>,
{
    type Compare = Self;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.eq_const(other).map(from_access).map(Self::Bool) },
            {
                let (real, imag) = this;

                let number = Complex::cast_from(other);
                let n_real = number.re();
                let n_imag = number.im();

                let real = real.eq_const(n_real.into())?;
                let imag = imag.eq_const(n_imag.into())?;
                let eq = real.and(imag)?;

                Ok(Self::Bool(DenseTensor::from_access(eq.into_inner())))
            },
            { this.eq_const(other).map(from_access).map(Self::Bool) }
        )
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Bool(this) => this.gt_const(other).map(from_access).map(Self::Bool),

            Self::C32(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C32(this).abs().and_then(|abs| abs.gt_const(other))
            }

            Self::C64(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C64(this).abs().and_then(|abs| abs.gt_const(other))
            }

            Self::F32(this) => this.gt_const(other).map(from_access).map(Self::Bool),
            Self::F64(this) => this.gt_const(other).map(from_access).map(Self::Bool),
            Self::I16(this) => this.gt_const(other).map(from_access).map(Self::Bool),
            Self::I32(this) => this.gt_const(other).map(from_access).map(Self::Bool),
            Self::I64(this) => this.gt_const(other).map(from_access).map(Self::Bool),
            Self::U8(this) => this.gt_const(other).map(from_access).map(Self::Bool),
            Self::U16(this) => this.gt_const(other).map(from_access).map(Self::Bool),
            Self::U32(this) => this.gt_const(other).map(from_access).map(Self::Bool),
            Self::U64(this) => this.gt_const(other).map(from_access).map(Self::Bool),
        }
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Bool(this) => this.ge_const(other).map(from_access).map(Self::Bool),

            Self::C32(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C32(this).abs().and_then(|abs| abs.ge_const(other))
            }

            Self::C64(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C64(this).abs().and_then(|abs| abs.ge_const(other))
            }

            Self::F32(this) => this.ge_const(other).map(from_access).map(Self::Bool),
            Self::F64(this) => this.ge_const(other).map(from_access).map(Self::Bool),
            Self::I16(this) => this.ge_const(other).map(from_access).map(Self::Bool),
            Self::I32(this) => this.ge_const(other).map(from_access).map(Self::Bool),
            Self::I64(this) => this.ge_const(other).map(from_access).map(Self::Bool),
            Self::U8(this) => this.ge_const(other).map(from_access).map(Self::Bool),
            Self::U16(this) => this.ge_const(other).map(from_access).map(Self::Bool),
            Self::U32(this) => this.ge_const(other).map(from_access).map(Self::Bool),
            Self::U64(this) => this.ge_const(other).map(from_access).map(Self::Bool),
        }
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Bool(this) => this.lt_const(other).map(from_access).map(Self::Bool),

            Self::C32(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C32(this).abs().and_then(|abs| abs.lt_const(other))
            }

            Self::C64(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C64(this).abs().and_then(|abs| abs.lt_const(other))
            }

            Self::F32(this) => this.lt_const(other).map(from_access).map(Self::Bool),
            Self::F64(this) => this.lt_const(other).map(from_access).map(Self::Bool),
            Self::I16(this) => this.lt_const(other).map(from_access).map(Self::Bool),
            Self::I32(this) => this.lt_const(other).map(from_access).map(Self::Bool),
            Self::I64(this) => this.lt_const(other).map(from_access).map(Self::Bool),
            Self::U8(this) => this.lt_const(other).map(from_access).map(Self::Bool),
            Self::U16(this) => this.lt_const(other).map(from_access).map(Self::Bool),
            Self::U32(this) => this.lt_const(other).map(from_access).map(Self::Bool),
            Self::U64(this) => this.lt_const(other).map(from_access).map(Self::Bool),
        }
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Bool(this) => this.le_const(other).map(from_access).map(Self::Bool),

            Self::C32(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C32(this).abs().and_then(|abs| abs.le_const(other))
            }

            Self::C64(this) => {
                let other = Complex::cast_from(other).abs().into();
                Self::C64(this).abs().and_then(|abs| abs.le_const(other))
            }

            Self::F32(this) => this.le_const(other).map(from_access).map(Self::Bool),
            Self::F64(this) => this.le_const(other).map(from_access).map(Self::Bool),
            Self::I16(this) => this.le_const(other).map(from_access).map(Self::Bool),
            Self::I32(this) => this.le_const(other).map(from_access).map(Self::Bool),
            Self::I64(this) => this.le_const(other).map(from_access).map(Self::Bool),
            Self::U8(this) => this.le_const(other).map(from_access).map(Self::Bool),
            Self::U16(this) => this.le_const(other).map(from_access).map(Self::Bool),
            Self::U32(this) => this.le_const(other).map(from_access).map(Self::Bool),
            Self::U64(this) => this.le_const(other).map(from_access).map(Self::Bool),
        }
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.ne_const(other).map(from_access).map(Self::Bool) },
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
            { this.ne_const(other).map(from_access).map(Self::Bool) }
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
                        Ok(Self::$var(from_access(cast.into())))
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
                    Ok(Self::Bool(from_access(cast.into())))
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
    FE: ThreadSafe,
{
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

impl<FE> TensorTrig for DenseView<FE>
where
    FE: ThreadSafe,
{
    type Unary = Self;

    fn asin(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn sin(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn asinh(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn sinh(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn acos(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn cos(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn acosh(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn cosh(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn atan(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn tan(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn tanh(self) -> TCResult<Self::Unary> {
        todo!()
    }

    fn atanh(self) -> TCResult<Self::Unary> {
        todo!()
    }
}

impl<FE> TensorUnary for DenseView<FE>
where
    FE: DenseCacheFile + AsType<Node>,
{
    type Unary = Self;

    fn abs(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.abs().map(from_access).map(Self::Bool),

            Self::C32(this) => {
                let (real, imag) = this;

                let abs = real
                    .pow_const(2.into())?
                    .add(imag.pow_const(2.into())?)?
                    .pow_const((0.5).into())?;

                Ok(Self::F32(from_access(abs)))
            }

            Self::C64(this) => {
                let (real, imag) = this;

                let abs = real
                    .pow_const(2.into())?
                    .add(imag.pow_const(2.into())?)?
                    .pow_const((0.5).into())?;

                Ok(Self::F64(from_access(abs)))
            }

            Self::F32(this) => this.abs().map(from_access).map(Self::F32),
            Self::F64(this) => this.abs().map(from_access).map(Self::F64),
            Self::I16(this) => this.abs().map(from_access).map(Self::I16),
            Self::I32(this) => this.abs().map(from_access).map(Self::I32),
            Self::I64(this) => this.abs().map(from_access).map(Self::I64),
            Self::U8(this) => this.abs().map(from_access).map(Self::U8),
            Self::U16(this) => this.abs().map(from_access).map(Self::U16),
            Self::U32(this) => this.abs().map(from_access).map(Self::U32),
            Self::U64(this) => this.abs().map(from_access).map(Self::U64),
        }
    }

    fn exp(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => Ok(Self::Bool(this)),

            Self::C32((a, b)) => {
                let r = Self::C32((a.clone(), b.clone()));
                let r = r.abs()?.exp()?;

                let b = Self::F32(b);
                let e_i_theta = match (b.clone().cos()?, b.sin()?) {
                    (Self::F32(b_cos), Self::F32(b_sin)) => {
                        Self::C32((from_access(b_cos), from_access(b_sin)))
                    }
                    _ => unreachable!("trigonometric function returned non-float value"),
                };

                r.mul(e_i_theta)
            }

            Self::C64((x, y)) => {
                let r = Self::C64((x.clone(), y.clone()));
                let r = r.abs()?.exp()?;

                let y = Self::F64(y);
                let e_i_theta = match (y.clone().cos()?, y.sin()?) {
                    (Self::F64(y_cos), Self::F64(y_sin)) => {
                        Self::C64((from_access(y_cos), from_access(y_sin)))
                    }
                    _ => unreachable!("trigonometric function returned non-float value"),
                };

                r.mul(e_i_theta)
            }

            Self::F32(this) => this.exp().map(from_access).map(Self::F32),
            Self::F64(this) => this.exp().map(from_access).map(Self::F64),
            Self::I16(this) => this.exp().map(from_access).map(Self::I16),
            Self::I32(this) => this.exp().map(from_access).map(Self::I32),
            Self::I64(this) => this.exp().map(from_access).map(Self::I64),
            Self::U8(this) => this.exp().map(from_access).map(Self::U8),
            Self::U16(this) => this.exp().map(from_access).map(Self::U16),
            Self::U32(this) => this.exp().map(from_access).map(Self::U32),
            Self::U64(this) => this.exp().map(from_access).map(Self::U64),
        }
    }

    fn ln(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.ln().map(from_access).map(Self::Bool),
            Self::C32((x, y)) => {
                let r = Self::C32((x.clone(), y.clone())).abs()?;
                let real = match r.ln()? {
                    Self::F32(real) => real,
                    _ => unreachable!("magnitude of a 32-bit complex tensor is not a 32-bit float"),
                };

                let imag = atan2(y, x)?;

                Ok(Self::C32((real, imag)))
            }
            Self::C64((x, y)) => {
                let r = Self::C64((x.clone(), y.clone())).abs()?;
                let real = match r.ln()? {
                    Self::F64(real) => real,
                    _ => unreachable!("magnitude of a 32-bit complex tensor is not a 32-bit float"),
                };

                let imag = atan2(y, x)?;

                Ok(Self::C64((real, imag)))
            }
            Self::F32(this) => this.ln().map(from_access).map(Self::F32),
            Self::F64(this) => this.ln().map(from_access).map(Self::F64),
            Self::I16(this) => this.ln().map(from_access).map(Self::I16),
            Self::I32(this) => this.ln().map(from_access).map(Self::I32),
            Self::I64(this) => this.ln().map(from_access).map(Self::I64),
            Self::U8(this) => this.ln().map(from_access).map(Self::U8),
            Self::U16(this) => this.ln().map(from_access).map(Self::U16),
            Self::U32(this) => this.ln().map(from_access).map(Self::U32),
            Self::U64(this) => this.ln().map(from_access).map(Self::U64),
        }
    }

    fn round(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.round().map(from_access).map(Self::Bool),
            Self::C32((real, imag)) => {
                let real = real.round().map(from_access)?;
                let imag = imag.round().map(from_access)?;
                Ok(Self::C32((real, imag)))
            }
            Self::C64((real, imag)) => {
                let real = real.round().map(from_access)?;
                let imag = imag.round().map(from_access)?;
                Ok(Self::C64((real, imag)))
            }
            Self::F32(this) => this.round().map(from_access).map(Self::F32),
            Self::F64(this) => this.round().map(from_access).map(Self::F64),
            Self::I16(this) => this.round().map(from_access).map(Self::I16),
            Self::I32(this) => this.round().map(from_access).map(Self::I32),
            Self::I64(this) => this.round().map(from_access).map(Self::I64),
            Self::U8(this) => this.round().map(from_access).map(Self::U8),
            Self::U16(this) => this.round().map(from_access).map(Self::U16),
            Self::U32(this) => this.round().map(from_access).map(Self::U32),
            Self::U64(this) => this.round().map(from_access).map(Self::U64),
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
fn from_access<FE, A, T>(tensor: DenseTensor<FE, A>) -> DenseTensor<FE, DenseAccess<FE, T>>
where
    A: Into<DenseAccess<FE, T>>,
    T: CDatatype,
{
    from_access(tensor)
}

#[inline]
fn atan2<FE, T: CDatatype>(
    _y: DenseTensor<FE, DenseAccess<FE, T>>,
    _x: DenseTensor<FE, DenseAccess<FE, T>>,
) -> TCResult<DenseTensor<FE, DenseAccess<FE, T>>> {
    Err(not_implemented!("atan2"))
}
