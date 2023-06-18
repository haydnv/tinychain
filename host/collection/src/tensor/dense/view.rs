use std::fmt;

use async_trait::async_trait;
use ha_ndarray::CDatatype;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::{ComplexType, FloatType, IntType, Number, NumberClass, NumberType, UIntType};
use tcgeneric::ThreadSafe;

use crate::tensor::complex::{ComplexCompare, ComplexMath, ComplexRead, ComplexTrig, ComplexUnary};
use crate::tensor::sparse::{sparse_from, Node};
use crate::tensor::{
    Axes, Coord, Range, Shape, SparseView, TensorBoolean, TensorBooleanConst, TensorCast,
    TensorCompare, TensorCompareConst, TensorConvert, TensorDiagonal, TensorInstance, TensorMath,
    TensorMathConst, TensorRead, TensorReduce, TensorTransform, TensorTrig, TensorUnary,
    TensorUnaryBoolean,
};

use super::{dense_from, DenseAccess, DenseCacheFile, DenseTensor, DenseUnaryCast};

type DenseComplex<Txn, FE, T> = (
    DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>,
    DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>,
);

pub enum DenseView<Txn, FE> {
    Bool(DenseTensor<Txn, FE, DenseAccess<Txn, FE, u8>>),
    C32(DenseComplex<Txn, FE, f32>),
    C64(DenseComplex<Txn, FE, f64>),
    F32(DenseTensor<Txn, FE, DenseAccess<Txn, FE, f32>>),
    F64(DenseTensor<Txn, FE, DenseAccess<Txn, FE, f64>>),
    I16(DenseTensor<Txn, FE, DenseAccess<Txn, FE, i16>>),
    I32(DenseTensor<Txn, FE, DenseAccess<Txn, FE, i32>>),
    I64(DenseTensor<Txn, FE, DenseAccess<Txn, FE, i64>>),
    U8(DenseTensor<Txn, FE, DenseAccess<Txn, FE, u8>>),
    U16(DenseTensor<Txn, FE, DenseAccess<Txn, FE, u16>>),
    U32(DenseTensor<Txn, FE, DenseAccess<Txn, FE, u32>>),
    U64(DenseTensor<Txn, FE, DenseAccess<Txn, FE, u64>>),
}

impl<Txn, FE> Clone for DenseView<Txn, FE> {
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

impl<Txn: ThreadSafe, FE: ThreadSafe> DenseView<Txn, FE> {
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

impl<Txn, FE> From<DenseTensor<Txn, FE, DenseAccess<Txn, FE, f32>>> for DenseView<Txn, FE> {
    fn from(tensor: DenseTensor<Txn, FE, DenseAccess<Txn, FE, f32>>) -> Self {
        Self::F32(tensor)
    }
}

impl<Txn, FE> From<DenseTensor<Txn, FE, DenseAccess<Txn, FE, f64>>> for DenseView<Txn, FE> {
    fn from(tensor: DenseTensor<Txn, FE, DenseAccess<Txn, FE, f64>>) -> Self {
        Self::F64(tensor)
    }
}

impl<Txn, FE> From<DenseComplex<Txn, FE, f32>> for DenseView<Txn, FE> {
    fn from(tensors: DenseComplex<Txn, FE, f32>) -> Self {
        Self::C32(tensors)
    }
}

impl<Txn, FE> From<DenseComplex<Txn, FE, f64>> for DenseView<Txn, FE> {
    fn from(tensors: DenseComplex<Txn, FE, f64>) -> Self {
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

impl<Txn: ThreadSafe, FE: ThreadSafe> TensorInstance for DenseView<Txn, FE> {
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

impl<Txn, FE> TensorBoolean<Self> for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
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
                let left = TensorCast::cast_into(left, NumberType::Bool)?;
                let right = TensorCast::cast_into(right, NumberType::Bool)?;
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
                let left = TensorCast::cast_into(left, NumberType::Bool)?;
                let right = TensorCast::cast_into(right, NumberType::Bool)?;
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
                let left = TensorCast::cast_into(left, NumberType::Bool)?;
                let right = TensorCast::cast_into(right, NumberType::Bool)?;
                left.xor(right)
            }
        )
    }
}

impl<Txn, FE> TensorBooleanConst for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        view_dispatch!(
            self,
            this,
            this.and_const(other).map(dense_from).map(Self::Bool),
            TensorCast::cast_into(Self::from(this), NumberType::Bool)
                .and_then(|this| this.and_const(other)),
            this.and_const(other).map(dense_from).map(Self::Bool)
        )
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        view_dispatch!(
            self,
            this,
            this.or_const(other).map(dense_from).map(Self::Bool),
            TensorCast::cast_into(Self::from(this), NumberType::Bool)
                .and_then(|this| this.or_const(other)),
            this.or_const(other).map(dense_from).map(Self::Bool)
        )
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        view_dispatch!(
            self,
            this,
            this.xor_const(other).map(dense_from).map(Self::Bool),
            TensorCast::cast_into(Self::from(this), NumberType::Bool)
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

impl<Txn, FE> TensorCompare<Self> for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
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

impl<Txn, FE> TensorCompareConst for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Compare = Self;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.eq_const(other).map(dense_from).map(Self::Bool) },
            ComplexCompare::eq_const((this.0.into(), this.1.into()), other),
            { this.eq_const(other).map(dense_from).map(Self::Bool) }
        )
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.gt_const(other).map(dense_from).map(Self::Bool) },
            ComplexCompare::gt_const((this.0.into(), this.1.into()), other),
            { this.gt_const(other).map(dense_from).map(Self::Bool) }
        )
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.ge_const(other).map(dense_from).map(Self::Bool) },
            ComplexCompare::ge_const((this.0.into(), this.1.into()), other),
            { this.ge_const(other).map(dense_from).map(Self::Bool) }
        )
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.lt_const(other).map(dense_from).map(Self::Bool) },
            ComplexCompare::lt_const((this.0.into(), this.1.into()), other),
            { this.lt_const(other).map(dense_from).map(Self::Bool) }
        )
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.le_const(other).map(dense_from).map(Self::Bool) },
            ComplexCompare::le_const((this.0.into(), this.1.into()), other),
            { this.le_const(other).map(dense_from).map(Self::Bool) }
        )
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        view_dispatch!(
            self,
            this,
            { this.ne_const(other).map(dense_from).map(Self::Bool) },
            ComplexCompare::ne_const((this.0.into(), this.1.into()), other),
            { this.ne_const(other).map(dense_from).map(Self::Bool) }
        )
    }
}

impl<Txn, FE> TensorConvert for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Dense = Self;
    type Sparse = SparseView<Txn, FE>;

    fn into_dense(self) -> Self::Dense {
        self
    }

    fn into_sparse(self) -> Self::Sparse {
        match self {
            Self::Bool(this) => SparseView::Bool(sparse_from(this.into_sparse())),
            _ => todo!(),
        }
    }
}

impl<Txn, FE> TensorCast for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
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
                        let cast = DenseUnaryCast::new(
                            this.accessor,
                            |block| block.cast(),
                            |n| n.cast_into(),
                        );
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
                    let cast =
                        DenseUnaryCast::new(this.accessor, |block| block.cast(), |n| n.cast_into());
                    Ok(Self::Bool(dense_from(cast.into())))
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

impl<Txn, FE> TensorDiagonal for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Diagonal = Self;

    fn diagonal(self) -> TCResult<Self::Diagonal> {
        match self {
            Self::Bool(this) => this.diagonal().map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.diagonal().map(dense_from)?;
                let im = im.diagonal().map(dense_from)?;
                debug_assert_eq!(re.shape(), im.shape());
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.diagonal().map(dense_from)?;
                let im = im.diagonal().map(dense_from)?;
                debug_assert_eq!(re.shape(), im.shape());
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.diagonal().map(dense_from).map(Self::F32),
            Self::F64(this) => this.diagonal().map(dense_from).map(Self::F64),
            Self::I16(this) => this.diagonal().map(dense_from).map(Self::I16),
            Self::I32(this) => this.diagonal().map(dense_from).map(Self::I32),
            Self::I64(this) => this.diagonal().map(dense_from).map(Self::I64),
            Self::U8(this) => this.diagonal().map(dense_from).map(Self::U8),
            Self::U16(this) => this.diagonal().map(dense_from).map(Self::U16),
            Self::U32(this) => this.diagonal().map(dense_from).map(Self::U32),
            Self::U64(this) => this.diagonal().map(dense_from).map(Self::U64),
        }
    }
}

impl<Txn, FE> TensorMath<Self> for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let this = TensorCast::cast_into(this, dtype)?;
                let that = TensorCast::cast_into(that, dtype)?;
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let this = TensorCast::cast_into(this, dtype)?;
                let that = TensorCast::cast_into(that, dtype)?;
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
                let this = TensorCast::cast_into(this, dtype)?;
                let that = TensorCast::cast_into(that, dtype)?;
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let this = TensorCast::cast_into(this, dtype)?;
                let that = TensorCast::cast_into(that, dtype)?;
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
                Self::C32(this).pow(that)
            }
            (Self::C64((x, y)), Self::F64(that)) => {
                ComplexMath::pow((x.into(), y.into()), that.into()).and_then(Self::complex_from)
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let this = TensorCast::cast_into(this, dtype)?;
                let that = TensorCast::cast_into(that, dtype)?;
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let that = TensorCast::cast_into(that, this.0.dtype())?;
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
                let this = TensorCast::cast_into(this, dtype)?;
                let that = TensorCast::cast_into(that, dtype)?;
                this.sub(that)
            }
        }
    }
}

macro_rules! math_const {
    ($this:ident, $that:ident, $complex:expr, $general:expr) => {
        match $this {
            Self::Bool(this) => $general(this, $that).map(dense_from).map(Self::Bool),
            Self::C32(this) => {
                ($complex)((this.0.into(), this.1.into()), $that).and_then(Self::complex_from)
            }
            Self::C64(this) => {
                ($complex)((this.0.into(), this.1.into()), $that).and_then(Self::complex_from)
            }
            Self::F32(this) => $general(this, $that).map(dense_from).map(Self::F32),
            Self::F64(this) => $general(this, $that).map(dense_from).map(Self::F64),
            Self::I16(this) => $general(this, $that).map(dense_from).map(Self::I16),
            Self::I32(this) => $general(this, $that).map(dense_from).map(Self::I32),
            Self::I64(this) => $general(this, $that).map(dense_from).map(Self::I64),
            Self::U8(this) => $general(this, $that).map(dense_from).map(Self::U8),
            Self::U16(this) => $general(this, $that).map(dense_from).map(Self::U16),
            Self::U32(this) => $general(this, $that).map(dense_from).map(Self::U32),
            Self::U64(this) => $general(this, $that).map(dense_from).map(Self::U64),
        }
    };
}

impl<Txn, FE> TensorMathConst for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
        math_const!(
            self,
            other,
            ComplexMath::add_const,
            TensorMathConst::add_const
        )
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        math_const!(
            self,
            other,
            ComplexMath::div_const,
            TensorMathConst::div_const
        )
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        math_const!(
            self,
            base,
            ComplexMath::log_const,
            TensorMathConst::log_const
        )
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        math_const!(
            self,
            other,
            ComplexMath::mul_const,
            TensorMathConst::mul_const
        )
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        math_const!(
            self,
            other,
            ComplexMath::pow_const,
            TensorMathConst::pow_const
        )
    }

    fn sub_const(self, other: Number) -> TCResult<Self::Combine> {
        math_const!(
            self,
            other,
            ComplexMath::sub_const,
            TensorMathConst::sub_const
        )
    }
}

#[async_trait]
impl<Txn, FE> TensorRead for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.read_value(txn_id, coord).await,
            ComplexRead::read_value((Self::from(this.0), Self::from(this.1)), txn_id, coord).await,
            this.read_value(txn_id, coord).await
        )
    }
}

#[async_trait]
impl<Txn, FE> TensorReduce for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Reduce = Self;

    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::Bool(this) => this.all(txn_id).await,
            Self::C32(this) => {
                let this = TensorCast::cast_into(Self::C32(this), NumberType::Bool)?;
                this.all(txn_id).await
            }
            Self::C64(this) => {
                let this = TensorCast::cast_into(Self::C64(this), NumberType::Bool)?;
                this.all(txn_id).await
            }
            Self::F32(this) => this.all(txn_id).await,
            Self::F64(this) => this.all(txn_id).await,
            Self::I16(this) => this.all(txn_id).await,
            Self::I32(this) => this.all(txn_id).await,
            Self::I64(this) => this.all(txn_id).await,
            Self::U8(this) => this.all(txn_id).await,
            Self::U16(this) => this.all(txn_id).await,
            Self::U32(this) => this.all(txn_id).await,
            Self::U64(this) => this.all(txn_id).await,
        }
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::Bool(this) => this.any(txn_id).await,
            Self::C32(this) => {
                let this = TensorCast::cast_into(Self::C32(this), NumberType::Bool)?;
                this.any(txn_id).await
            }
            Self::C64(this) => {
                let this = TensorCast::cast_into(Self::C64(this), NumberType::Bool)?;
                this.any(txn_id).await
            }
            Self::F32(this) => this.any(txn_id).await,
            Self::F64(this) => this.any(txn_id).await,
            Self::I16(this) => this.any(txn_id).await,
            Self::I32(this) => this.any(txn_id).await,
            Self::I64(this) => this.any(txn_id).await,
            Self::U8(this) => this.any(txn_id).await,
            Self::U16(this) => this.any(txn_id).await,
            Self::U32(this) => this.any(txn_id).await,
            Self::U64(this) => this.any(txn_id).await,
        }
    }

    fn max(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Bool(this) => this.max(axes, keepdims).map(dense_from).map(Self::Bool),
            Self::C32(_) | Self::C64(_) => {
                Err(not_implemented!("maximum value of a complex tensor"))
            }
            Self::F32(this) => this.max(axes, keepdims).map(dense_from).map(Self::F32),
            Self::F64(this) => this.max(axes, keepdims).map(dense_from).map(Self::F64),
            Self::I16(this) => this.max(axes, keepdims).map(dense_from).map(Self::I16),
            Self::I32(this) => this.max(axes, keepdims).map(dense_from).map(Self::I32),
            Self::I64(this) => this.max(axes, keepdims).map(dense_from).map(Self::I64),
            Self::U8(this) => this.max(axes, keepdims).map(dense_from).map(Self::U8),
            Self::U16(this) => this.max(axes, keepdims).map(dense_from).map(Self::U16),
            Self::U32(this) => this.max(axes, keepdims).map(dense_from).map(Self::U32),
            Self::U64(this) => this.max(axes, keepdims).map(dense_from).map(Self::U64),
        }
    }

    async fn max_all(self, txn_id: TxnId) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.max_all(txn_id).await,
            Err(not_implemented!("maximum value of a complex tensor")),
            this.max_all(txn_id).await
        )
    }

    fn min(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Bool(this) => this.min(axes, keepdims).map(dense_from).map(Self::Bool),
            Self::C32(_) | Self::C64(_) => {
                Err(not_implemented!("minimum value of a complex tensor"))
            }
            Self::F32(this) => this.min(axes, keepdims).map(dense_from).map(Self::F32),
            Self::F64(this) => this.min(axes, keepdims).map(dense_from).map(Self::F64),
            Self::I16(this) => this.min(axes, keepdims).map(dense_from).map(Self::I16),
            Self::I32(this) => this.min(axes, keepdims).map(dense_from).map(Self::I32),
            Self::I64(this) => this.min(axes, keepdims).map(dense_from).map(Self::I64),
            Self::U8(this) => this.min(axes, keepdims).map(dense_from).map(Self::U8),
            Self::U16(this) => this.min(axes, keepdims).map(dense_from).map(Self::U16),
            Self::U32(this) => this.min(axes, keepdims).map(dense_from).map(Self::U32),
            Self::U64(this) => this.min(axes, keepdims).map(dense_from).map(Self::U64),
        }
    }

    async fn min_all(self, txn_id: TxnId) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.min_all(txn_id).await,
            Err(not_implemented!("minimum value of a complex tensor")),
            this.min_all(txn_id).await
        )
    }

    fn product(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Bool(this) => this.product(axes, keepdims).map(dense_from).map(Self::Bool),

            Self::C32(_) | Self::C64(_) => Err(not_implemented!("product of a complex tensor")),
            Self::F32(this) => this.product(axes, keepdims).map(dense_from).map(Self::F32),
            Self::F64(this) => this.product(axes, keepdims).map(dense_from).map(Self::F64),
            Self::I16(this) => this.product(axes, keepdims).map(dense_from).map(Self::I16),
            Self::I32(this) => this.product(axes, keepdims).map(dense_from).map(Self::I32),
            Self::I64(this) => this.product(axes, keepdims).map(dense_from).map(Self::I64),
            Self::U8(this) => this.product(axes, keepdims).map(dense_from).map(Self::U8),
            Self::U16(this) => this.product(axes, keepdims).map(dense_from).map(Self::U16),
            Self::U32(this) => this.product(axes, keepdims).map(dense_from).map(Self::U32),
            Self::U64(this) => this.product(axes, keepdims).map(dense_from).map(Self::U64),
        }
    }

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.product_all(txn_id).await,
            Err(not_implemented!("product of a complex tensor")),
            this.product_all(txn_id).await
        )
    }

    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Bool(this) => this.sum(axes, keepdims).map(dense_from).map(Self::Bool),
            Self::C32(_) | Self::C64(_) => Err(not_implemented!("sum of a complex tensor")),
            Self::F32(this) => this.sum(axes, keepdims).map(dense_from).map(Self::F32),
            Self::F64(this) => this.sum(axes, keepdims).map(dense_from).map(Self::F64),
            Self::I16(this) => this.sum(axes, keepdims).map(dense_from).map(Self::I16),
            Self::I32(this) => this.sum(axes, keepdims).map(dense_from).map(Self::I32),
            Self::I64(this) => this.sum(axes, keepdims).map(dense_from).map(Self::I64),
            Self::U8(this) => this.sum(axes, keepdims).map(dense_from).map(Self::U8),
            Self::U16(this) => this.sum(axes, keepdims).map(dense_from).map(Self::U16),
            Self::U32(this) => this.sum(axes, keepdims).map(dense_from).map(Self::U32),
            Self::U64(this) => this.sum(axes, keepdims).map(dense_from).map(Self::U64),
        }
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.sum_all(txn_id).await,
            Err(not_implemented!("sum of a complex tensor")),
            this.sum_all(txn_id).await
        )
    }
}

impl<Txn, FE> TensorTransform for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = Self;
    type Slice = Self;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        match self {
            Self::Bool(this) => this.broadcast(shape).map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.broadcast(shape.clone()).map(dense_from)?;
                let im = im.broadcast(shape).map(dense_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.broadcast(shape.clone()).map(dense_from)?;
                let im = im.broadcast(shape).map(dense_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.broadcast(shape).map(dense_from).map(Self::F32),
            Self::F64(this) => this.broadcast(shape).map(dense_from).map(Self::F64),
            Self::I16(this) => this.broadcast(shape).map(dense_from).map(Self::I16),
            Self::I32(this) => this.broadcast(shape).map(dense_from).map(Self::I32),
            Self::I64(this) => this.broadcast(shape).map(dense_from).map(Self::I64),
            Self::U8(this) => this.broadcast(shape).map(dense_from).map(Self::U8),
            Self::U16(this) => this.broadcast(shape).map(dense_from).map(Self::U16),
            Self::U32(this) => this.broadcast(shape).map(dense_from).map(Self::U32),
            Self::U64(this) => this.broadcast(shape).map(dense_from).map(Self::U64),
        }
    }

    fn expand(self, axes: Axes) -> TCResult<Self::Expand> {
        match self {
            Self::Bool(this) => this.expand(axes).map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.expand(axes.to_vec()).map(dense_from)?;
                let im = im.expand(axes).map(dense_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.expand(axes.to_vec()).map(dense_from)?;
                let im = im.expand(axes).map(dense_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.expand(axes).map(dense_from).map(Self::F32),
            Self::F64(this) => this.expand(axes).map(dense_from).map(Self::F64),
            Self::I16(this) => this.expand(axes).map(dense_from).map(Self::I16),
            Self::I32(this) => this.expand(axes).map(dense_from).map(Self::I32),
            Self::I64(this) => this.expand(axes).map(dense_from).map(Self::I64),
            Self::U8(this) => this.expand(axes).map(dense_from).map(Self::U8),
            Self::U16(this) => this.expand(axes).map(dense_from).map(Self::U16),
            Self::U32(this) => this.expand(axes).map(dense_from).map(Self::U32),
            Self::U64(this) => this.expand(axes).map(dense_from).map(Self::U64),
        }
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        match self {
            Self::Bool(this) => this.reshape(shape).map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.reshape(shape.clone()).map(dense_from)?;
                let im = im.reshape(shape).map(dense_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.reshape(shape.clone()).map(dense_from)?;
                let im = im.reshape(shape).map(dense_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.reshape(shape).map(dense_from).map(Self::F32),
            Self::F64(this) => this.reshape(shape).map(dense_from).map(Self::F64),
            Self::I16(this) => this.reshape(shape).map(dense_from).map(Self::I16),
            Self::I32(this) => this.reshape(shape).map(dense_from).map(Self::I32),
            Self::I64(this) => this.reshape(shape).map(dense_from).map(Self::I64),
            Self::U8(this) => this.reshape(shape).map(dense_from).map(Self::U8),
            Self::U16(this) => this.reshape(shape).map(dense_from).map(Self::U16),
            Self::U32(this) => this.reshape(shape).map(dense_from).map(Self::U32),
            Self::U64(this) => this.reshape(shape).map(dense_from).map(Self::U64),
        }
    }

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        match self {
            Self::Bool(this) => this.slice(range).map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.slice(range.clone()).map(dense_from)?;
                let im = im.slice(range).map(dense_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.slice(range.clone()).map(dense_from)?;
                let im = im.slice(range).map(dense_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.slice(range).map(dense_from).map(Self::F32),
            Self::F64(this) => this.slice(range).map(dense_from).map(Self::F64),
            Self::I16(this) => this.slice(range).map(dense_from).map(Self::I16),
            Self::I32(this) => this.slice(range).map(dense_from).map(Self::I32),
            Self::I64(this) => this.slice(range).map(dense_from).map(Self::I64),
            Self::U8(this) => this.slice(range).map(dense_from).map(Self::U8),
            Self::U16(this) => this.slice(range).map(dense_from).map(Self::U16),
            Self::U32(this) => this.slice(range).map(dense_from).map(Self::U32),
            Self::U64(this) => this.slice(range).map(dense_from).map(Self::U64),
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        match self {
            Self::Bool(this) => this.transpose(permutation).map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.transpose(permutation.clone()).map(dense_from)?;
                let im = im.transpose(permutation).map(dense_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.transpose(permutation.clone()).map(dense_from)?;
                let im = im.transpose(permutation).map(dense_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.transpose(permutation).map(dense_from).map(Self::F32),
            Self::F64(this) => this.transpose(permutation).map(dense_from).map(Self::F64),
            Self::I16(this) => this.transpose(permutation).map(dense_from).map(Self::I16),
            Self::I32(this) => this.transpose(permutation).map(dense_from).map(Self::I32),
            Self::I64(this) => this.transpose(permutation).map(dense_from).map(Self::I64),
            Self::U8(this) => this.transpose(permutation).map(dense_from).map(Self::U8),
            Self::U16(this) => this.transpose(permutation).map(dense_from).map(Self::U16),
            Self::U32(this) => this.transpose(permutation).map(dense_from).map(Self::U32),
            Self::U64(this) => this.transpose(permutation).map(dense_from).map(Self::U64),
        }
    }
}

macro_rules! view_trig {
    ($this:ident, $general32:ident, $general64:ident, $complex:expr) => {
        match $this {
            Self::Bool(this) => {
                let accessor = DenseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(DenseAccess::from(accessor).into()))
            }
            Self::C32((re, im)) => $complex((re.into(), im.into())).and_then(Self::complex_from),
            Self::C64((re, im)) => $complex((re.into(), im.into())).and_then(Self::complex_from),
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

impl<Txn, FE> TensorTrig for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Unary = Self;

    fn asin(self) -> TCResult<Self::Unary> {
        view_trig!(self, asin_f32, asin_f64, |_: (Self, Self)| Err(
            not_implemented!("arcsine of a complex number")
        ))
    }

    fn sin(self) -> TCResult<Self::Unary> {
        view_trig!(self, sin_f32, sin_f64, ComplexTrig::sin)
    }

    fn sinh(self) -> TCResult<Self::Unary> {
        view_trig!(self, sinh_f32, sinh_f64, ComplexTrig::sinh)
    }

    fn acos(self) -> TCResult<Self::Unary> {
        view_trig!(self, acos_f32, acos_f64, |_: (Self, Self)| Err(
            not_implemented!("arccosine of a complex number")
        ))
    }

    fn cos(self) -> TCResult<Self::Unary> {
        view_trig!(self, cos_f32, cos_f64, ComplexTrig::cos)
    }

    fn cosh(self) -> TCResult<Self::Unary> {
        view_trig!(self, cosh_f32, cosh_f64, ComplexTrig::cosh)
    }

    fn atan(self) -> TCResult<Self::Unary> {
        view_trig!(self, atan_f32, atan_f64, |_: (Self, Self)| Err(
            not_implemented!("arctangent of a complex number")
        ))
    }

    fn tan(self) -> TCResult<Self::Unary> {
        view_trig!(self, tan_f32, tan_f64, ComplexTrig::tan)
    }

    fn tanh(self) -> TCResult<Self::Unary> {
        view_trig!(self, tanh_f32, tanh_f64, ComplexTrig::tanh)
    }
}

impl<Txn, FE> TensorUnary for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Unary = Self;

    fn abs(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.abs().map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => ComplexUnary::abs((Self::from(re), Self::from(im))),
            Self::C64((re, im)) => ComplexUnary::abs((Self::from(re), Self::from(im))),
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
            Self::Bool(this) => this.exp().map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => {
                ComplexUnary::exp((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::C64((re, im)) => {
                ComplexUnary::exp((Self::from(re), Self::from(im))).and_then(Self::complex_from)
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
            Self::Bool(this) => this.ln().map(dense_from).map(Self::Bool),
            Self::C32((re, im)) => {
                ComplexUnary::ln((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::C64((re, im)) => {
                ComplexUnary::ln((Self::from(re), Self::from(im))).and_then(Self::complex_from)
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
            Self::C32((re, im)) => {
                ComplexUnary::round((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::C64((re, im)) => {
                ComplexUnary::round((Self::from(re), Self::from(im))).and_then(Self::complex_from)
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

impl<Txn, FE> TensorUnaryBoolean for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Unary = Self;

    fn not(self) -> TCResult<Self::Unary> {
        view_dispatch!(
            self,
            this,
            this.not().map(dense_from).map(Self::Bool),
            ComplexUnary::not((this.0.into(), this.1.into())),
            this.not().map(dense_from).map(Self::Bool)
        )
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> fmt::Debug for DenseView<Txn, FE> {
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
fn atan2<Txn, FE, T: CDatatype>(
    _y: DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>,
    _x: DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>,
) -> TCResult<DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>> {
    Err(not_implemented!("atan2"))
}
