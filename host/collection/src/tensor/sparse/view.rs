use std::fmt;
use std::pin::Pin;

use async_trait::async_trait;
use destream::de;
use futures::{Stream, TryStreamExt};
use ha_ndarray::{Array, Buffer, CType, NDArrayBoolean, NDArrayRead};
use log::trace;
use rayon::prelude::*;
use safecast::{AsType, CastFrom, CastInto};
use smallvec::{smallvec, SmallVec};

use tc_error::*;
use tc_transact::lock::PermitRead;
use tc_transact::{Transaction, TxnId};
use tc_value::{
    Complex, ComplexType, DType, FloatType, IntType, Number, NumberClass, NumberType, UIntType,
};
use tcgeneric::ThreadSafe;

use crate::tensor::complex::{ComplexCompare, ComplexMath, ComplexRead, ComplexTrig, ComplexUnary};
use crate::tensor::dense::{dense_from, DenseCacheFile, DenseView};
use crate::tensor::{
    autoqueue, broadcast_shape, Axes, Coord, Range, Shape, TensorBoolean, TensorBooleanConst,
    TensorCast, TensorCompare, TensorCompareConst, TensorCond, TensorConvert, TensorDiagonal,
    TensorInstance, TensorMatMul, TensorMath, TensorMathConst, TensorPermitRead, TensorRead,
    TensorReduce, TensorTransform, TensorTrig, TensorUnary, TensorUnaryBoolean,
};

use super::{sparse_from, Node, SparseAccess, SparseCombine, SparseTensor, SparseUnaryCast};

type SparseComplex<Txn, FE, T> = (
    SparseTensor<Txn, FE, SparseAccess<Txn, FE, T>>,
    SparseTensor<Txn, FE, SparseAccess<Txn, FE, T>>,
);

pub enum SparseView<Txn, FE> {
    Bool(SparseTensor<Txn, FE, SparseAccess<Txn, FE, u8>>),
    C32(SparseComplex<Txn, FE, f32>),
    C64(SparseComplex<Txn, FE, f64>),
    F32(SparseTensor<Txn, FE, SparseAccess<Txn, FE, f32>>),
    F64(SparseTensor<Txn, FE, SparseAccess<Txn, FE, f64>>),
    I16(SparseTensor<Txn, FE, SparseAccess<Txn, FE, i16>>),
    I32(SparseTensor<Txn, FE, SparseAccess<Txn, FE, i32>>),
    I64(SparseTensor<Txn, FE, SparseAccess<Txn, FE, i64>>),
    U8(SparseTensor<Txn, FE, SparseAccess<Txn, FE, u8>>),
    U16(SparseTensor<Txn, FE, SparseAccess<Txn, FE, u16>>),
    U32(SparseTensor<Txn, FE, SparseAccess<Txn, FE, u32>>),
    U64(SparseTensor<Txn, FE, SparseAccess<Txn, FE, u64>>),
}

impl<Txn: ThreadSafe, FE: ThreadSafe> SparseView<Txn, FE> {
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

impl<Txn, FE> Clone for SparseView<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Bool(this) => Self::Bool(this.clone()),
            Self::C32(this) => Self::C32(this.clone()),
            Self::C64(this) => Self::C64(this.clone()),
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

impl<Txn, FE> SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    pub async fn into_elements(
        self,
        txn_id: TxnId,
    ) -> TCResult<Pin<Box<dyn Stream<Item = TCResult<(Coord, Number)>> + Send>>> {
        view_dispatch!(
            self,
            this,
            {
                let elements = this.into_elements(txn_id).await?;
                let elements = elements.map_ok(|(coord, n)| (coord, (n != 0).into()));
                Ok(Box::pin(elements))
            },
            into_complex_elements(this, txn_id).await,
            {
                let elements = this.into_elements(txn_id).await?;
                let elements = elements.map_ok(|(coord, n)| (coord, n.into()));
                Ok(Box::pin(elements))
            }
        )
    }
}

async fn into_complex_elements<Txn, FE, T>(
    tensors: SparseComplex<Txn, FE, T>,
    txn_id: TxnId,
) -> TCResult<Pin<Box<dyn Stream<Item = TCResult<(Coord, Number)>> + Send>>>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node> + Clone,
    T: CType + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Complex: From<(T, T)>,
    Number: From<T> + CastInto<T>,
{
    let (re, im) = tensors;

    debug_assert_eq!(re.shape(), im.shape());
    let shape = re.shape().clone();
    let ndim = shape.len();

    let blocks = super::access::merge_blocks_outer(
        re.into_inner(),
        im.into_inner(),
        txn_id,
        shape,
        Range::default(),
        Axes::default(),
    )
    .await?;

    let elements = blocks
        .map_ok(move |(coords, (re, im))| async move {
            let queue = autoqueue(&coords)?;

            let coords = coords
                .read(&queue)
                .and_then(|buffer| buffer.to_slice())
                .map(|slice| slice.into_vec())?;

            let re = re
                .read(&queue)
                .and_then(|buffer| buffer.to_slice())
                .map(|slice| slice.into_vec())?;

            let im = im
                .read(&queue)
                .and_then(|buffer| buffer.to_slice())
                .map(|slice| slice.into_vec())?;

            let values = re
                .into_par_iter()
                .zip(im)
                .map(|(r, i)| Number::Complex((r, i).into()));

            let elements = coords
                .par_chunks(ndim)
                .map(Coord::from_slice)
                .zip(values)
                .map(Ok)
                .collect::<Vec<TCResult<_>>>();

            TCResult::Ok(futures::stream::iter(elements))
        })
        .try_buffered(num_cpus::get())
        .try_flatten();

    Ok(Box::pin(elements))
}

impl<Txn: ThreadSafe, FE: ThreadSafe> TensorInstance for SparseView<Txn, FE> {
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

impl<Txn, FE> TensorBoolean<Self> for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
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

impl<Txn, FE> TensorBooleanConst for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        view_dispatch!(
            self,
            this,
            this.and_const(other).map(sparse_from).map(Self::Bool),
            TensorCast::cast_into(Self::from(this), NumberType::Bool)
                .and_then(|this| this.and_const(other)),
            this.and_const(other).map(sparse_from).map(Self::Bool)
        )
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        Err(bad_request!("cannot call OR {} on {:?} because the result would not be sparse (consider converting to a dense tensor first)", other, self))
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        Err(bad_request!("cannot call XOR {} on {:?} because the result would not be sparse (consider converting to a dense tensor first)", other, self))
    }
}

impl<Txn, FE> TensorCast for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Cast = Self;

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        trace!("cast {:?} into {:?}", self, dtype);

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
                    Err(bad_request!("cannot cast {this:?} into a complex tensor")),
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
                    Err(bad_request!("cannot cast {this:?} into a complex tensor"))
                )
            }
            NumberType::Complex(ComplexType::C64) => {
                view_dispatch!(
                    self,
                    this,
                    Err(bad_request!("cannot cast {this:?} into a complex tensor")),
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
                    Err(bad_request!("cannot cast {this:?} into a complex tensor"))
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

impl<Txn, FE> TensorCond<Self, Self> for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Cond = Self;

    fn cond(self, then: Self, or_else: Self) -> TCResult<Self::Cond> {
        let this = if let Self::Bool(this) = self {
            this
        } else {
            let this = TensorCast::cast_into(self, NumberType::Bool)?;
            return this.cond(then, or_else);
        };

        match (then, or_else) {
            (Self::Bool(then), Self::Bool(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::Bool),

            (Self::C32((then_re, then_im)), Self::C32((else_re, else_im))) => {
                let re = this.clone().cond(then_re, else_re)?;
                let im = this.cond(then_im, else_im)?;
                Ok(SparseView::C32((sparse_from(re), sparse_from(im))))
            }

            (Self::C64((then_re, then_im)), Self::C64((else_re, else_im))) => {
                let re = this.clone().cond(then_re, else_re)?;
                let im = this.cond(then_im, else_im)?;
                Ok(SparseView::C64((sparse_from(re), sparse_from(im))))
            }

            (Self::F32(then), Self::F32(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::F32),

            (Self::F64(then), Self::F64(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::F64),

            (Self::I16(then), Self::I16(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::I16),

            (Self::I32(then), Self::I32(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::I32),

            (Self::I64(then), Self::I64(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::I64),

            (Self::U8(then), Self::U8(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::U8),

            (Self::U16(then), Self::U16(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::U16),

            (Self::U32(then), Self::U32(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::U32),

            (Self::U64(then), Self::U64(or_else)) => this
                .cond(then, or_else)
                .map(sparse_from)
                .map(SparseView::U64),

            (then, or_else) if then.dtype() < or_else.dtype() => {
                let then = TensorCast::cast_into(then, or_else.dtype())?;
                SparseView::Bool(this).cond(then, or_else)
            }

            (then, or_else) => {
                let or_else = TensorCast::cast_into(or_else, then.dtype())?;
                SparseView::Bool(this).cond(then, or_else)
            }
        }
    }
}

impl<Txn, FE> TensorConvert for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Dense = DenseView<Txn, FE>;
    type Sparse = Self;

    fn into_dense(self) -> Self::Dense {
        match self {
            Self::Bool(this) => DenseView::Bool(dense_from(this.into_dense())),
            Self::C32((re, im)) => {
                let re = dense_from(re.into_dense());
                let im = dense_from(im.into_dense());
                DenseView::C32((re, im))
            }
            Self::C64((re, im)) => {
                let re = dense_from(re.into_dense());
                let im = dense_from(im.into_dense());
                DenseView::C64((re, im))
            }
            Self::F32(this) => DenseView::F32(dense_from(this.into_dense())),
            Self::F64(this) => DenseView::F64(dense_from(this.into_dense())),
            Self::I16(this) => DenseView::I16(dense_from(this.into_dense())),
            Self::I32(this) => DenseView::I32(dense_from(this.into_dense())),
            Self::I64(this) => DenseView::I64(dense_from(this.into_dense())),
            Self::U8(this) => DenseView::U8(dense_from(this.into_dense())),
            Self::U16(this) => DenseView::U16(dense_from(this.into_dense())),
            Self::U32(this) => DenseView::U32(dense_from(this.into_dense())),
            Self::U64(this) => DenseView::U64(dense_from(this.into_dense())),
        }
    }

    fn into_sparse(self) -> Self::Sparse {
        self
    }
}

macro_rules! view_compare {
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

impl<Txn, FE> TensorCompare<Self> for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
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

impl<Txn, FE> TensorCompareConst for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Compare = Self;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        if bool::cast_from(other) {
            view_dispatch!(
                self,
                this,
                this.eq_const(other).map(sparse_from).map(Self::Bool),
                ComplexCompare::eq_const(
                    (SparseView::from(this.0), SparseView::from(this.1)),
                    other
                ),
                this.eq_const(other).map(sparse_from).map(Self::Bool)
            )
        } else {
            Err(bad_request!("cannot calculate {:?} == {} because the result would not be sparse (consider converting to a dense tensor first)", self, other))
        }
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        if other.ge(&self.dtype().zero().into()) {
            view_dispatch!(
                self,
                this,
                this.gt_const(other).map(sparse_from).map(Self::Bool),
                ComplexCompare::gt_const(
                    (SparseView::from(this.0), SparseView::from(this.1)),
                    other
                ),
                this.gt_const(other).map(sparse_from).map(Self::Bool)
            )
        } else {
            Err(bad_request!("cannot calculate {:?} > {} because the result would not be sparse (consider converting to a dense tensor first)", self, other))
        }
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        if other.gt(&self.dtype().zero().into()) {
            view_dispatch!(
                self,
                this,
                this.ge_const(other).map(sparse_from).map(Self::Bool),
                ComplexCompare::ge_const(
                    (SparseView::from(this.0), SparseView::from(this.1)),
                    other
                ),
                this.ge_const(other).map(sparse_from).map(Self::Bool)
            )
        } else {
            Err(bad_request!("cannot calculate {:?} >= {} because the result would not be sparse (consider converting to a dense tensor first)", self, other))
        }
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        if other.le(&self.dtype().zero().into()) {
            view_dispatch!(
                self,
                this,
                this.lt_const(other).map(sparse_from).map(Self::Bool),
                ComplexCompare::lt_const(
                    (SparseView::from(this.0), SparseView::from(this.1)),
                    other
                ),
                this.lt_const(other).map(sparse_from).map(Self::Bool)
            )
        } else {
            Err(bad_request!("cannot calculate {:?} < {} because the result would not be sparse (consider converting to a dense tensor first)", self, other))
        }
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        if other.lt(&self.dtype().zero().into()) {
            view_dispatch!(
                self,
                this,
                this.le_const(other).map(sparse_from).map(Self::Bool),
                ComplexCompare::le_const(
                    (SparseView::from(this.0), SparseView::from(this.1)),
                    other
                ),
                this.le_const(other).map(sparse_from).map(Self::Bool)
            )
        } else {
            Err(bad_request!("cannot calculate {:?} <= {} because the result would not be sparse (consider converting to a dense tensor first)", self, other))
        }
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        Err(bad_request!("cannot calculate {:?} != {} because the result would not be sparse (consider converting to a dense tensor first)", self, other))
    }
}

impl<Txn, FE> TensorDiagonal for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Diagonal = Self;

    fn diagonal(self) -> TCResult<Self::Diagonal> {
        Err(not_implemented!("diagonal of a sparse tensor"))
    }
}

impl<Txn, FE> TensorMath<Self> for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Combine = Self;
    type LeftCombine = Self;

    fn add(self, other: Self) -> TCResult<Self::Combine> {
        match (self, other) {
            (Self::Bool(this), Self::Bool(that)) => this.or(that).map(sparse_from).map(Self::Bool),
            (Self::C32((a, b)), Self::C32((c, d))) => {
                ComplexMath::add((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C32((re, im)), Self::F32(that)) => {
                ComplexMath::add_real((re.into(), im.into()), that.into())
                    .and_then(Self::complex_from)
            }
            (Self::C32(this), that) if that.dtype().is_real() => {
                let that = TensorCast::cast_into(that, FloatType::F32.into())?;
                Self::C32(this).add(that)
            }
            (Self::C64((a, b)), Self::C64((c, d))) => {
                ComplexMath::add((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C64((re, im)), Self::F64(that)) => {
                ComplexMath::add_real((re.into(), im.into()), that.into())
                    .and_then(Self::complex_from)
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = TensorCast::cast_into(that, FloatType::F64.into())?;
                Self::C64(this).add(that)
            }
            (Self::F32(this), Self::F32(that)) => this.add(that).map(sparse_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.add(that).map(sparse_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.add(that).map(sparse_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.add(that).map(sparse_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.add(that).map(sparse_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.add(that).map(sparse_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.add(that).map(sparse_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.add(that).map(sparse_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.add(that).map(sparse_from).map(Self::U64),
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
            (Self::Bool(this), Self::Bool(that)) => this.div(that).map(sparse_from).map(Self::Bool),
            (Self::C32((a, b)), Self::C32((c, d))) => {
                ComplexMath::div((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C32((re, im)), Self::F32(that)) => {
                ComplexMath::div_real((re.into(), im.into()), that.into())
                    .and_then(Self::complex_from)
            }
            (Self::C32(this), that) if that.dtype().is_real() => {
                let that = TensorCast::cast_into(that, this.0.dtype())?;
                Self::C32(this).div(that)
            }
            (Self::C64((a, b)), Self::C64((c, d))) => {
                ComplexMath::div((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C64((re, im)), Self::F64(that)) => {
                ComplexMath::div_real((re.into(), im.into()), that.into())
                    .and_then(Self::complex_from)
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = TensorCast::cast_into(that, this.0.dtype())?;
                Self::C64(this).div(that)
            }
            (Self::F32(this), Self::F32(that)) => this.div(that).map(sparse_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.div(that).map(sparse_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.div(that).map(sparse_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.div(that).map(sparse_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.div(that).map(sparse_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.div(that).map(sparse_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.div(that).map(sparse_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.div(that).map(sparse_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.div(that).map(sparse_from).map(Self::U64),
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
            (Self::F32(this), Self::F32(that)) => this.log(that).map(sparse_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.log(that).map(sparse_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.log(that).map(sparse_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.log(that).map(sparse_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.log(that).map(sparse_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.log(that).map(sparse_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.log(that).map(sparse_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.log(that).map(sparse_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.log(that).map(sparse_from).map(Self::U64),
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
            (Self::Bool(this), Self::Bool(that)) => this.mul(that).map(sparse_from).map(Self::Bool),
            (Self::C32((a, b)), Self::C32((c, d))) => {
                ComplexMath::mul((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C32((re, im)), Self::F32(that)) => {
                ComplexMath::mul_real((re.into(), im.into()), that.into())
                    .and_then(Self::complex_from)
            }
            (Self::C32(this), that) if that.dtype().is_real() => {
                let that = TensorCast::cast_into(that, this.0.dtype())?;
                Self::C32(this).mul(that)
            }
            (Self::C64((a, b)), Self::C64((c, d))) => {
                ComplexMath::mul((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C64((re, im)), Self::F64(that)) => {
                ComplexMath::mul_real((re.into(), im.into()), that.into())
                    .and_then(Self::complex_from)
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = TensorCast::cast_into(that, this.0.dtype())?;
                Self::C64(this).mul(that)
            }
            (Self::F32(this), Self::F32(that)) => this.mul(that).map(sparse_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.mul(that).map(sparse_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.mul(that).map(sparse_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.mul(that).map(sparse_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.mul(that).map(sparse_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.mul(that).map(sparse_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.mul(that).map(sparse_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.mul(that).map(sparse_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.mul(that).map(sparse_from).map(Self::U64),
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
            (Self::Bool(this), Self::Bool(that)) => this.pow(that).map(sparse_from).map(Self::Bool),
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
            (Self::F32(this), Self::F32(that)) => this.pow(that).map(sparse_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.pow(that).map(sparse_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.pow(that).map(sparse_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.pow(that).map(sparse_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.pow(that).map(sparse_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.pow(that).map(sparse_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.pow(that).map(sparse_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.pow(that).map(sparse_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.pow(that).map(sparse_from).map(Self::U64),
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
            (Self::Bool(this), Self::Bool(that)) => this.or(that).map(sparse_from).map(Self::Bool),
            (Self::C32((a, b)), Self::C32((c, d))) => {
                ComplexMath::sub((a.into(), b.into()), (c.into(), d.into()))
                    .and_then(Self::complex_from)
            }
            (Self::C32((a, b)), Self::F32(that)) => {
                let real = a.sub(that.clone()).map(sparse_from)?;
                let imag = b.sub(that).map(sparse_from)?;
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
                let real = a.sub(that.clone()).map(sparse_from)?;
                let imag = b.sub(that).map(sparse_from)?;
                Ok(Self::C64((real, imag)))
            }
            (Self::C64(this), that) if that.dtype().is_real() => {
                let that = TensorCast::cast_into(that, this.0.dtype())?;
                Self::C64(this).sub(that)
            }
            (Self::F32(this), Self::F32(that)) => this.sub(that).map(sparse_from).map(Self::F32),
            (Self::F64(this), Self::F64(that)) => this.sub(that).map(sparse_from).map(Self::F64),
            (Self::I16(this), Self::I16(that)) => this.sub(that).map(sparse_from).map(Self::I16),
            (Self::I32(this), Self::I32(that)) => this.sub(that).map(sparse_from).map(Self::I32),
            (Self::I64(this), Self::I64(that)) => this.sub(that).map(sparse_from).map(Self::I64),
            (Self::U8(this), Self::U8(that)) => this.sub(that).map(sparse_from).map(Self::U8),
            (Self::U16(this), Self::U16(that)) => this.sub(that).map(sparse_from).map(Self::U16),
            (Self::U32(this), Self::U32(that)) => this.sub(that).map(sparse_from).map(Self::U32),
            (Self::U64(this), Self::U64(that)) => this.sub(that).map(sparse_from).map(Self::U64),
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
            Self::Bool(this) => $general(this, $that).map(sparse_from).map(Self::Bool),
            Self::C32(this) => {
                ($complex)((this.0.into(), this.1.into()), $that).and_then(Self::complex_from)
            }
            Self::C64(this) => {
                ($complex)((this.0.into(), this.1.into()), $that).and_then(Self::complex_from)
            }
            Self::F32(this) => $general(this, $that).map(sparse_from).map(Self::F32),
            Self::F64(this) => $general(this, $that).map(sparse_from).map(Self::F64),
            Self::I16(this) => $general(this, $that).map(sparse_from).map(Self::I16),
            Self::I32(this) => $general(this, $that).map(sparse_from).map(Self::I32),
            Self::I64(this) => $general(this, $that).map(sparse_from).map(Self::I64),
            Self::U8(this) => $general(this, $that).map(sparse_from).map(Self::U8),
            Self::U16(this) => $general(this, $that).map(sparse_from).map(Self::U16),
            Self::U32(this) => $general(this, $that).map(sparse_from).map(Self::U32),
            Self::U64(this) => $general(this, $that).map(sparse_from).map(Self::U64),
        }
    };
}

impl<Txn, FE> TensorMathConst for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Combine = Self;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
        Err(bad_request!("cannot add {} to {:?} because the result would not be sparse (consider converting to a dense tensor first)", other, self))
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
        Err(bad_request!("cannot subtract {} from {:?} because the result would not be sparse (consider converting to a dense tensor first)", other, self))
    }
}

impl<Txn, FE> TensorMatMul<Self> for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type MatMul = Self;

    fn matmul(self, other: Self) -> TCResult<Self::MatMul> {
        assert_eq!(self.ndim(), other.ndim());

        // example shapes: [2, 3] @ [3, 180]
        let ndim = self.ndim(); // 2
        let this = self.expand(smallvec![ndim])?; // [2, 3] -> [2, 3, 1]
        let that = other.expand(smallvec![ndim - 2])?; // [3, 180] -> [1, 3, 180]

        trace!("{:?} @ {:?}", this, that);

        let shape = broadcast_shape(this.shape(), that.shape())?;
        let this = this.broadcast(shape.clone())?;
        let that = that.broadcast(shape)?;

        trace!("outer product is {:?} * {:?}", this, that);

        let outer_product = this.mul(that)?; // [2, 3, 180]
        let reduce_axis = outer_product.ndim() - 2;
        outer_product.sum(smallvec![reduce_axis], false) // [2, 3, 180] -> [2, 180]
    }
}

#[async_trait]
impl<Txn: Transaction<FE>, FE: DenseCacheFile + AsType<Node>> TensorRead for SparseView<Txn, FE> {
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
impl<Txn: Transaction<FE>, FE: DenseCacheFile + AsType<Node>> TensorReduce for SparseView<Txn, FE> {
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
            Self::Bool(this) => this.max(axes, keepdims).map(sparse_from).map(Self::Bool),
            Self::C32(_) | Self::C64(_) => {
                Err(not_implemented!("maximum value of a complex tensor"))
            }
            Self::F32(this) => this.max(axes, keepdims).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.max(axes, keepdims).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.max(axes, keepdims).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.max(axes, keepdims).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.max(axes, keepdims).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.max(axes, keepdims).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.max(axes, keepdims).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.max(axes, keepdims).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.max(axes, keepdims).map(sparse_from).map(Self::U64),
        }
    }

    async fn max_all(self, txn_id: TxnId) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.max_all(txn_id).await,
            Err(not_implemented!(
                "maximum value of a complex tensor {this:?}"
            )),
            this.max_all(txn_id).await
        )
    }

    fn min(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Bool(this) => this.min(axes, keepdims).map(sparse_from).map(Self::Bool),
            Self::C32(_) | Self::C64(_) => {
                Err(not_implemented!("minimum value of a complex tensor"))
            }
            Self::F32(this) => this.min(axes, keepdims).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.min(axes, keepdims).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.min(axes, keepdims).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.min(axes, keepdims).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.min(axes, keepdims).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.min(axes, keepdims).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.min(axes, keepdims).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.min(axes, keepdims).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.min(axes, keepdims).map(sparse_from).map(Self::U64),
        }
    }

    async fn min_all(self, txn_id: TxnId) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.min_all(txn_id).await,
            Err(not_implemented!(
                "minimum value of a complex tensor {this:?}"
            )),
            this.min_all(txn_id).await
        )
    }

    fn product(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Bool(this) => this
                .product(axes, keepdims)
                .map(sparse_from)
                .map(Self::Bool),

            Self::C32(_) | Self::C64(_) => Err(not_implemented!("product of a complex tensor")),
            Self::F32(this) => this.product(axes, keepdims).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.product(axes, keepdims).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.product(axes, keepdims).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.product(axes, keepdims).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.product(axes, keepdims).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.product(axes, keepdims).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.product(axes, keepdims).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.product(axes, keepdims).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.product(axes, keepdims).map(sparse_from).map(Self::U64),
        }
    }

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.product_all(txn_id).await,
            Err(not_implemented!("product of a complex tensor {this:?}")),
            this.product_all(txn_id).await
        )
    }

    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Bool(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::Bool),
            Self::C32(_) | Self::C64(_) => Err(not_implemented!("sum of a complex tensor")),
            Self::F32(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.sum(axes, keepdims).map(sparse_from).map(Self::U64),
        }
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        view_dispatch!(
            self,
            this,
            this.sum_all(txn_id).await,
            Err(not_implemented!("sum of a complex tensor {this:?}")),
            this.sum_all(txn_id).await
        )
    }
}

impl<Txn, FE> TensorTransform for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = Self;
    type Slice = Self;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        if self.shape() == &shape {
            return Ok(self);
        }

        match self {
            Self::Bool(this) => this.broadcast(shape).map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.broadcast(shape.clone()).map(sparse_from)?;
                let im = im.broadcast(shape).map(sparse_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.broadcast(shape.clone()).map(sparse_from)?;
                let im = im.broadcast(shape).map(sparse_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.broadcast(shape).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.broadcast(shape).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.broadcast(shape).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.broadcast(shape).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.broadcast(shape).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.broadcast(shape).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.broadcast(shape).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.broadcast(shape).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.broadcast(shape).map(sparse_from).map(Self::U64),
        }
    }

    fn expand(self, axes: Axes) -> TCResult<Self::Expand> {
        if axes.is_empty() {
            return Ok(self);
        }

        match self {
            Self::Bool(this) => this.expand(axes).map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.expand(axes.clone()).map(sparse_from)?;
                let im = im.expand(axes).map(sparse_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.expand(axes.clone()).map(sparse_from)?;
                let im = im.expand(axes).map(sparse_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.expand(axes).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.expand(axes).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.expand(axes).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.expand(axes).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.expand(axes).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.expand(axes).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.expand(axes).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.expand(axes).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.expand(axes).map(sparse_from).map(Self::U64),
        }
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        if self.shape() == &shape {
            return Ok(self);
        }

        match self {
            Self::Bool(this) => this.reshape(shape).map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.reshape(shape.clone()).map(sparse_from)?;
                let im = im.reshape(shape).map(sparse_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.reshape(shape.clone()).map(sparse_from)?;
                let im = im.reshape(shape).map(sparse_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.reshape(shape).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.reshape(shape).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.reshape(shape).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.reshape(shape).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.reshape(shape).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.reshape(shape).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.reshape(shape).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.reshape(shape).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.reshape(shape).map(sparse_from).map(Self::U64),
        }
    }

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        if self.shape().is_covered_by(&range) {
            trace!("no need to slice {self:?} by {range:?}");
            return Ok(self);
        }

        match self {
            Self::Bool(this) => this.slice(range).map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.slice(range.clone()).map(sparse_from)?;
                let im = im.slice(range).map(sparse_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.slice(range.clone()).map(sparse_from)?;
                let im = im.slice(range).map(sparse_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.slice(range).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.slice(range).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.slice(range).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.slice(range).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.slice(range).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.slice(range).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.slice(range).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.slice(range).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.slice(range).map(sparse_from).map(Self::U64),
        }
    }

    fn transpose(self, permutation: Option<Axes>) -> TCResult<Self::Transpose> {
        if let Some(permutation) = &permutation {
            if permutation.len() == self.ndim()
                && permutation.iter().copied().enumerate().all(|(i, x)| i == x)
            {
                return Ok(self);
            }
        }

        match self {
            Self::Bool(this) => this.transpose(permutation).map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => {
                let re = re.transpose(permutation.clone()).map(sparse_from)?;
                let im = im.transpose(permutation).map(sparse_from)?;
                Ok(Self::C32((re, im)))
            }
            Self::C64((re, im)) => {
                let re = re.transpose(permutation.clone()).map(sparse_from)?;
                let im = im.transpose(permutation).map(sparse_from)?;
                Ok(Self::C64((re, im)))
            }
            Self::F32(this) => this.transpose(permutation).map(sparse_from).map(Self::F32),
            Self::F64(this) => this.transpose(permutation).map(sparse_from).map(Self::F64),
            Self::I16(this) => this.transpose(permutation).map(sparse_from).map(Self::I16),
            Self::I32(this) => this.transpose(permutation).map(sparse_from).map(Self::I32),
            Self::I64(this) => this.transpose(permutation).map(sparse_from).map(Self::I64),
            Self::U8(this) => this.transpose(permutation).map(sparse_from).map(Self::U8),
            Self::U16(this) => this.transpose(permutation).map(sparse_from).map(Self::U16),
            Self::U32(this) => this.transpose(permutation).map(sparse_from).map(Self::U32),
            Self::U64(this) => this.transpose(permutation).map(sparse_from).map(Self::U64),
        }
    }
}

macro_rules! view_trig {
    ($this:ident, $general32:ident, $general64:ident, $complex:expr) => {
        match $this {
            Self::Bool(this) => {
                let accessor = SparseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(SparseAccess::from(accessor).into()))
            }
            Self::C32((re, im)) => $complex((re.into(), im.into())).and_then(Self::complex_from),
            Self::C64((re, im)) => $complex((re.into(), im.into())).and_then(Self::complex_from),
            Self::F32(this) => {
                let accessor = SparseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(SparseAccess::from(accessor).into()))
            }
            Self::F64(this) => {
                let accessor = SparseUnaryCast::$general64(this.into_inner());
                Ok(Self::F64(SparseAccess::from(accessor).into()))
            }
            Self::I16(this) => {
                let accessor = SparseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(SparseAccess::from(accessor).into()))
            }
            Self::I32(this) => {
                let accessor = SparseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(SparseAccess::from(accessor).into()))
            }
            Self::I64(this) => {
                let accessor = SparseUnaryCast::$general64(this.into_inner());
                Ok(Self::F64(SparseAccess::from(accessor).into()))
            }
            Self::U8(this) => {
                let accessor = SparseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(SparseAccess::from(accessor).into()))
            }
            Self::U16(this) => {
                let accessor = SparseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(SparseAccess::from(accessor).into()))
            }
            Self::U32(this) => {
                let accessor = SparseUnaryCast::$general32(this.into_inner());
                Ok(Self::F32(SparseAccess::from(accessor).into()))
            }
            Self::U64(this) => {
                let accessor = SparseUnaryCast::$general64(this.into_inner());
                Ok(Self::F64(SparseAccess::from(accessor).into()))
            }
        }
    };
}

impl<Txn: Transaction<FE>, FE: DenseCacheFile + AsType<Node>> TensorTrig for SparseView<Txn, FE> {
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

impl<Txn, FE> TensorUnary for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Unary = Self;

    fn abs(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.abs().map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => ComplexUnary::abs((Self::from(re), Self::from(im))),
            Self::C64((re, im)) => ComplexUnary::abs((Self::from(re), Self::from(im))),
            Self::F32(this) => this.abs().map(sparse_from).map(Self::F32),
            Self::F64(this) => this.abs().map(sparse_from).map(Self::F64),
            Self::I16(this) => this.abs().map(sparse_from).map(Self::I16),
            Self::I32(this) => this.abs().map(sparse_from).map(Self::I32),
            Self::I64(this) => this.abs().map(sparse_from).map(Self::I64),
            Self::U8(this) => this.abs().map(sparse_from).map(Self::U8),
            Self::U16(this) => this.abs().map(sparse_from).map(Self::U16),
            Self::U32(this) => this.abs().map(sparse_from).map(Self::U32),
            Self::U64(this) => this.abs().map(sparse_from).map(Self::U64),
        }
    }

    fn exp(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.exp().map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => {
                ComplexUnary::exp((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::C64((re, im)) => {
                ComplexUnary::exp((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::F32(this) => this.exp().map(sparse_from).map(Self::F32),
            Self::F64(this) => this.exp().map(sparse_from).map(Self::F64),
            Self::I16(this) => this.exp().map(sparse_from).map(Self::I16),
            Self::I32(this) => this.exp().map(sparse_from).map(Self::I32),
            Self::I64(this) => this.exp().map(sparse_from).map(Self::I64),
            Self::U8(this) => this.exp().map(sparse_from).map(Self::U8),
            Self::U16(this) => this.exp().map(sparse_from).map(Self::U16),
            Self::U32(this) => this.exp().map(sparse_from).map(Self::U32),
            Self::U64(this) => this.exp().map(sparse_from).map(Self::U64),
        }
    }

    fn ln(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.ln().map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => {
                ComplexUnary::ln((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::C64((re, im)) => {
                ComplexUnary::ln((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::F32(this) => this.ln().map(sparse_from).map(Self::F32),
            Self::F64(this) => this.ln().map(sparse_from).map(Self::F64),
            Self::I16(this) => this.ln().map(sparse_from).map(Self::I16),
            Self::I32(this) => this.ln().map(sparse_from).map(Self::I32),
            Self::I64(this) => this.ln().map(sparse_from).map(Self::I64),
            Self::U8(this) => this.ln().map(sparse_from).map(Self::U8),
            Self::U16(this) => this.ln().map(sparse_from).map(Self::U16),
            Self::U32(this) => this.ln().map(sparse_from).map(Self::U32),
            Self::U64(this) => this.ln().map(sparse_from).map(Self::U64),
        }
    }

    fn round(self) -> TCResult<Self::Unary> {
        match self {
            Self::Bool(this) => this.round().map(sparse_from).map(Self::Bool),
            Self::C32((re, im)) => {
                ComplexUnary::round((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::C64((re, im)) => {
                ComplexUnary::round((Self::from(re), Self::from(im))).and_then(Self::complex_from)
            }
            Self::F32(this) => this.round().map(sparse_from).map(Self::F32),
            Self::F64(this) => this.round().map(sparse_from).map(Self::F64),
            Self::I16(this) => this.round().map(sparse_from).map(Self::I16),
            Self::I32(this) => this.round().map(sparse_from).map(Self::I32),
            Self::I64(this) => this.round().map(sparse_from).map(Self::I64),
            Self::U8(this) => this.round().map(sparse_from).map(Self::U8),
            Self::U16(this) => this.round().map(sparse_from).map(Self::U16),
            Self::U32(this) => this.round().map(sparse_from).map(Self::U32),
            Self::U64(this) => this.round().map(sparse_from).map(Self::U64),
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> TensorUnaryBoolean for SparseView<Txn, FE> {
    type Unary = Self;

    fn not(self) -> TCResult<Self::Unary> {
        Err(bad_request!("a sparse tensor does not support the logical not operation because the result would be dense (consider converting to a dense tensor first)"))
    }
}

#[async_trait]
impl<Txn: ThreadSafe, FE: ThreadSafe> TensorPermitRead for SparseView<Txn, FE> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        view_dispatch!(
            self,
            this,
            this.accessor.read_permit(txn_id, range).await,
            {
                // always acquire these permits in-order to avoid the risk of a deadlock
                let mut re = this.0.accessor.read_permit(txn_id, range.clone()).await?;
                let im = this.1.accessor.read_permit(txn_id, range).await?;
                re.extend(im);
                Ok(re)
            },
            this.accessor.read_permit(txn_id, range).await
        )
    }
}

impl<Txn, FE> From<SparseTensor<Txn, FE, SparseAccess<Txn, FE, f32>>> for SparseView<Txn, FE> {
    fn from(tensor: SparseTensor<Txn, FE, SparseAccess<Txn, FE, f32>>) -> Self {
        Self::F32(tensor)
    }
}

impl<Txn, FE> From<SparseTensor<Txn, FE, SparseAccess<Txn, FE, f64>>> for SparseView<Txn, FE> {
    fn from(tensor: SparseTensor<Txn, FE, SparseAccess<Txn, FE, f64>>) -> Self {
        Self::F64(tensor)
    }
}

impl<Txn, FE> From<SparseComplex<Txn, FE, f32>> for SparseView<Txn, FE> {
    fn from(tensors: SparseComplex<Txn, FE, f32>) -> Self {
        Self::C32(tensors)
    }
}

impl<Txn, FE> From<SparseComplex<Txn, FE, f64>> for SparseView<Txn, FE> {
    fn from(tensors: SparseComplex<Txn, FE, f64>) -> Self {
        Self::C64(tensors)
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> fmt::Debug for SparseView<Txn, FE> {
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
fn expect_bool<Txn, FE>(
    view: SparseView<Txn, FE>,
) -> TCResult<SparseTensor<Txn, FE, SparseAccess<Txn, FE, u8>>>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    match TensorCast::cast_into(view, NumberType::Bool)? {
        SparseView::Bool(that) => Ok(that),
        _ => unreachable!("failed to cast sparse tensor into boolean"),
    }
}
