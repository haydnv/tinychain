use async_trait::async_trait;
use futures::try_join;
use safecast::{AsType, CastFrom};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::{Complex, Float, Number, NumberClass, NumberInstance};

use super::dense::{DenseBase, DenseCacheFile, DenseView};
use super::sparse::{Node, SparseBase, SparseView};
use super::{
    Coord, TensorBoolean, TensorCompare, TensorCompareConst, TensorInstance, TensorMath,
    TensorMathConst, TensorRead, TensorTrig, TensorUnary, TensorUnaryBoolean,
};

#[async_trait]
pub(crate) trait ComplexRead: TensorInstance + TensorRead {
    async fn read_value(this: (Self, Self), txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        let (re, im) = this;
        let (re, im) = try_join!(
            re.read_value(txn_id, coord.to_vec()),
            im.read_value(txn_id, coord)
        )?;

        let re = Float::cast_from(re);
        let im = Float::cast_from(im);
        Ok(Number::Complex((re, im).into()))
    }
}

impl<Txn: Transaction<FE>, FE: DenseCacheFile + AsType<Node>> ComplexRead for DenseBase<Txn, FE> {}
impl<Txn, FE> ComplexRead for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
}

impl<Txn, FE> ComplexRead for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
}

impl<Txn, FE> ComplexRead for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
}

pub(crate) trait ComplexUnary:
    TensorInstance
    + TensorBoolean<Self, Combine = Self, LeftCombine = Self>
    + TensorMath<Self, Combine = Self, LeftCombine = Self>
    + TensorMathConst<Combine = Self>
    + TensorTrig<Unary = Self>
    + TensorUnary<Unary = Self>
    + TensorUnaryBoolean<Unary = Self>
    + Clone
{
    fn abs(this: (Self, Self)) -> TCResult<Self> {
        let (real, imag) = this;
        real.pow_const(2.into())?
            .add(imag.pow_const(2.into())?)?
            .pow_const((0.5).into())
    }

    fn exp(_this: (Self, Self)) -> TCResult<(Self, Self)> {
        Err(not_implemented!("complex exponentiation"))
    }

    fn ln(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let (x, y) = this.clone();
        let r = ComplexUnary::abs(this)?;
        let real = r.ln()?;
        let imag = atan2(y, x)?;
        Ok((real, imag))
    }

    fn not(this: (Self, Self)) -> TCResult<Self> {
        let (x, y) = this;
        (x.or(y)?).not()
    }

    fn round(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let real = this.0.round()?;
        let imag = this.1.round()?;
        Ok((real, imag))
    }
}

impl<Txn, FE> ComplexUnary for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
}

impl<Txn, FE> ComplexUnary for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
}

pub(crate) trait ComplexCompare:
    ComplexUnary + TensorCompare<Self, Compare = Self> + TensorCompareConst<Compare = Self>
{
    fn eq(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let (this_r, this_i) = this;
        let (that_r, that_i) = that;

        let real = this_r.eq(that_r)?;
        let imag = this_i.eq(that_i)?;

        real.and(imag).map_err(TCError::from)
    }

    fn eq_const(this: (Self, Self), that: Number) -> TCResult<Self> {
        let (lr, li) = this;

        if let Number::Complex(that) = that {
            let (rr, ri) = that.into();
            let real = lr.eq_const(rr.into())?;
            let imag = li.eq_const(ri.into())?;
            real.and(imag)
        } else {
            let real = lr.eq_const(that)?;
            let imag = li.not()?;
            real.and(imag)
        }
    }

    fn gt(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let this = ComplexUnary::abs(this)?;
        let that = ComplexUnary::abs(that)?;
        this.gt(that)
    }

    fn gt_const(this: (Self, Self), that: Number) -> TCResult<Self> {
        let this = ComplexUnary::abs(this)?;
        let that = if that.class().is_complex() {
            that.abs()
        } else {
            that
        };

        this.gt_const(that)
    }

    fn ge(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let this = ComplexUnary::abs(this)?;
        let that = ComplexUnary::abs(that)?;
        this.ge(that)
    }

    fn ge_const(this: (Self, Self), that: Number) -> TCResult<Self> {
        let this = ComplexUnary::abs(this)?;
        let that = if that.class().is_complex() {
            that.abs()
        } else {
            that
        };

        this.ge_const(that)
    }

    fn lt(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let this = ComplexUnary::abs(this)?;
        let that = ComplexUnary::abs(that)?;
        this.lt(that)
    }

    fn lt_const(this: (Self, Self), that: Number) -> TCResult<Self> {
        let this = ComplexUnary::abs(this)?;
        let that = if that.class().is_complex() {
            that.abs()
        } else {
            that
        };

        this.lt_const(that)
    }

    fn le(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let this = ComplexUnary::abs(this)?;
        let that = ComplexUnary::abs(that)?;
        this.le(that)
    }

    fn le_const(this: (Self, Self), that: Number) -> TCResult<Self> {
        let this = ComplexUnary::abs(this)?;
        let that = if that.class().is_complex() {
            that.abs()
        } else {
            that
        };

        this.le_const(that)
    }

    fn ne(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let (this_r, this_i) = this;
        let (that_r, that_i) = that;

        let real = this_r.ne(that_r)?;
        let imag = this_i.ne(that_i)?;

        real.or(imag).map_err(TCError::from)
    }

    fn ne_const(this: (Self, Self), that: Number) -> TCResult<Self> {
        let (lr, li) = this;

        if let Number::Complex(that) = that {
            let (rr, ri) = that.into();
            let real = lr.ne_const(rr.into())?;
            let imag = li.ne_const(ri.into())?;
            real.or(imag)
        } else {
            let real = lr.ne_const(that)?;
            real.or(li)
        }
    }
}

impl<Txn, FE> ComplexCompare for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
}

impl<Txn, FE> ComplexCompare for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
}

pub(crate) trait ComplexMath: ComplexUnary + Clone {
    fn add(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        let real = this.0.add(that.0)?;
        let imag = this.1.add(that.1)?;
        Ok((real, imag))
    }

    fn add_const(this: (Self, Self), that: Number) -> TCResult<(Self, Self)> {
        let (a, b) = this;

        if let Number::Complex(that) = that {
            let (c, d) = that.into();
            let real = a.add_const(c.into())?;
            let imag = b.add_const(d.into())?;
            Ok((real, imag))
        } else {
            let real = a.add_const(that)?;
            Ok((real, b))
        }
    }

    fn add_real(this: (Self, Self), that: Self) -> TCResult<(Self, Self)> {
        let (real, imag) = this;
        let real = real.clone().add(that)?;
        Ok((real, imag))
    }

    fn const_div(this: Number, that: (Self, Self)) -> TCResult<(Self, Self)> {
        if let Number::Complex(this) = this {
            let (a, b) = this.into();
            let (c, d) = that;

            let denom = c
                .clone()
                .pow_const(2.into())?
                .add(d.clone().pow_const(2.into())?)?;

            let real_num = c
                .clone()
                .mul_const(a.into())?
                .add(d.clone().mul_const(b.into())?)?;

            let imag_num = c.mul_const(b.into())?.sub(d.mul_const(a.into())?)?;

            let real = real_num.div(denom.clone())?;
            let imag = imag_num.div(denom)?;

            Ok((real, imag))
        } else {
            let (real, imag) = that;

            let num_re = real.clone().mul_const(this)?;
            let num_im = imag.clone().mul_const(this)?;
            let denom = real.pow_const(2.into())?.add(imag.pow_const(2.into())?)?;

            ComplexMath::div_real((num_re, num_im), denom)
        }
    }

    fn div(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        let (a, b) = this;
        let (c, d) = that;

        let denom = c
            .clone()
            .pow_const(2.into())?
            .add(d.clone().pow_const(2.into())?)?;

        let real_num = a.clone().mul(c.clone())?.add(b.clone().mul(d.clone())?)?;
        let imag_num = b.mul(c)?.sub(a.mul(d)?)?;

        let real = real_num.div(denom.clone())?;
        let imag = imag_num.div(denom)?;

        Ok((real, imag))
    }

    fn div_real(this: (Self, Self), that: Self) -> TCResult<(Self, Self)> {
        let (real, imag) = this;
        let real = real.div(that.clone())?;
        let imag = imag.div(that)?;
        Ok((real, imag))
    }

    fn div_const(this: (Self, Self), that: Number) -> TCResult<(Self, Self)> {
        let (a, b) = this;

        if let Number::Complex(that) = that {
            let (c, d) = that.into();
            let (c, d) = (Number::from(c), Number::from(d));

            let denom = c.pow(2.into()) + d.pow(2.into());

            let real_num = a.clone().mul_const(c)?.add(b.clone().mul_const(d)?)?;
            let imag_num = b.mul_const(c)?.sub(a.mul_const(d)?)?;

            let real = real_num.div_const(denom)?;
            let imag = imag_num.div_const(denom)?;

            Ok((real, imag))
        } else {
            let real = a.div_const(that)?;
            let imag = b.div_const(that)?;
            Ok((real, imag))
        }
    }

    fn real_div_const(this: Self, that: Complex) -> TCResult<(Self, Self)> {
        let (real, imag) = that.into();
        let num_re = this.clone().mul_const(real.into())?;
        let num_im = this.mul_const((imag * (-1.).into()).into())?;
        let denom = real.pow(2.into()) + imag.pow(2.into());
        ComplexMath::div_const((num_re, num_im), denom.into())
    }

    fn log(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        ComplexMath::div(ComplexUnary::ln(this)?, ComplexUnary::ln(that)?)
    }

    fn log_const(this: (Self, Self), that: Number) -> TCResult<(Self, Self)> {
        ComplexMath::div_const(ComplexUnary::ln(this)?, that.ln())
    }

    fn mul(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        let (a, b) = this;
        let (c, d) = that;

        let real = a.clone().mul(c.clone())?.sub(b.clone().mul(d.clone())?)?;
        let imag = a.mul(d)?.add(b.mul(c)?)?;

        Ok((real, imag))
    }

    fn mul_const(this: (Self, Self), that: Number) -> TCResult<(Self, Self)> {
        let (a, b) = this;

        if let Number::Complex(that) = that {
            let (c, d) = that.into();
            let (c, d) = (Number::from(c), Number::from(d));

            let real = a.clone().mul_const(c)?.sub(b.clone().mul_const(d)?)?;
            let imag = a.mul_const(d)?.add(b.mul_const(c)?)?;

            Ok((real, imag))
        } else {
            let real = a.mul_const(that)?;
            let imag = b.mul_const(that)?;
            Ok((real, imag))
        }
    }

    fn mul_real(this: (Self, Self), that: Self) -> TCResult<(Self, Self)> {
        let (real, imag) = this;

        let real = real.mul(that.clone())?;
        let imag = imag.mul(that)?;

        Ok((real, imag))
    }

    fn real_mul_const(this: Self, that: Complex) -> TCResult<(Self, Self)> {
        let (real, imag) = that.into();
        let real = this.clone().mul_const(real.into())?;
        let imag = this.mul_const(imag.into())?;
        Ok((real, imag))
    }

    fn pow(this: (Self, Self), that: Self) -> TCResult<(Self, Self)>
    where
        Self: ComplexTrig,
    {
        let (x, y) = this.clone();

        let r = ComplexUnary::abs(this)?;
        let r = r.pow(that.clone())?;

        let theta = atan2(y, x)?;
        let theta = that.mul(theta)?;
        let theta_cos = theta.clone().cos()?;
        let theta_sin = theta.sin()?;

        Ok((r.clone().mul(theta_cos)?, r.mul(theta_sin)?))
    }

    fn pow_const(this: (Self, Self), that: Number) -> TCResult<(Self, Self)>
    where
        Self: ComplexTrig,
    {
        if let Number::Complex(that) = that {
            Err(bad_request!(
                "complex exponents are not supported: {}",
                that
            ))
        } else {
            let r = ComplexUnary::abs(this.clone())?;
            let r = r.pow_const(that)?;

            let (x, y) = this;
            let theta = atan2(y, x)?;
            let theta = theta.mul_const(that)?;
            let theta_cos = theta.clone().cos()?;
            let theta_sin = theta.sin()?;

            Ok((r.clone().mul(theta_cos)?, r.mul(theta_sin)?))
        }
    }

    fn sub(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        let real = this.0.sub(that.0)?;
        let imag = this.1.sub(that.1)?;
        Ok((real, imag))
    }

    fn sub_real(this: (Self, Self), that: Self) -> TCResult<(Self, Self)> {
        let (real, imag) = this;
        let real = real.sub(that)?;
        Ok((real, imag))
    }

    fn sub_const(this: (Self, Self), that: Number) -> TCResult<(Self, Self)> {
        let (a, b) = this;

        if let Number::Complex(that) = that {
            let (c, d) = that.into();
            let real = a.sub_const(c.into())?;
            let imag = b.sub_const(d.into())?;
            Ok((real, imag))
        } else {
            let real = a.sub_const(that)?;
            Ok((real, b))
        }
    }
}

impl<Txn, FE> ComplexMath for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
}

impl<Txn, FE> ComplexMath for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
}

pub(crate) trait ComplexTrig: ComplexMath + Clone {
    fn sin(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let (x, y) = this;
        let real = x.clone().sin()?.add(y.clone().cosh()?)?;
        let imag = x.cos()?.add(y.sinh()?)?;
        Ok((real, imag))
    }

    fn sinh(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let (x, y) = this;
        let real = x.clone().sinh()?.mul(y.clone().cos()?)?;
        let imag = x.cosh()?.mul(y.sin()?)?;
        Ok((real, imag))
    }

    fn cos(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let (x, y) = this;
        let real = x.clone().cos()?.mul(y.clone().cosh()?)?;
        let imag = x.sin()?.mul(y.sinh()?)?;
        Ok((real, imag))
    }

    fn cosh(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let (x, y) = this;
        let real = x.clone().cosh()?.mul(y.clone().cos()?)?;
        let imag = x.sinh()?.mul(y.sin()?)?;
        Ok((real, imag))
    }

    fn tan(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let (x, y) = this;

        let num_real = x.clone().sin()?.mul(y.clone().cosh()?)?;
        let num_imag = x.clone().cos()?.mul(y.clone().sinh()?)?;
        let num = (num_real, num_imag);

        let denom_real = x.clone().cos()?.mul(y.clone().cosh()?)?;
        let denom_imag = x.clone().sin()?.mul(y.clone().sinh()?)?;
        let denom = (denom_real, denom_imag);

        ComplexMath::div(num, denom)
    }

    fn tanh(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let (x, y) = this;

        let num_real = x.clone().sinh()?.mul(y.clone().cos()?)?;
        let num_imag = x.clone().cosh()?.mul(y.clone().sin()?)?;
        let num = (num_real, num_imag);

        let denom_real = x.clone().cosh()?.mul(y.clone().cos()?)?;
        let denom_imag = x.sinh()?.mul(y.sin()?)?;
        let denom = (denom_real, denom_imag);

        ComplexMath::div(num, denom)
    }
}

impl<Txn, FE> ComplexTrig for DenseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
}

impl<Txn, FE> ComplexTrig for SparseView<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
}

#[inline]
fn atan2<F>(_y: F, _x: F) -> TCResult<F>
where
    F: TensorMath<F, Combine = F, LeftCombine = F>,
{
    Err(not_implemented!("atan2"))
}
