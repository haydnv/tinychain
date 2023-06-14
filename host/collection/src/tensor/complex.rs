use safecast::AsType;
use tc_error::*;
use tcgeneric::ThreadSafe;

use crate::tensor::{
    TensorBoolean, TensorCompare, TensorInstance, TensorMath, TensorMathConst, TensorTrig,
    TensorUnary,
};

use super::dense::{DenseCacheFile, DenseView};
use super::sparse::{Node, SparseView};

pub(crate) trait TensorComplex:
    TensorInstance
    + TensorMath<Self, Combine = Self, LeftCombine = Self>
    + TensorMathConst<Combine = Self, DenseCombine = Self>
    + TensorUnary<Unary = Self>
    + Clone
{
    fn abs(this: (Self, Self)) -> TCResult<Self> {
        let (real, imag) = this;
        real.pow_const(2.into())?
            .add(imag.pow_const(2.into())?)?
            .pow_const((0.5).into())
    }

    fn ln(this: (Self, Self)) -> TCResult<(Self, Self)> {
        let (x, y) = this.clone();
        let r = TensorComplex::abs(this)?;
        let real = r.ln()?;
        let imag = atan2(y, x)?;
        Ok((real, imag))
    }
}

impl<FE: DenseCacheFile + AsType<Node> + Clone> TensorComplex for DenseView<FE> {}
impl<FE: ThreadSafe + AsType<Node>> TensorComplex for SparseView<FE> {}

pub(crate) trait ComplexCompare:
    TensorComplex
    + TensorBoolean<Self, Combine = Self, LeftCombine = Self>
    + TensorCompare<Self, Compare = Self>
{
    fn eq(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let (this_r, this_i) = this;
        let (that_r, that_i) = that;

        let real = this_r.eq(that_r)?;
        let imag = this_i.eq(that_i)?;

        real.and(imag).map_err(TCError::from)
    }

    fn gt(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let this = TensorComplex::abs(this)?;
        let that = TensorComplex::abs(that)?;
        this.gt(that)
    }

    fn ge(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let this = TensorComplex::abs(this)?;
        let that = TensorComplex::abs(that)?;
        this.ge(that)
    }

    fn lt(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let this = TensorComplex::abs(this)?;
        let that = TensorComplex::abs(that)?;
        this.lt(that)
    }

    fn le(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let this = TensorComplex::abs(this)?;
        let that = TensorComplex::abs(that)?;
        this.le(that)
    }

    fn ne(this: (Self, Self), that: (Self, Self)) -> TCResult<Self> {
        let (this_r, this_i) = this;
        let (that_r, that_i) = that;

        let real = this_r.ne(that_r)?;
        let imag = this_i.ne(that_i)?;

        real.or(imag).map_err(TCError::from)
    }
}

impl<FE: DenseCacheFile + AsType<Node> + Clone> ComplexCompare for DenseView<FE> {}
impl<FE: ThreadSafe + AsType<Node>> ComplexCompare for SparseView<FE> {}

pub(crate) trait ComplexMath: TensorComplex + TensorTrig<Unary = Self> + Clone {
    fn add(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        let real = this.0.add(that.0)?;
        let imag = this.1.add(that.1)?;
        Ok((real, imag))
    }

    fn div(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        let (a, b) = this;
        let (c, d) = that;

        let denom = c
            .clone()
            .pow_const(2.into())?
            .add(d.clone().pow_const(2.into())?)?;

        let real_num = a.clone().mul(c.clone())?.add(b.clone().mul(d.clone())?)?;
        let real = real_num.div(denom.clone())?;

        let imag_num = b.mul(c)?.sub(a.mul(d)?)?;
        let imag = imag_num.div(denom)?;

        Ok((real, imag))
    }

    fn log(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        ComplexMath::div(TensorComplex::ln(this)?, TensorComplex::ln(that)?)
    }

    fn mul(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        let (a, b) = this;
        let (c, d) = that;

        let real = a.clone().mul(c.clone())?.sub(b.clone().mul(d.clone())?)?;
        let imag = a.mul(d)?.add(b.mul(c)?)?;

        Ok((real, imag))
    }

    fn pow(this: (Self, Self), that: Self) -> TCResult<(Self, Self)>
    where
        Self: ComplexTrig,
    {
        let (x, y) = this.clone();

        let r = TensorComplex::abs(this)?;
        let r = r.pow(that.clone())?;

        let theta = atan2(y, x)?;
        let theta = that.mul(theta)?;
        let theta_cos = theta.clone().cos()?;
        let theta_sin = theta.sin()?;

        Ok((r.clone().mul(theta_cos)?, r.mul(theta_sin)?))
    }

    fn sub(this: (Self, Self), that: (Self, Self)) -> TCResult<(Self, Self)> {
        let real = this.0.sub(that.0)?;
        let imag = this.1.sub(that.1)?;
        Ok((real, imag))
    }
}

impl<FE: DenseCacheFile + AsType<Node> + Clone> ComplexMath for DenseView<FE> {}
impl<FE: ThreadSafe + AsType<Node>> ComplexMath for SparseView<FE> {}

pub(crate) trait ComplexTrig: TensorComplex + Clone {
    fn sin(this: (Self, Self)) -> TCResult<(Self, Self)> {
        todo!()
    }

    fn cos(this: (Self, Self)) -> TCResult<(Self, Self)> {
        todo!()
    }
}

impl<FE: DenseCacheFile + AsType<Node> + Clone> ComplexTrig for DenseView<FE> {}
impl<FE: ThreadSafe + AsType<Node>> ComplexTrig for SparseView<FE> {}

#[inline]
fn atan2<F>(_y: F, _x: F) -> TCResult<F>
where
    F: TensorMath<F, Combine = F, LeftCombine = F>,
{
    Err(not_implemented!("atan2"))
}
