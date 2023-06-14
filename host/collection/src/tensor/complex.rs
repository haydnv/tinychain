use safecast::AsType;
use tc_error::*;
use tcgeneric::ThreadSafe;

use crate::tensor::{TensorBoolean, TensorCompare, TensorInstance, TensorMath, TensorMathConst};

use super::dense::{DenseCacheFile, DenseView};
use super::sparse::{Node, SparseView};

trait TensorComplex:
    TensorInstance
    + TensorMath<Self, Combine = Self, LeftCombine = Self>
    + TensorMathConst<Combine = Self, DenseCombine = Self>
{
    fn abs(this: (Self, Self)) -> TCResult<Self> {
        let (real, imag) = this;
        real.pow_const(2.into())?
            .add(imag.pow_const(2.into())?)?
            .pow_const((0.5).into())
    }
}

impl<FE: DenseCacheFile + AsType<Node> + Clone> TensorComplex for DenseView<FE> {}
impl<FE: ThreadSafe + AsType<Node>> TensorComplex for SparseView<FE> {}

trait TensorComplexCompare:
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

impl<FE: DenseCacheFile + AsType<Node> + Clone> TensorComplexCompare for DenseView<FE> {}
impl<FE: ThreadSafe + AsType<Node>> TensorComplexCompare for SparseView<FE> {}
