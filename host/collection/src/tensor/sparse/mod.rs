use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, TryFutureExt};
use ha_ndarray::{CDatatype, NDArrayMath, NDArrayRead, NDArrayTransform};
use safecast::{AsType, CastFrom};

mod access;
mod base;
mod schema;
mod stream;

use tc_error::*;
use tc_value::{DType, Number, NumberType};
use tcgeneric::ThreadSafe;

use crate::tensor::dense::{DenseSparse, DenseTensor};

use super::{
    Axes, Coord, Range, Shape, TensorBoolean, TensorCompare, TensorCompareConst, TensorConvert,
    TensorInstance, TensorTransform,
};

pub use access::*;

const BLOCK_SIZE: usize = 4_096;

pub use base::SparseBase;
pub use schema::{IndexSchema, Schema};

pub type Blocks<C, V> = Pin<Box<dyn Stream<Item = Result<(C, V), TCError>> + Send>>;
pub type Elements<T> = Pin<Box<dyn Stream<Item = Result<(Coord, T), TCError>> + Send>>;
pub type Node = b_table::b_tree::Node<Vec<Vec<Number>>>;

#[async_trait]
pub trait SparseInstance: TensorInstance + fmt::Debug {
    type CoordBlock: NDArrayRead<DType = u64> + NDArrayMath + NDArrayTransform;
    type ValueBlock: NDArrayRead<DType = Self::DType>;
    type Blocks: Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), TCError>> + Send;
    type DType: CDatatype + DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError>;

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError>;

    async fn filled_at(
        self,
        range: Range,
        axes: Axes,
    ) -> Result<stream::FilledAt<Elements<Self::DType>>, TCError>
    where
        Self: Sized,
    {
        let ndim = self.ndim();

        let elided = (0..ndim).filter(|x| !axes.contains(x));

        let mut order = Vec::with_capacity(ndim);
        order.copy_from_slice(&axes);
        order.extend(elided);

        self.elements(range, order)
            .map_ok(|elements| stream::FilledAt::new(elements, axes, ndim))
            .await
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError>;
}

pub struct SparseTensor<FE, A> {
    accessor: A,
    phantom: PhantomData<FE>,
}

impl<FE, A> SparseTensor<FE, A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<FE, A: Clone> Clone for SparseTensor<FE, A> {
    fn clone(&self) -> Self {
        Self {
            accessor: self.accessor.clone(),
            phantom: PhantomData,
        }
    }
}

impl<FE, A> TensorInstance for SparseTensor<FE, A>
where
    FE: Send + Sync + 'static,
    A: SparseInstance,
{
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<FE, L, R> TensorBoolean<SparseTensor<FE, R>> for SparseTensor<FE, L>
where
    FE: AsType<Node> + ThreadSafe,
    L: SparseInstance + Into<SparseAccessCast<FE>>,
    R: SparseInstance<DType = L::DType> + Into<SparseAccessCast<FE>>,
    Number: From<L::DType> + From<R::DType>,
{
    type Combine = SparseTensor<FE, SparseCombine<FE, u8>>;
    type LeftCombine = SparseTensor<FE, SparseLeftCombine<FE, u8>>;

    fn and(self, other: SparseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        let access = SparseLeftCombine::new(self.accessor, other.accessor, |l, r| {
            if bool::cast_from(l) && bool::cast_from(r) {
                1
            } else {
                0
            }
        })?;

        Ok(SparseTensor::from(access))
    }

    fn or(self, other: SparseTensor<FE, R>) -> TCResult<Self::Combine> {
        let access = SparseCombine::new(self.accessor, other.accessor, |l, r| {
            if bool::cast_from(l) && bool::cast_from(r) {
                1
            } else {
                0
            }
        })?;

        Ok(SparseTensor::from(access))
    }

    fn xor(self, other: SparseTensor<FE, R>) -> TCResult<Self::Combine> {
        let access = SparseCombine::new(self.accessor, other.accessor, |l, r| {
            if bool::cast_from(l) && bool::cast_from(r) {
                1
            } else {
                0
            }
        })?;

        Ok(SparseTensor::from(access))
    }
}

impl<FE: AsType<Node> + ThreadSafe, A: SparseInstance> TensorConvert for SparseTensor<FE, A> {
    type Dense = DenseTensor<FE, DenseSparse<A>>;

    fn into_dense(self) -> Self::Dense {
        DenseSparse::from(self.accessor).into()
    }
}

impl<FE, L, R> TensorCompare<SparseTensor<FE, R>> for SparseTensor<FE, L>
where
    FE: AsType<Node> + ThreadSafe,
    L: SparseInstance + Into<SparseAccess<FE, L::DType>> + fmt::Debug,
    R: SparseInstance<DType = L::DType> + Into<SparseAccess<FE, R::DType>> + fmt::Debug,
    SparseAccessCast<FE>: From<SparseAccess<FE, L::DType>>,
{
    type Compare = SparseTensor<FE, SparseCombine<FE, u8>>;

    fn eq(self, other: SparseTensor<FE, R>) -> TCResult<Self::Compare> {
        Err(bad_request!(
            "cannot compare {:?} with {:?} because the result would be dense",
            self,
            other
        ))
    }

    fn gt(self, other: SparseTensor<FE, R>) -> TCResult<Self::Compare> {
        SparseCombine::new(self.accessor.into(), other.accessor.into(), |l, r| {
            if l.gt(&r) {
                1
            } else {
                0
            }
        })
        .map(SparseTensor::from)
    }

    fn ge(self, other: SparseTensor<FE, R>) -> TCResult<Self::Compare> {
        Err(bad_request!(
            "cannot compare {:?} with {:?} because the result would be dense",
            self,
            other
        ))
    }

    fn lt(self, other: SparseTensor<FE, R>) -> TCResult<Self::Compare> {
        SparseCombine::new(self.accessor.into(), other.accessor.into(), |l, r| {
            if l.lt(&r) {
                1
            } else {
                0
            }
        })
        .map(SparseTensor::from)
    }

    fn le(self, other: SparseTensor<FE, R>) -> TCResult<Self::Compare> {
        Err(bad_request!(
            "cannot compare {:?} with {:?} because the result would be dense",
            self,
            other
        ))
    }

    fn ne(self, other: SparseTensor<FE, R>) -> TCResult<Self::Compare> {
        SparseCombine::new(self.accessor.into(), other.accessor.into(), |l, r| {
            if l.ne(&r) {
                1
            } else {
                0
            }
        })
        .map(SparseTensor::from)
    }
}

impl<FE, A> TensorCompareConst for SparseTensor<FE, A>
where
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance + Into<SparseAccess<FE, A::DType>>,
    SparseAccessCast<FE>: From<SparseAccess<FE, A::DType>>,
{
    type Compare = SparseTensor<FE, SparseCombineConst<FE, u8>>;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.eq(&r) { 1 } else { 0 };
        let sparse = SparseCombineConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.gt(&r) { 1 } else { 0 };
        let sparse = SparseCombineConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.ge(&r) { 1 } else { 0 };
        let sparse = SparseCombineConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.lt(&r) { 1 } else { 0 };
        let sparse = SparseCombineConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.le(&r) { 1 } else { 0 };
        let sparse = SparseCombineConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.ne(&r) { 1 } else { 0 };
        let sparse = SparseCombineConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }
}

impl<FE, A> TensorTransform for SparseTensor<FE, A>
where
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance + Into<SparseAccess<FE, A::DType>>,
    A::DType: CastFrom<Number> + fmt::Debug,
    Number: From<A::DType>,
{
    type Broadcast = SparseTensor<FE, SparseBroadcast<FE, A::DType>>;
    type Expand = SparseTensor<FE, SparseExpand<A>>;
    type Reshape = SparseTensor<FE, SparseReshape<A>>;
    type Slice = SparseTensor<FE, SparseSlice<A>>;
    type Transpose = SparseTensor<FE, SparseTranspose<A>>;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        let accessor = SparseBroadcast::new(self.accessor, shape)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn expand(self, axes: Axes) -> TCResult<Self::Expand> {
        let accessor = SparseExpand::new(self.accessor, axes)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        let accessor = SparseReshape::new(self.accessor, shape)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        let accessor = SparseSlice::new(self.accessor, range)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> TCResult<Self::Transpose> {
        let accessor = SparseTranspose::new(self.accessor, permutation)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }
}

impl<FE, A> From<A> for SparseTensor<FE, A>
where
    A: SparseInstance,
{
    fn from(accessor: A) -> Self {
        Self {
            accessor,
            phantom: PhantomData,
        }
    }
}

impl<FE, A: fmt::Debug> fmt::Debug for SparseTensor<FE, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.accessor.fmt(f)
    }
}
