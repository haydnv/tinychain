use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use collate::Collate;
use futures::{Stream, StreamExt, TryFutureExt, TryStreamExt};
use ha_ndarray::*;
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{DType, Number, NumberClass, NumberCollator, NumberInstance, NumberType};
use tcgeneric::ThreadSafe;

use super::dense::{DenseAccess, DenseAccessCast, DenseSparse, DenseTensor};

use super::{
    Axes, Coord, Range, Shape, TensorBoolean, TensorBooleanConst, TensorCompare,
    TensorCompareConst, TensorConvert, TensorInstance, TensorMath, TensorMathConst,
    TensorPermitRead, TensorReduce, TensorTransform, TensorUnary, TensorUnaryBoolean,
};

pub use access::*;
pub use base::SparseBase;
pub use schema::{IndexSchema, Schema};
pub use view::*;

mod access;
mod base;
mod schema;
mod stream;
mod view;

const BLOCK_SIZE: usize = 4_096;

pub type Blocks<C, V> = Pin<Box<dyn Stream<Item = Result<(C, V), TCError>> + Send>>;
pub type Elements<T> = Pin<Box<dyn Stream<Item = Result<(Coord, T), TCError>> + Send>>;
pub type Node = b_table::b_tree::Node<Vec<Vec<Number>>>;

#[async_trait]
pub trait SparseInstance: TensorInstance + fmt::Debug {
    type CoordBlock: NDArrayRead<DType = u64> + NDArrayMath + NDArrayTransform;
    type ValueBlock: NDArrayRead<DType = Self::DType> + Into<Array<Self::DType>>;
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

impl<FE, T: CDatatype> SparseTensor<FE, SparseAccess<FE, T>> {
    pub fn from_access<A: Into<SparseAccess<FE, T>>>(accessor: A) -> Self {
        Self {
            accessor: accessor.into(),
            phantom: PhantomData,
        }
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

impl<FE: ThreadSafe, A: TensorInstance> TensorInstance for SparseTensor<FE, A> {
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
    L: SparseInstance + Into<SparseAccess<FE, L::DType>>,
    R: SparseInstance<DType = L::DType> + Into<SparseAccess<FE, R::DType>>,
    Number: From<L::DType> + From<R::DType>,
    SparseAccessCast<FE>: From<SparseAccess<FE, L::DType>>,
{
    type Combine = SparseTensor<FE, SparseCompare<FE, u8>>;
    type LeftCombine = SparseTensor<FE, SparseCompareLeft<FE, u8>>;

    fn and(self, other: SparseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        let access =
            SparseCompareLeft::new(self.accessor.into(), other.accessor.into(), |l, r| {
                if bool::cast_from(l) && bool::cast_from(r) {
                    1
                } else {
                    0
                }
            })?;

        Ok(SparseTensor::from(access))
    }

    fn or(self, other: SparseTensor<FE, R>) -> TCResult<Self::Combine> {
        let access = SparseCompare::new(self.accessor.into(), other.accessor.into(), |l, r| {
            if bool::cast_from(l) && bool::cast_from(r) {
                1
            } else {
                0
            }
        })?;

        Ok(SparseTensor::from(access))
    }

    fn xor(self, other: SparseTensor<FE, R>) -> TCResult<Self::Combine> {
        let access = SparseCompare::new(self.accessor.into(), other.accessor.into(), |l, r| {
            if bool::cast_from(l) && bool::cast_from(r) {
                1
            } else {
                0
            }
        })?;

        Ok(SparseTensor::from(access))
    }
}

impl<FE, A> TensorBooleanConst for SparseTensor<FE, A>
where
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance + Into<SparseAccess<FE, A::DType>>,
    DenseAccess<FE, A::DType>: From<DenseSparse<A>>,
    DenseAccessCast<FE>: From<DenseAccess<FE, A::DType>>,
    SparseAccessCast<FE>: From<SparseAccess<FE, A::DType>>,
{
    type Combine = SparseTensor<FE, SparseCompareConst<FE, u8>>;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        let access = SparseCompareConst::new(self.accessor.into(), other, |l, r| {
            if l.and(r).cast_into() {
                1
            } else {
                0
            }
        });

        Ok(SparseTensor::from(access))
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        Err(bad_request!("cannot call OR {} on {:?} because the result would not be sparse (consider converting to a dense tensor first)", other, self))
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        Err(bad_request!("cannot call XOR {} on {:?} because the result would not be sparse (consider converting to a dense tensor first)", other, self))
    }
}

impl<FE: AsType<Node> + ThreadSafe, A: SparseInstance> TensorConvert for SparseTensor<FE, A> {
    type Dense = DenseTensor<FE, DenseSparse<A>>;
    type Sparse = Self;

    fn into_dense(self) -> Self::Dense {
        DenseSparse::from(self.accessor).into()
    }

    fn into_sparse(self) -> Self::Sparse {
        self
    }
}

impl<FE, L, R> TensorCompare<SparseTensor<FE, R>> for SparseTensor<FE, L>
where
    FE: AsType<Node> + ThreadSafe,
    L: SparseInstance + Into<SparseAccess<FE, L::DType>> + fmt::Debug,
    R: SparseInstance<DType = L::DType> + Into<SparseAccess<FE, R::DType>> + fmt::Debug,
    SparseAccessCast<FE>: From<SparseAccess<FE, L::DType>>,
{
    type Compare = SparseTensor<FE, SparseCompare<FE, u8>>;

    fn eq(self, other: SparseTensor<FE, R>) -> TCResult<Self::Compare> {
        Err(bad_request!(
            "cannot compare {:?} with {:?} because the result would be dense",
            self,
            other
        ))
    }

    fn gt(self, other: SparseTensor<FE, R>) -> TCResult<Self::Compare> {
        SparseCompare::new(self.accessor.into(), other.accessor.into(), |l, r| {
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
        SparseCompare::new(self.accessor.into(), other.accessor.into(), |l, r| {
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
        SparseCompare::new(self.accessor.into(), other.accessor.into(), |l, r| {
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
    type Compare = SparseTensor<FE, SparseCompareConst<FE, u8>>;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.eq(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.gt(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.ge(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.lt(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.le(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.ne(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, cmp);
        Ok(sparse.into())
    }
}

impl<FE, L, R, T> TensorMath<SparseTensor<FE, R>> for SparseTensor<FE, L>
where
    FE: ThreadSafe,
    L: SparseInstance<DType = T>,
    R: SparseInstance<DType = T>,
    T: CDatatype + DType,
{
    type Combine = SparseTensor<FE, SparseCombine<L, R, T>>;
    type LeftCombine = SparseTensor<FE, SparseCombineLeft<L, R, T>>;

    fn add(self, other: SparseTensor<FE, R>) -> TCResult<Self::Combine> {
        SparseCombine::new(
            self.accessor,
            other.accessor,
            |l, r| l.add(r).map(Array::from).map_err(TCError::from),
            |l, r| l + r,
        )
        .map(SparseTensor::from)
    }

    fn div(self, other: SparseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        SparseCombineLeft::new(
            self.accessor,
            other.accessor,
            |l, r| l.div(r).map(Array::from).map_err(TCError::from),
            |l, r| l / r,
        )
        .map(SparseTensor::from)
    }

    fn log(self, base: SparseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn log<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            let right = right.cast()?;
            left.log(right).map(Array::from).map_err(TCError::from)
        }

        SparseCombineLeft::new(self.accessor, base.accessor, log, |l, r| {
            T::from_f64(T::to_f64(l).log(T::to_f64(r)))
        })
        .map(SparseTensor::from)
    }

    fn mul(self, other: SparseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        SparseCombineLeft::new(
            self.accessor,
            other.accessor,
            |l, r| l.mul(r).map(Array::from).map_err(TCError::from),
            |l, r| l * r,
        )
        .map(SparseTensor::from)
    }

    fn pow(self, other: SparseTensor<FE, R>) -> TCResult<Self::LeftCombine> {
        fn pow<T: CDatatype>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
            let right = right.cast()?;
            left.pow(right).map(Array::from).map_err(TCError::from)
        }

        SparseCombineLeft::new(self.accessor, other.accessor, pow, |l, r| {
            T::from_f64(T::to_f64(l).pow(T::to_f64(r)))
        })
        .map(SparseTensor::from)
    }

    fn sub(self, other: SparseTensor<FE, R>) -> TCResult<Self::Combine> {
        SparseCombine::new(
            self.accessor,
            other.accessor,
            |l, r| l.add(r).map(Array::from).map_err(TCError::from),
            |l, r| l + r,
        )
        .map(SparseTensor::from)
    }
}

impl<FE, A> TensorMathConst for SparseTensor<FE, A>
where
    FE: ThreadSafe,
    A: SparseInstance,
    A::DType: CastFrom<Number>,
    <A::DType as CDatatype>::Float: CastFrom<Number>,
    Number: From<A::DType>,
{
    type Combine = SparseTensor<FE, SparseCombineConst<A, A::DType>>;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
        Err(bad_request!("cannot add {other} to {self:?} because the result would not be sparse (consider converting to a dense tensor first)"))
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        if bool::cast_from(other) {
            let access = SparseCombineConst::new(
                self.accessor,
                other,
                |l, r| {
                    l.div_scalar(r.cast_into())
                        .map(Array::from)
                        .map_err(TCError::from)
                },
                |l, r| l / r.cast_into(),
            );

            Ok(SparseTensor::from(access))
        } else {
            Err(bad_request!("cannot divide {self:?} by {other}"))
        }
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        let access = SparseCombineConst::new(
            self.accessor,
            base,
            |l, r| {
                l.log_scalar(r.cast_into())
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |l, r| {
                Number::from(l)
                    .log(tc_value::Float::cast_from(r))
                    .cast_into()
            },
        );

        Ok(SparseTensor::from(access))
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        let access = SparseCombineConst::new(
            self.accessor,
            other,
            |l, r| {
                l.mul_scalar(r.cast_into())
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |l, r| l * r.cast_into(),
        );

        Ok(SparseTensor::from(access))
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        let access = SparseCombineConst::new(
            self.accessor,
            other,
            |l, r| {
                l.pow_scalar(r.cast_into())
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |l, r| Number::from(l).pow(r).cast_into(),
        );

        Ok(SparseTensor::from(access))
    }

    fn sub_const(self, other: Number) -> TCResult<Self::Combine> {
        Err(bad_request!("cannot subtract {other} from {self:?} because the result would not be sparse (consider converting to a dense tensor first)"))
    }
}

#[async_trait]
impl<FE, A> TensorReduce for SparseTensor<FE, A>
where
    FE: ThreadSafe,
    A: SparseInstance + TensorPermitRead + Clone,
    Number: From<A::DType>,
{
    type Reduce = SparseTensor<FE, SparseReduce<A, A::DType>>;

    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;

        let range = Range::all(self.shape());
        let axes = (0..self.ndim()).into_iter().collect();
        let filled_at = self.accessor.filled_at(Range::default(), axes).await?;

        let mut coords = futures::stream::iter(range.affected())
            .zip(filled_at)
            .map(|(expected, result)| result.map(|actual| (expected, actual)));

        while let Some((expected, actual)) = coords.try_next().await? {
            if expected != actual {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;

        let axes = (0..self.ndim()).into_iter().collect();
        let mut filled_at = self.accessor.filled_at(Range::default(), axes).await?;
        filled_at.try_next().map_ok(|r| r.is_some()).await
    }

    fn max(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        let block_op = |block: Array<A::DType>| block.max().map_err(TCError::from);

        fn max_value<T: Into<Number> + Copy>(l: T, r: T) -> T {
            match NumberCollator::default().cmp(&l.into(), &r.into()) {
                Ordering::Less => r,
                Ordering::Equal | Ordering::Greater => l,
            }
        }

        SparseReduce::new(
            self.accessor,
            A::DType::min(),
            axes,
            keepdims,
            block_op,
            max_value,
        )
        .map(SparseTensor::from)
    }

    async fn max_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;

        let collator = NumberCollator::default();
        let blocks = self.accessor.blocks(Range::default(), vec![]).await?;

        blocks
            .map(|result| {
                result.and_then(|(_coords, values)| {
                    values.max().map(Number::from).map_err(TCError::from)
                })
            })
            .try_fold(A::DType::min().into(), move |max, block_max| {
                let max = match collator.cmp(&max, &block_max) {
                    Ordering::Less => block_max,
                    Ordering::Equal | Ordering::Greater => max,
                };

                futures::future::ready(Ok(max))
            })
            .await
    }

    fn min(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        let block_op = |block: Array<A::DType>| block.min().map_err(TCError::from);

        fn min_value<T: Into<Number> + Copy>(l: T, r: T) -> T {
            match NumberCollator::default().cmp(&l.into(), &r.into()) {
                Ordering::Greater => r,
                Ordering::Equal | Ordering::Less => l,
            }
        }

        SparseReduce::new(
            self.accessor,
            A::DType::max(),
            axes,
            keepdims,
            block_op,
            min_value,
        )
        .map(SparseTensor::from)
    }

    async fn min_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;

        let collator = NumberCollator::default();
        let blocks = self.accessor.blocks(Range::default(), vec![]).await?;

        blocks
            .map(|result| {
                result.and_then(|(_coords, values)| {
                    values.max().map(Number::from).map_err(TCError::from)
                })
            })
            .try_fold(A::DType::min().into(), move |max, block_max| {
                let max = match collator.cmp(&max, &block_max) {
                    Ordering::Less => block_max,
                    Ordering::Equal | Ordering::Greater => max,
                };

                futures::future::ready(Ok(max))
            })
            .await
    }

    fn product(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        SparseReduce::new(
            self.accessor,
            A::DType::one(),
            axes,
            keepdims,
            |block| block.product().map_err(TCError::from),
            |l, r| l * r,
        )
        .map(SparseTensor::from)
    }

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;

        let blocks = self.accessor.blocks(Range::default(), vec![]).await?;

        blocks
            .map(|result| {
                result.and_then(|(_coords, values)| values.product().map_err(TCError::from))
            })
            .try_fold(A::DType::one(), move |product, block_product| {
                futures::future::ready(Ok(product * block_product))
            })
            .map_ok(Number::from)
            .await
    }

    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        SparseReduce::new(
            self.accessor,
            A::DType::one(),
            axes,
            keepdims,
            |block| block.sum().map_err(TCError::from),
            |l, r| l + r,
        )
        .map(SparseTensor::from)
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        let _permit = self.accessor.read_permit(txn_id, Range::default()).await?;

        let blocks = self.accessor.blocks(Range::default(), vec![]).await?;

        blocks
            .map(|result| result.and_then(|(_coords, values)| values.sum().map_err(TCError::from)))
            .try_fold(A::DType::one(), move |sum, block_sum| {
                futures::future::ready(Ok(sum + block_sum))
            })
            .map_ok(Number::from)
            .await
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

impl<FE, A> TensorUnary for SparseTensor<FE, A>
where
    FE: ThreadSafe,
    A: SparseInstance,
{
    type Unary = SparseTensor<FE, SparseUnary<A, A::DType>>;

    fn abs(self) -> TCResult<Self::Unary> {
        let accessor = SparseUnary::new(
            self.accessor,
            |arr| arr.abs().map(Array::from).map_err(TCError::from),
            |n| A::DType::abs(n),
        );

        Ok(accessor.into())
    }

    fn exp(self) -> TCResult<Self::Unary> {
        let accessor = SparseUnary::new(
            self.accessor,
            |arr| arr.exp().map(Array::from).map_err(TCError::from),
            |n| A::DType::from_f64(A::DType::to_f64(n).exp()),
        );

        Ok(accessor.into())
    }

    fn ln(self) -> TCResult<Self::Unary> {
        let accessor = SparseUnary::new(
            self.accessor,
            |arr| arr.ln().map(Array::from).map_err(TCError::from),
            |n| A::DType::from_f64(A::DType::to_f64(n).ln()),
        );

        Ok(accessor.into())
    }

    fn round(self) -> TCResult<Self::Unary> {
        let accessor = SparseUnary::new(
            self.accessor,
            |arr| arr.round().map(Array::from).map_err(TCError::from),
            |n| A::DType::round(n),
        );

        Ok(accessor.into())
    }
}

impl<FE, A> TensorUnaryBoolean for SparseTensor<FE, A>
where
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance + Into<SparseAccess<FE, A::DType>>,
    SparseAccessCast<FE>: From<SparseAccess<FE, A::DType>>,
{
    type Unary = SparseTensor<FE, SparseUnaryCast<FE, u8>>;

    fn not(self) -> TCResult<Self::Unary> {
        let accessor = SparseUnaryCast::new(
            self.accessor.into(),
            |block| block.not(),
            |n| if n == NumberType::Number.zero() { 1 } else { 0 },
        );

        Ok(accessor.into())
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

#[inline]
pub fn sparse_from<FE, A, T>(tensor: SparseTensor<FE, A>) -> SparseTensor<FE, SparseAccess<FE, T>>
where
    A: Into<SparseAccess<FE, T>>,
    T: CDatatype,
{
    SparseTensor::from_access(tensor.into_inner())
}
