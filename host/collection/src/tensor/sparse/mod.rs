use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use collate::Collate;
use futures::{try_join, Stream, StreamExt, TryFutureExt, TryStreamExt};
use ha_ndarray::*;
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::{
    Complex, ComplexType, DType, Number, NumberClass, NumberCollator, NumberInstance, NumberType,
};
use tcgeneric::ThreadSafe;

use super::block::Block;
use super::complex::ComplexRead;
use super::dense::{DenseAccess, DenseAccessCast, DenseSparse, DenseTensor};

use super::{
    Axes, Coord, Range, Shape, TensorBoolean, TensorBooleanConst, TensorCast, TensorCompare,
    TensorCompareConst, TensorConvert, TensorInstance, TensorMath, TensorMathConst,
    TensorPermitRead, TensorRead, TensorReduce, TensorTransform, TensorUnary, TensorUnaryBoolean,
    TensorWrite, TensorWriteDual,
};

pub use access::*;
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
    type CoordBlock: NDArrayRead<DType = u64> + NDArrayMath + NDArrayTransform + Into<Array<u64>>;
    type ValueBlock: NDArrayRead<DType = Self::DType> + Into<Array<Self::DType>>;
    type Blocks: Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), TCError>> + Send;
    type DType: CDatatype + DType;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError>;

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError>;

    async fn filled_at(
        self,
        txn_id: TxnId,
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

        self.elements(txn_id, range, order)
            .map_ok(|elements| stream::FilledAt::new(elements, axes, ndim))
            .await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError>;
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
        let access = SparseCompareLeft::new(
            self.accessor.into(),
            other.accessor.into(),
            |l, r| l.and(r),
            |l, r| {
                if bool::cast_from(l) && bool::cast_from(r) {
                    1
                } else {
                    0
                }
            },
        )?;

        Ok(SparseTensor::from(access))
    }

    fn or(self, other: SparseTensor<FE, R>) -> TCResult<Self::Combine> {
        let access = SparseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            |l, r| l.or(r),
            |l, r| {
                if bool::cast_from(l) && bool::cast_from(r) {
                    1
                } else {
                    0
                }
            },
        )?;

        Ok(SparseTensor::from(access))
    }

    fn xor(self, other: SparseTensor<FE, R>) -> TCResult<Self::Combine> {
        let access = SparseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            |l, r| l.xor(r),
            |l, r| {
                if bool::cast_from(l) && bool::cast_from(r) {
                    1
                } else {
                    0
                }
            },
        )?;

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
        let cmp = |l: Number, r: Number| {
            if bool::cast_from(l.and(r)) {
                1
            } else {
                0
            }
        };

        let access = SparseCompareConst::new(self.accessor.into(), other, Block::and_scalar, cmp);

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
        SparseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            |l, r| l.gt(r),
            |l, r| {
                if l.gt(&r) {
                    1
                } else {
                    0
                }
            },
        )
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
        SparseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            |l, r| l.lt(r),
            |l, r| {
                if l.lt(&r) {
                    1
                } else {
                    0
                }
            },
        )
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
        SparseCompare::new(
            self.accessor.into(),
            other.accessor.into(),
            |l, r| l.ne(r),
            |l, r| {
                if l.ne(&r) {
                    1
                } else {
                    0
                }
            },
        )
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
        let sparse = SparseCompareConst::new(self.accessor.into(), other, Block::eq_scalar, cmp);
        Ok(sparse.into())
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.gt(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, Block::gt_scalar, cmp);
        Ok(sparse.into())
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.ge(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, Block::ge_scalar, cmp);
        Ok(sparse.into())
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.lt(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, Block::lt_scalar, cmp);
        Ok(sparse.into())
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.le(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, Block::le_scalar, cmp);
        Ok(sparse.into())
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        let cmp = |l: Number, r: Number| if l.ne(&r) { 1 } else { 0 };
        let sparse = SparseCompareConst::new(self.accessor.into(), other, Block::ne_scalar, cmp);
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
            |l, r| if r == T::zero() { T::zero() } else { l / r },
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
impl<FE, A> TensorRead for SparseTensor<FE, A>
where
    FE: ThreadSafe,
    A: SparseInstance + TensorPermitRead,
    Number: From<A::DType>,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        let _permit = self
            .accessor
            .read_permit(txn_id, coord.to_vec().into())
            .await?;

        self.accessor
            .read_value(txn_id, coord)
            .map_ok(Number::from)
            .await
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
        let filled_at = self
            .accessor
            .filled_at(txn_id, Range::default(), axes)
            .await?;

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
        let mut filled_at = self
            .accessor
            .filled_at(txn_id, Range::default(), axes)
            .await?;

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
        let blocks = self
            .accessor
            .blocks(txn_id, Range::default(), vec![])
            .await?;

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
        let blocks = self
            .accessor
            .blocks(txn_id, Range::default(), vec![])
            .await?;

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

        let blocks = self
            .accessor
            .blocks(txn_id, Range::default(), vec![])
            .await?;

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

        let blocks = self
            .accessor
            .blocks(txn_id, Range::default(), vec![])
            .await?;

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

// macro_rules! base_dispatch {
//     ($this:ident, $var:ident, $bool:expr, $complex:expr, $general:expr) => {
//         match $this {
//             SparseBase::Bool($var) => $bool,
//             SparseBase::C32($var) => $complex,
//             SparseBase::C64($var) => $complex,
//             SparseBase::F32($var) => $general,
//             SparseBase::F64($var) => $general,
//             SparseBase::I16($var) => $general,
//             SparseBase::I32($var) => $general,
//             SparseBase::I64($var) => $general,
//             SparseBase::U8($var) => $general,
//             SparseBase::U16($var) => $general,
//             SparseBase::U32($var) => $general,
//             SparseBase::U64($var) => $general,
//         }
//     };
// }
//
// pub enum SparseBase<Txn, FE> {
//     Bool(base::SparseBase<Txn, FE, u8>),
//     C32(
//         (
//             base::SparseBase<Txn, FE, f32>,
//             base::SparseBase<Txn, FE, f32>,
//         ),
//     ),
//     C64(
//         (
//             base::SparseBase<Txn, FE, f64>,
//             base::SparseBase<Txn, FE, f64>,
//         ),
//     ),
//     F32(base::SparseBase<Txn, FE, f32>),
//     F64(base::SparseBase<Txn, FE, f64>),
//     I16(base::SparseBase<Txn, FE, i16>),
//     I32(base::SparseBase<Txn, FE, i32>),
//     I64(base::SparseBase<Txn, FE, i64>),
//     U8(base::SparseBase<Txn, FE, u8>),
//     U16(base::SparseBase<Txn, FE, u16>),
//     U32(base::SparseBase<Txn, FE, u32>),
//     U64(base::SparseBase<Txn, FE, u64>),
// }
//
// impl<Txn, FE> SparseBase<Txn, FE>
// where
//     FE: AsType<Node> + ThreadSafe,
// {
//     pub fn into_view(self) -> TCResult<SparseView<FE>> {
//          todo!()
//     }
// }
//
// impl<Txn, FE> TensorInstance for SparseBase<Txn, FE>
// where
//     Txn: ThreadSafe,
//     FE: ThreadSafe,
// {
//     fn dtype(&self) -> NumberType {
//         match self {
//             Self::Bool(this) => this.dtype(),
//             Self::C32(_) => NumberType::Complex(ComplexType::C32),
//             Self::C64(_) => NumberType::Complex(ComplexType::C64),
//             Self::F32(this) => this.dtype(),
//             Self::F64(this) => this.dtype(),
//             Self::I16(this) => this.dtype(),
//             Self::I32(this) => this.dtype(),
//             Self::I64(this) => this.dtype(),
//             Self::U8(this) => this.dtype(),
//             Self::U16(this) => this.dtype(),
//             Self::U32(this) => this.dtype(),
//             Self::U64(this) => this.dtype(),
//         }
//     }
//
//     fn shape(&self) -> &Shape {
//         base_dispatch!(self, this, this.shape(), this.0.shape(), this.shape())
//     }
// }
//
// #[async_trait]
// impl<Txn, FE> TensorRead for SparseBase<Txn, FE>
// where
//     Txn: Transaction<FE>,
//     FE: AsType<Node> + ThreadSafe,
// {
//     async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
//         base_dispatch!(
//             self,
//             this,
//             this.read_value(txn_id, coord).await,
//             ComplexRead::read_value((Self::from(this.0), Self::from(this.1)), txn_id, coord).await,
//             this.read_value(txn_id, coord).await
//         )
//     }
// }
//
// #[async_trait]
// impl<Txn, FE> TensorWrite for SparseBase<Txn, FE>
// where
//     Txn: Transaction<FE>,
//     FE: AsType<Node> + ThreadSafe,
// {
//     async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()> {
//         base_dispatch!(
//             self,
//             this,
//             this.write_value(txn_id, range, value).await,
//             {
//                 let (r_value, i_value) = Complex::cast_from(value).into();
//
//                 try_join!(
//                     this.0
//                         .write_value(txn_id, range.clone(), Number::Float(r_value)),
//                     this.1.write_value(txn_id, range, Number::Float(i_value))
//                 )?;
//
//                 Ok(())
//             },
//             this.write_value(txn_id, range, value).await
//         )
//     }
//
//     async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
//         base_dispatch!(
//             self,
//             this,
//             this.write_value_at(txn_id, coord, value).await,
//             {
//                 let (r_value, i_value) = Complex::cast_from(value).into();
//
//                 try_join!(
//                     this.0
//                         .write_value_at(txn_id, coord.to_vec(), Number::Float(r_value)),
//                     this.1
//                         .write_value_at(txn_id, coord.to_vec(), Number::Float(i_value))
//                 )?;
//
//                 Ok(())
//             },
//             this.write_value_at(txn_id, coord, value).await
//         )
//     }
// }
//
// #[async_trait]
// impl<Txn, FE> TensorWriteDual<SparseView<FE>> for SparseBase<Txn, FE>
// where
//     Txn: Transaction<FE>,
//     FE: AsType<Node> + ThreadSafe,
// {
//     async fn write(self, txn_id: TxnId, range: Range, value: SparseView<FE>) -> TCResult<()> {
//         match (self, value) {
//             (Self::Bool(this), SparseView::Bool(that)) => this.write(txn_id, range, that).await,
//             (Self::C32((lr, li)), SparseView::C32((rr, ri))) => {
//                 try_join!(
//                     TensorWriteDual::write(lr, txn_id, range.clone(), rr),
//                     TensorWriteDual::write(li, txn_id, range, ri),
//                 )?;
//
//                 Ok(())
//             }
//             (Self::C64((lr, li)), SparseView::C64((rr, ri))) => {
//                 try_join!(
//                     TensorWriteDual::write(lr, txn_id, range.clone(), rr),
//                     TensorWriteDual::write(li, txn_id, range, ri),
//                 )?;
//
//                 Ok(())
//             }
//             (Self::F32(this), SparseView::F32(that)) => this.write(txn_id, range, that).await,
//             (Self::F64(this), SparseView::F64(that)) => this.write(txn_id, range, that).await,
//             (Self::I16(this), SparseView::I16(that)) => this.write(txn_id, range, that).await,
//             (Self::I32(this), SparseView::I32(that)) => this.write(txn_id, range, that).await,
//             (Self::I64(this), SparseView::I64(that)) => this.write(txn_id, range, that).await,
//             (Self::U8(this), SparseView::U8(that)) => this.write(txn_id, range, that).await,
//             (Self::U16(this), SparseView::U16(that)) => this.write(txn_id, range, that).await,
//             (Self::U32(this), SparseView::U32(that)) => this.write(txn_id, range, that).await,
//             (Self::U64(this), SparseView::U64(that)) => this.write(txn_id, range, that).await,
//             (this, that) => {
//                 let value = TensorCast::cast_into(that, this.dtype())?;
//                 this.write(txn_id, range, value).await
//             }
//         }
//     }
// }
//
// impl<Txn, FE> From<base::SparseBase<Txn, FE, f32>> for SparseBase<Txn, FE> {
//     fn from(base: base::SparseBase<Txn, FE, f32>) -> Self {
//         Self::F32(base)
//     }
// }
//
// impl<Txn, FE> From<base::SparseBase<Txn, FE, f64>> for SparseBase<Txn, FE> {
//     fn from(base: base::SparseBase<Txn, FE, f64>) -> Self {
//         Self::F64(base)
//     }
// }

#[inline]
pub fn sparse_from<FE, A, T>(tensor: SparseTensor<FE, A>) -> SparseTensor<FE, SparseAccess<FE, T>>
where
    A: Into<SparseAccess<FE, T>>,
    T: CDatatype,
{
    SparseTensor::from_access(tensor.into_inner())
}
