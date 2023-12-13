//! A sparse tensor

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;
use std::pin::Pin;

use async_trait::async_trait;
use b_table::Row;
use collate::Collate;
use destream::de;
use futures::{join, try_join, Stream, StreamExt, TryFutureExt, TryStreamExt};
use ha_ndarray::*;
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_transact::fs;
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{
    Complex, ComplexType, DType, FloatType, IntType, Number, NumberClass, NumberCollator,
    NumberInstance, NumberType, UIntType, ValueType,
};
use tcgeneric::{Instance, TCBoxTryStream, ThreadSafe};

use super::block::Block;
use super::complex::ComplexRead;
use super::dense::{DenseAccess, DenseAccessCast, DenseCacheFile, DenseSparse, DenseTensor};

use super::{
    Axes, AxisRange, Coord, Range, Schema as TensorSchema, Shape, TensorBoolean,
    TensorBooleanConst, TensorCast, TensorCompare, TensorCompareConst, TensorCond, TensorConvert,
    TensorInstance, TensorMath, TensorMathConst, TensorPermitRead, TensorRead, TensorReduce,
    TensorTransform, TensorType, TensorUnary, TensorUnaryBoolean, TensorWrite, TensorWriteDual,
    IMAG, REAL,
};

pub use access::*;
use itertools::Itertools;
pub use schema::{IndexSchema, Schema};
pub use view::*;

mod access;
mod base;
mod file;
mod schema;
mod stream;
mod view;

mod reduce {
    use tcgeneric::TCBoxTryFuture;

    use super::*;

    pub(super) fn reduce_all<A>(txn_id: TxnId, accessor: A) -> TCBoxTryFuture<'static, bool>
    where
        A: SparseInstance + TensorPermitRead,
    {
        Box::pin(async move {
            let _permit = accessor.read_permit(txn_id, Range::default()).await?;

            let axes = (0..accessor.ndim()).into_iter().collect();
            let mut affected = Range::all(accessor.shape()).affected();
            let mut filled_at = accessor
                .filled_at(txn_id, Range::default(), Axes::default(), axes)
                .await?;

            while let Some(actual) = filled_at.try_next().await? {
                let expected = affected.next().expect("coord");
                if actual.as_slice() != expected.as_slice() {
                    return Ok(false);
                }
            }

            Ok(affected.next().is_none())
        })
    }

    pub(super) fn reduce_any<A>(txn_id: TxnId, accessor: A) -> TCBoxTryFuture<'static, bool>
    where
        A: SparseInstance + TensorPermitRead,
    {
        Box::pin(async move {
            let _permit = accessor.read_permit(txn_id, Range::default()).await?;

            let axes = (0..accessor.ndim()).into_iter().collect();
            let mut filled_at = accessor
                .filled_at(txn_id, Range::default(), Axes::default(), axes)
                .await?;

            filled_at.try_next().map_ok(|r| r.is_some()).await
        })
    }

    pub(super) fn reduce_max<A>(txn_id: TxnId, accessor: A) -> TCBoxTryFuture<'static, A::DType>
    where
        A: SparseInstance + TensorPermitRead,
        A::DType: Into<Number>,
    {
        Box::pin(async move {
            let _permit = accessor.read_permit(txn_id, Range::default()).await?;

            // TODO: use a type-specific collator
            let collator = NumberCollator::default();
            let blocks = accessor
                .blocks(txn_id, Range::default(), Axes::default())
                .await?;

            blocks
                .map(|result| {
                    result.and_then(|(_coords, values)| values.max_all().map_err(TCError::from))
                })
                .try_fold(A::DType::MIN, move |max, block_max| {
                    let max = match collator.cmp(&max.into(), &block_max.into()) {
                        Ordering::Less => block_max,
                        Ordering::Equal | Ordering::Greater => max,
                    };

                    futures::future::ready(Ok(max))
                })
                .await
        })
    }

    pub(super) fn reduce_min<A>(txn_id: TxnId, accessor: A) -> TCBoxTryFuture<'static, A::DType>
    where
        A: SparseInstance + TensorPermitRead,
        A::DType: Into<Number>,
    {
        Box::pin(async move {
            let _permit = accessor.read_permit(txn_id, Range::default()).await?;

            // TODO: use a type-specific collator
            let collator = NumberCollator::default();
            let blocks = accessor
                .blocks(txn_id, Range::default(), Axes::default())
                .await?;

            blocks
                .map(|result| {
                    result.and_then(|(_coords, values)| values.max_all().map_err(TCError::from))
                })
                .try_fold(A::DType::MAX, move |max, block_max| {
                    let max = match collator.cmp(&max.into(), &block_max.into()) {
                        Ordering::Less => block_max,
                        Ordering::Equal | Ordering::Greater => max,
                    };

                    futures::future::ready(Ok(max))
                })
                .await
        })
    }

    pub(super) fn reduce_product<A>(txn_id: TxnId, accessor: A) -> TCBoxTryFuture<'static, A::DType>
    where
        A: SparseInstance + TensorPermitRead + Clone + fmt::Debug,
    {
        Box::pin(async move {
            log::debug!("calculate the total product of {accessor:?}");

            let _permit = accessor.read_permit(txn_id, Range::default()).await?;

            if !reduce_all(txn_id, accessor.clone()).await? {
                return Ok(A::DType::ZERO);
            } else {
                log::trace!("all elements in {accessor:?} are filled...");
            }

            let blocks = accessor
                .blocks(txn_id, Range::default(), Axes::default())
                .await?;

            let product: A::DType = blocks
                .map(|result| {
                    result.and_then(|(_coords, values)| values.product_all().map_err(TCError::from))
                })
                .try_fold(
                    A::DType::ONE,
                    move |product: A::DType, block_product: A::DType| {
                        futures::future::ready(Ok(A::DType::mul(product, block_product)))
                    },
                )
                .await?;

            Ok(product)
        })
    }

    pub(super) fn reduce_sum<A>(txn_id: TxnId, accessor: A) -> TCBoxTryFuture<'static, A::DType>
    where
        A: SparseInstance + TensorPermitRead,
    {
        Box::pin(async move {
            let _permit = accessor.read_permit(txn_id, Range::default()).await?;

            let blocks = accessor
                .blocks(txn_id, Range::default(), Axes::default())
                .await?;

            blocks
                .map(|result| {
                    result.and_then(|(_coords, values)| values.sum_all().map_err(TCError::from))
                })
                .try_fold(A::DType::ZERO, move |sum, block_sum| {
                    futures::future::ready(Ok(A::DType::add(sum, block_sum)))
                })
                .await
        })
    }
}

const BLOCK_SIZE: usize = 4_096;

pub type Blocks<T, C, V> =
    Pin<Box<dyn Stream<Item = Result<(Array<u64, C>, Array<T, V>), TCError>> + Send>>;
pub type Elements<T> = Pin<Box<dyn Stream<Item = Result<(Coord, T), TCError>> + Send>>;
pub type Node = b_table::Node<Number>;

#[async_trait]
pub trait SparseInstance: TensorInstance + fmt::Debug {
    type CoordBlock: Access<u64>;
    type ValueBlock: Access<Self::DType>;
    type Blocks: Stream<
            Item = Result<
                (
                    Array<u64, Self::CoordBlock>,
                    Array<Self::DType, Self::ValueBlock>,
                ),
                TCError,
            >,
        > + Send;
    type DType: CType + DType;

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
        order: Axes,
        axes: Axes,
    ) -> TCResult<TCBoxTryStream<'static, Coord>>
    where
        Self: Sized,
    {
        log::debug!("{:?} filled at {:?}...", self, axes);

        self.shape().validate_axes(&axes, false)?;
        self.shape().validate_axes(&order, false)?;

        if axes.is_empty() {
            #[cfg(debug_assertions)]
            panic!("cannot group an empty set of axes");

            #[cfg(not(debug_assertions))]
            Err(bad_request!("cannot group an empty set of axes"))
        } else if (0..self.ndim()).into_iter().all(|x| axes.contains(&x)) {
            let elements = self.elements(txn_id, range, order).await?;
            let filled_at = elements.map_ok(move |(coord, _value)| {
                axes.iter().copied().map(|x| coord[x]).collect::<Coord>()
            });

            Ok(Box::pin(filled_at))
        } else if axes.iter().zip(&order).all(|(x, o)| x == o) {
            let ndim = self.ndim();
            let order = order
                .into_iter() // make sure the selected axes are ordered
                .chain(axes.iter().copied()) // then make sure all axes are present
                .chain(0..ndim) // then remove duplicates
                .unique()
                .collect::<Axes>();

            // TODO: use Table::group_by to implement this
            let filled_at = self
                .elements(txn_id, range, order)
                .map_ok(|elements| stream::FilledAt::new(elements, axes, ndim))
                .await?;

            Ok(Box::pin(filled_at))
        } else {
            #[cfg(debug_assertions)]
            panic!(
                "cannot group axes {axes:?} of {this:?} by order {order:?}",
                this = self
            );

            #[cfg(not(debug_assertions))]
            Err(bad_request!(
                "cannot group axes {axes:?} of {this:?} by order {order:?}",
                this = self
            ))
        }
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError>;
}

pub struct SparseTensor<Txn, FE, A> {
    accessor: A,
    phantom: PhantomData<(Txn, FE)>,
}

impl<Txn, FE, A: Clone> Clone for SparseTensor<Txn, FE, A> {
    fn clone(&self) -> Self {
        Self {
            accessor: self.accessor.clone(),
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE, A> SparseTensor<Txn, FE, A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<Txn, FE, T: CType> SparseTensor<Txn, FE, SparseAccess<Txn, FE, T>> {
    pub fn from_access<A: Into<SparseAccess<Txn, FE, T>>>(accessor: A) -> Self {
        Self {
            accessor: accessor.into(),
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE, A> SparseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: SparseInstance + TensorPermitRead,
{
    pub async fn into_elements(
        self,
        txn_id: TxnId,
    ) -> TCResult<stream::Elements<Elements<A::DType>>> {
        let permit = self.accessor.read_permit(txn_id, Range::default()).await?;
        let elements = self
            .accessor
            .elements(txn_id, Range::default(), Axes::default())
            .await?;

        Ok(stream::Elements::new(permit, elements))
    }
}

impl<Txn, FE, A> TensorInstance for SparseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<Txn, FE, L, R> TensorBoolean<SparseTensor<Txn, FE, R>> for SparseTensor<Txn, FE, L>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    L: SparseInstance + Into<SparseAccess<Txn, FE, L::DType>>,
    R: SparseInstance<DType = L::DType> + Into<SparseAccess<Txn, FE, R::DType>>,
    Number: From<L::DType> + From<R::DType>,
    SparseAccessCast<Txn, FE>: From<SparseAccess<Txn, FE, L::DType>>,
{
    type Combine = SparseTensor<Txn, FE, SparseCompare<Txn, FE, u8>>;
    type LeftCombine = SparseTensor<Txn, FE, SparseCompareLeft<Txn, FE, u8>>;

    fn and(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
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

    fn or(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Combine> {
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

    fn xor(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Combine> {
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

impl<Txn, FE, A> TensorBooleanConst for SparseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance + Into<SparseAccess<Txn, FE, A::DType>>,
    DenseAccess<Txn, FE, A::DType>: From<DenseSparse<A>>,
    DenseAccessCast<Txn, FE>: From<DenseAccess<Txn, FE, A::DType>>,
    SparseAccessCast<Txn, FE>: From<SparseAccess<Txn, FE, A::DType>>,
{
    type Combine = SparseTensor<Txn, FE, SparseCompareConst<Txn, FE, u8>>;

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

impl<Txn, FE, Cond, Then, OrElse, T>
    TensorCond<SparseTensor<Txn, FE, Then>, SparseTensor<Txn, FE, OrElse>>
    for SparseTensor<Txn, FE, Cond>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    Cond: SparseInstance<DType = u8> + fmt::Debug,
    Then: SparseInstance<DType = T> + fmt::Debug,
    OrElse: SparseInstance<DType = T> + fmt::Debug,
{
    type Cond = SparseTensor<Txn, FE, SparseCond<Cond, Then, OrElse>>;

    fn cond(
        self,
        then: SparseTensor<Txn, FE, Then>,
        or_else: SparseTensor<Txn, FE, OrElse>,
    ) -> TCResult<Self::Cond> {
        SparseCond::new(self.accessor, then.accessor, or_else.accessor).map(SparseTensor::from)
    }
}

impl<Txn, FE, A> TensorConvert for SparseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance,
{
    type Dense = DenseTensor<Txn, FE, DenseSparse<A>>;
    type Sparse = Self;

    fn into_dense(self) -> Self::Dense {
        DenseSparse::from(self.accessor).into()
    }

    fn into_sparse(self) -> Self::Sparse {
        self
    }
}

impl<Txn, FE, L, R> TensorCompare<SparseTensor<Txn, FE, R>> for SparseTensor<Txn, FE, L>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    L: SparseInstance + Into<SparseAccess<Txn, FE, L::DType>> + fmt::Debug,
    R: SparseInstance<DType = L::DType> + Into<SparseAccess<Txn, FE, R::DType>> + fmt::Debug,
    SparseAccessCast<Txn, FE>: From<SparseAccess<Txn, FE, L::DType>>,
{
    type Compare = SparseTensor<Txn, FE, SparseCompare<Txn, FE, u8>>;

    fn eq(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        Err(bad_request!(
            "cannot compare {:?} with {:?} because the result would be dense",
            self,
            other
        ))
    }

    fn gt(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
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

    fn ge(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        Err(bad_request!(
            "cannot compare {:?} with {:?} because the result would be dense",
            self,
            other
        ))
    }

    fn lt(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
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

    fn le(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
        Err(bad_request!(
            "cannot compare {:?} with {:?} because the result would be dense",
            self,
            other
        ))
    }

    fn ne(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Compare> {
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

impl<Txn, FE, A> TensorCompareConst for SparseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance + Into<SparseAccess<Txn, FE, A::DType>>,
    SparseAccessCast<Txn, FE>: From<SparseAccess<Txn, FE, A::DType>>,
{
    type Compare = SparseTensor<Txn, FE, SparseCompareConst<Txn, FE, u8>>;

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

impl<Txn, FE, L, R, T> TensorMath<SparseTensor<Txn, FE, R>> for SparseTensor<Txn, FE, L>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    L: SparseInstance<DType = T>,
    R: SparseInstance<DType = T>,
    T: CType + DType,
{
    type Combine = SparseTensor<Txn, FE, SparseCombine<L, R, T>>;
    type LeftCombine = SparseTensor<Txn, FE, SparseCombineLeft<L, R, T>>;

    fn add(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Combine> {
        SparseCombine::new(
            self.accessor,
            other.accessor,
            |l, r| l.add(r).map(Array::from).map_err(TCError::from),
            |l, r| T::add(l, r),
        )
        .map(SparseTensor::from)
    }

    fn div(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        SparseCombineLeft::new(
            self.accessor,
            other.accessor,
            |l, r| l.div(r).map(Array::from).map_err(TCError::from),
            |l, r| T::div(l, r),
        )
        .map(SparseTensor::from)
    }

    fn log(self, base: SparseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn log<T: CType>(left: ArrayAccess<T>, right: ArrayAccess<T>) -> TCResult<ArrayAccess<T>> {
            let right = right.cast()?;
            left.log(right).map(Array::from).map_err(TCError::from)
        }

        SparseCombineLeft::new(self.accessor, base.accessor, log, |l, r| {
            T::from_f64(T::to_f64(l).log(T::to_f64(r)))
        })
        .map(SparseTensor::from)
    }

    fn mul(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        SparseCombineLeft::new(
            self.accessor,
            other.accessor,
            |l, r| l.mul(r).map(Array::from).map_err(TCError::from),
            |l, r| T::mul(l, r),
        )
        .map(SparseTensor::from)
    }

    fn pow(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::LeftCombine> {
        fn pow<T: CType>(left: ArrayAccess<T>, right: ArrayAccess<T>) -> TCResult<ArrayAccess<T>> {
            let right = right.cast()?;
            left.pow(right).map(Array::from).map_err(TCError::from)
        }

        SparseCombineLeft::new(self.accessor, other.accessor, pow, |l, r| {
            T::from_f64(T::to_f64(l).pow(T::to_f64(r)))
        })
        .map(SparseTensor::from)
    }

    fn sub(self, other: SparseTensor<Txn, FE, R>) -> TCResult<Self::Combine> {
        SparseCombine::new(
            self.accessor,
            other.accessor,
            |l, r| l.sub(r).map(Array::from).map_err(TCError::from),
            |l, r| T::sub(l, r),
        )
        .map(SparseTensor::from)
    }
}

impl<Txn, FE, A> TensorMathConst for SparseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: SparseInstance,
    A::DType: CastFrom<Number>,
    <A::DType as CType>::Float: CastFrom<Number>,
    Number: From<A::DType>,
{
    type Combine = SparseTensor<Txn, FE, SparseCombineConst<A, A::DType>>;

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
                |l, r| A::DType::div(l, r.cast_into()),
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
            |l, r| A::DType::mul(l, r.cast_into()),
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
impl<Txn, FE, A> TensorRead for SparseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: ThreadSafe,
    A: SparseInstance + TensorPermitRead,
    Number: From<A::DType>,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        let _permit = self
            .accessor
            .read_permit(txn_id, coord.clone().into())
            .await?;

        self.accessor
            .read_value(txn_id, coord)
            .map_ok(Number::from)
            .await
    }
}

#[async_trait]
impl<Txn, FE, A> TensorReduce for SparseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<A::DType>> + AsType<Node>,
    A: SparseInstance + TensorPermitRead + Clone,
    A::DType: fmt::Debug,
    Buffer<A::DType>: de::FromStream<Context = ()>,
    Number: From<A::DType> + CastInto<A::DType>,
    SparseAccess<Txn, FE, A::DType>: From<A>,
{
    type Reduce = SparseTensor<Txn, FE, SparseReduce<Txn, FE, A::DType>>;

    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        reduce::reduce_all(txn_id, self.accessor).await
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        reduce::reduce_any(txn_id, self.accessor).await
    }

    fn max(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        SparseReduce::new(self.accessor, axes, keepdims, reduce::reduce_max).map(SparseTensor::from)
    }

    async fn max_all(self, txn_id: TxnId) -> TCResult<Number> {
        reduce::reduce_max(txn_id, self.accessor)
            .map_ok(Number::from)
            .await
    }

    fn min(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        SparseReduce::new(self.accessor, axes, keepdims, reduce::reduce_min).map(SparseTensor::from)
    }

    async fn min_all(self, txn_id: TxnId) -> TCResult<Number> {
        reduce::reduce_min(txn_id, self.accessor)
            .map_ok(Number::from)
            .await
    }

    fn product(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        SparseReduce::new(self.accessor, axes, keepdims, reduce::reduce_product)
            .map(SparseTensor::from)
    }

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number> {
        reduce::reduce_product(txn_id, self.accessor)
            .map_ok(Number::from)
            .await
    }

    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        SparseReduce::new(self.accessor, axes, keepdims, reduce::reduce_sum).map(SparseTensor::from)
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        reduce::reduce_sum(txn_id, self.accessor)
            .map_ok(Number::from)
            .await
    }
}

impl<Txn, FE, A> TensorTransform for SparseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance + Into<SparseAccess<Txn, FE, A::DType>>,
    A::DType: CastFrom<Number> + fmt::Debug,
    Number: From<A::DType>,
{
    type Broadcast = SparseTensor<Txn, FE, SparseBroadcast<Txn, FE, A::DType>>;
    type Expand = SparseTensor<Txn, FE, SparseExpand<A>>;
    type Reshape = SparseTensor<Txn, FE, SparseReshape<A>>;
    type Slice = SparseTensor<Txn, FE, SparseSlice<A>>;
    type Transpose = SparseTensor<Txn, FE, SparseTranspose<A>>;

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

impl<Txn, FE, A> TensorUnary for SparseTensor<Txn, FE, A>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    A: SparseInstance,
{
    type Unary = SparseTensor<Txn, FE, SparseUnary<A, A::DType>>;

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

impl<Txn, FE, A> TensorUnaryBoolean for SparseTensor<Txn, FE, A>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
    A: SparseInstance + Into<SparseAccess<Txn, FE, A::DType>>,
    SparseAccessCast<Txn, FE>: From<SparseAccess<Txn, FE, A::DType>>,
{
    type Unary = SparseTensor<Txn, FE, SparseUnaryCast<Txn, FE, u8>>;

    fn not(self) -> TCResult<Self::Unary> {
        let accessor = SparseUnaryCast::new(
            self.accessor.into(),
            |block| block.not(),
            |n| if n == NumberType::Number.zero() { 1 } else { 0 },
        );

        Ok(accessor.into())
    }
}

impl<Txn, FE, A> From<A> for SparseTensor<Txn, FE, A> {
    fn from(accessor: A) -> Self {
        Self {
            accessor,
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE, A: fmt::Debug> fmt::Debug for SparseTensor<Txn, FE, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.accessor.fmt(f)
    }
}

pub enum SparseBase<Txn, FE> {
    Bool(base::SparseBase<Txn, FE, u8>),
    C32(
        (
            base::SparseBase<Txn, FE, f32>,
            base::SparseBase<Txn, FE, f32>,
        ),
    ),
    C64(
        (
            base::SparseBase<Txn, FE, f64>,
            base::SparseBase<Txn, FE, f64>,
        ),
    ),
    F32(base::SparseBase<Txn, FE, f32>),
    F64(base::SparseBase<Txn, FE, f64>),
    I16(base::SparseBase<Txn, FE, i16>),
    I32(base::SparseBase<Txn, FE, i32>),
    I64(base::SparseBase<Txn, FE, i64>),
    U8(base::SparseBase<Txn, FE, u8>),
    U16(base::SparseBase<Txn, FE, u16>),
    U32(base::SparseBase<Txn, FE, u32>),
    U64(base::SparseBase<Txn, FE, u64>),
}

impl<Txn, FE> Clone for SparseBase<Txn, FE> {
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

impl<Txn: ThreadSafe, FE: ThreadSafe> Instance for SparseBase<Txn, FE> {
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Sparse
    }
}

macro_rules! base_dispatch {
    ($this:ident, $var:ident, $bool:expr, $complex:expr, $general:expr) => {
        match $this {
            SparseBase::Bool($var) => $bool,
            SparseBase::C32($var) => $complex,
            SparseBase::C64($var) => $complex,
            SparseBase::F32($var) => $general,
            SparseBase::F64($var) => $general,
            SparseBase::I16($var) => $general,
            SparseBase::I32($var) => $general,
            SparseBase::I64($var) => $general,
            SparseBase::U8($var) => $general,
            SparseBase::U16($var) => $general,
            SparseBase::U32($var) => $general,
            SparseBase::U64($var) => $general,
        }
    };
}

macro_rules! base_view_dispatch {
    ($self:ident, $other:ident, $this:ident, $that:ident, $bool:expr, $complex:expr, $general:expr, $mismatch:expr) => {
        match ($self, $other) {
            (SparseBase::Bool($this), SparseView::Bool($that)) => $bool,
            (SparseBase::C32($this), SparseView::C32($that)) => $complex,
            (SparseBase::C64($this), SparseView::C64($that)) => $complex,
            (SparseBase::F32($this), SparseView::F32($that)) => $general,
            (SparseBase::F64($this), SparseView::F64($that)) => $general,
            (SparseBase::I16($this), SparseView::I16($that)) => $general,
            (SparseBase::I32($this), SparseView::I32($that)) => $general,
            (SparseBase::I64($this), SparseView::I64($that)) => $general,
            (SparseBase::U8($this), SparseView::U8($that)) => $general,
            (SparseBase::U16($this), SparseView::U16($that)) => $general,
            (SparseBase::U32($this), SparseView::U32($that)) => $general,
            (SparseBase::U64($this), SparseView::U64($that)) => $general,
            ($this, $that) => $mismatch,
        }
    };
}

impl<Txn, FE> TensorInstance for SparseBase<Txn, FE>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
{
    fn dtype(&self) -> NumberType {
        match self {
            Self::Bool(this) => this.dtype(),
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
        base_dispatch!(self, this, this.shape(), this.0.shape(), this.shape())
    }
}

#[async_trait]
impl<Txn, FE> TensorRead for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        base_dispatch!(
            self,
            this,
            this.read_value(txn_id, coord).map_ok(Number::from).await,
            ComplexRead::read_value((Self::from(this.0), Self::from(this.1)), txn_id, coord).await,
            this.read_value(txn_id, coord).map_ok(Number::from).await
        )
    }
}

#[async_trait]
impl<Txn, FE> TensorWrite for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()> {
        if bool::cast_from(value) {
            return Err(bad_request!("cannot write a scalar value {} to a sparse range {:?} because the result would be dense", value, range));
        }

        base_dispatch!(
            self,
            this,
            {
                let mut guard = this.write().await;
                guard.clear(txn_id, range).await
            },
            {
                let (mut r_guard, mut i_guard) = join!(this.0.write(), this.1.write());

                try_join!(
                    r_guard.clear(txn_id, range.clone()),
                    i_guard.clear(txn_id, range)
                )?;

                Ok(())
            },
            {
                let mut guard = this.write().await;
                guard.clear(txn_id, range).await
            }
        )
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        base_dispatch!(
            self,
            this,
            {
                let mut guard = this.write().await;
                guard.write_value(txn_id, coord, value.cast_into()).await
            },
            {
                let (r_value, i_value) = Complex::cast_from(value).into();

                let (mut r_guard, mut i_guard) = join!(this.0.write(), this.1.write());

                try_join!(
                    r_guard.write_value(txn_id, coord.clone(), r_value.cast_into()),
                    i_guard.write_value(txn_id, coord, i_value.cast_into())
                )?;

                Ok(())
            },
            {
                let mut guard = this.write().await;
                guard.write_value(txn_id, coord, value.cast_into()).await
            }
        )
    }
}

#[async_trait]
impl<Txn, FE> TensorWriteDual<SparseView<Txn, FE>> for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + ThreadSafe,
{
    async fn write(self, txn_id: TxnId, range: Range, value: SparseView<Txn, FE>) -> TCResult<()> {
        base_view_dispatch!(
            self,
            value,
            this,
            that,
            {
                if this.shape().is_covered_by(&range) {
                    let mut guard = this.write().await;
                    guard.overwrite(txn_id, that.accessor).await
                } else {
                    let slice = SparseSlice::new(this.clone(), range)?;
                    let mut guard = slice.write().await;
                    guard.overwrite(txn_id, that.accessor).await
                }
            },
            {
                debug_assert_eq!(this.0.shape(), this.1.shape());
                if this.0.shape().is_covered_by(&range) {
                    let (mut r_guard, mut i_guard) = join!(this.0.write(), this.1.write());

                    try_join!(
                        r_guard.overwrite(txn_id, that.0.accessor),
                        i_guard.overwrite(txn_id, that.1.accessor)
                    )?;

                    Ok(())
                } else {
                    let r_slice = SparseSlice::new(this.0.clone(), range.clone())?;
                    let i_slice = SparseSlice::new(this.1.clone(), range)?;

                    let (mut r_guard, mut i_guard) = join!(r_slice.write(), i_slice.write());

                    try_join!(
                        r_guard.overwrite(txn_id, that.0.accessor),
                        i_guard.overwrite(txn_id, that.1.accessor)
                    )?;

                    Ok(())
                }
            },
            {
                if this.shape().is_covered_by(&range) {
                    let mut guard = this.write().await;
                    guard.overwrite(txn_id, that.accessor).await
                } else {
                    let slice = SparseSlice::new(this.clone(), range)?;
                    let mut guard = slice.write().await;
                    guard.overwrite(txn_id, that.accessor).await
                }
            },
            {
                let value = TensorCast::cast_into(that, this.dtype())?;
                this.write(txn_id, range, value).await
            }
        )
    }
}

#[async_trait]
impl<Txn, FE> Transact for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + for<'a> fs::FileSave<'a> + Clone,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        base_dispatch!(
            self,
            this,
            this.commit(txn_id).await,
            {
                join!(this.0.commit(txn_id), this.1.commit(txn_id));
            },
            this.commit(txn_id).await
        )
    }

    async fn rollback(&self, txn_id: &TxnId) {
        base_dispatch!(
            self,
            this,
            this.rollback(txn_id).await,
            {
                join!(this.0.rollback(txn_id), this.1.rollback(txn_id));
            },
            this.rollback(txn_id).await
        )
    }

    async fn finalize(&self, txn_id: &TxnId) {
        base_dispatch!(
            self,
            this,
            this.finalize(txn_id).await,
            {
                join!(this.0.finalize(txn_id), this.1.finalize(txn_id));
            },
            this.finalize(txn_id).await
        )
    }
}

#[async_trait]
impl<Txn, FE> fs::Persist<FE> for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
{
    type Txn = Txn;
    type Schema = TensorSchema;

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let dtype = schema.dtype;
        let shape = schema.shape;
        let schema = Schema::new(shape);

        match dtype {
            NumberType::Bool => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::Bool)
                    .await
            }
            NumberType::Complex(ComplexType::C32) => {
                let (re, im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::SparseBase::create(txn_id, schema.clone(), re),
                    base::SparseBase::create(txn_id, schema, im)
                )?;

                Ok(Self::C32((re, im)))
            }
            NumberType::Complex(ComplexType::C64) => {
                let (re, im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::SparseBase::create(txn_id, schema.clone(), re),
                    base::SparseBase::create(txn_id, schema, im)
                )?;

                Ok(Self::C64((re, im)))
            }
            NumberType::Float(FloatType::F32) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::F32)
                    .await
            }
            NumberType::Float(FloatType::F64) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::F64)
                    .await
            }
            NumberType::Int(IntType::I16) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::I16)
                    .await
            }
            NumberType::Int(IntType::I32) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::I32)
                    .await
            }
            NumberType::Int(IntType::I64) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::I64)
                    .await
            }
            NumberType::UInt(UIntType::U8) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::U8)
                    .await
            }
            NumberType::UInt(UIntType::U16) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::U16)
                    .await
            }
            NumberType::UInt(UIntType::U32) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::U32)
                    .await
            }
            NumberType::UInt(UIntType::U64) => {
                base::SparseBase::create(txn_id, schema, store)
                    .map_ok(Self::U64)
                    .await
            }
            other => Err(bad_request!(
                "cannot create a dense tensor of type {other:?}"
            )),
        }
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let dtype = schema.dtype;
        let shape = schema.shape;
        let schema = Schema::new(shape);

        match dtype {
            NumberType::Bool => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::Bool)
                    .await
            }
            NumberType::Complex(ComplexType::C32) => {
                let (re, im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::SparseBase::load(txn_id, schema.clone(), re),
                    base::SparseBase::load(txn_id, schema, im)
                )?;

                Ok(Self::C32((re, im)))
            }
            NumberType::Complex(ComplexType::C64) => {
                let (re, im) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::SparseBase::load(txn_id, schema.clone(), re),
                    base::SparseBase::load(txn_id, schema, im)
                )?;

                Ok(Self::C64((re, im)))
            }
            NumberType::Float(FloatType::F32) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::F32)
                    .await
            }
            NumberType::Float(FloatType::F64) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::F64)
                    .await
            }
            NumberType::Int(IntType::I16) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::I16)
                    .await
            }
            NumberType::Int(IntType::I32) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::I32)
                    .await
            }
            NumberType::Int(IntType::I64) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::I64)
                    .await
            }
            NumberType::UInt(UIntType::U8) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::U8)
                    .await
            }
            NumberType::UInt(UIntType::U16) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::U16)
                    .await
            }
            NumberType::UInt(UIntType::U32) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::U32)
                    .await
            }
            NumberType::UInt(UIntType::U64) => {
                base::SparseBase::load(txn_id, schema, store)
                    .map_ok(Self::U64)
                    .await
            }
            other => Err(bad_request!("cannot load a dense tensor of type {other:?}")),
        }
    }

    fn dir(&self) -> fs::Inner<FE> {
        base_dispatch!(
            self,
            this,
            this.dir(),
            this.0.dir(), // FIXME: this should return the parent dir!
            this.dir()
        )
    }
}

#[async_trait]
impl<Txn, FE> fs::CopyFrom<FE, SparseView<Txn, FE>> for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn copy_from(
        txn: &Txn,
        store: fs::Dir<FE>,
        instance: SparseView<Txn, FE>,
    ) -> TCResult<Self> {
        match instance {
            SparseView::Bool(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::Bool)
                    .await
            }
            SparseView::C32((re, im)) => {
                let txn_id = *txn.id();
                let (r_dir, i_dir) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::SparseBase::copy_from(txn, r_dir, re.into_inner()),
                    base::SparseBase::copy_from(txn, i_dir, im.into_inner())
                )?;

                Ok(Self::C32((re, im)))
            }
            SparseView::C64((re, im)) => {
                let txn_id = *txn.id();
                let (r_dir, i_dir) = try_join!(
                    store.create_dir(txn_id, REAL.into()),
                    store.create_dir(txn_id, IMAG.into())
                )?;

                let (re, im) = try_join!(
                    base::SparseBase::copy_from(txn, r_dir, re.into_inner()),
                    base::SparseBase::copy_from(txn, i_dir, im.into_inner())
                )?;

                Ok(Self::C64((re, im)))
            }
            SparseView::F32(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::F32)
                    .await
            }
            SparseView::F64(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::F64)
                    .await
            }
            SparseView::I16(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::I16)
                    .await
            }
            SparseView::I32(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::I32)
                    .await
            }
            SparseView::I64(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::I64)
                    .await
            }
            SparseView::U8(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::U8)
                    .await
            }
            SparseView::U16(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::U16)
                    .await
            }
            SparseView::U32(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::U32)
                    .await
            }
            SparseView::U64(that) => {
                base::SparseBase::copy_from(txn, store, that.into_inner())
                    .map_ok(Self::U64)
                    .await
            }
        }
    }
}

#[async_trait]
impl<Txn, FE> fs::Restore<FE> for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        match (self, backup) {
            (Self::Bool(this), Self::Bool(that)) => this.restore(txn_id, that).await,
            (Self::C32((lr, li)), Self::C32((rr, ri))) => {
                try_join!(lr.restore(txn_id, rr), li.restore(txn_id, ri))?;
                Ok(())
            }
            (Self::C64((lr, li)), Self::C64((rr, ri))) => {
                try_join!(lr.restore(txn_id, rr), li.restore(txn_id, ri))?;
                Ok(())
            }
            (Self::F32(this), Self::F32(that)) => this.restore(txn_id, that).await,
            (Self::F64(this), Self::F64(that)) => this.restore(txn_id, that).await,
            (Self::I16(this), Self::I16(that)) => this.restore(txn_id, that).await,
            (Self::I32(this), Self::I32(that)) => this.restore(txn_id, that).await,
            (Self::I64(this), Self::I64(that)) => this.restore(txn_id, that).await,
            (Self::U8(this), Self::U8(that)) => this.restore(txn_id, that).await,
            (Self::U16(this), Self::U16(that)) => this.restore(txn_id, that).await,
            (Self::U32(this), Self::U32(that)) => this.restore(txn_id, that).await,
            (Self::U64(this), Self::U64(that)) => this.restore(txn_id, that).await,
            (this, that) => Err(bad_request!("cannot restore {this:?} from {that:?}")),
        }
    }
}

#[async_trait]
impl<Txn, FE> de::FromStream for SparseBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(SparseVisitor::new(txn)).await
    }
}

impl<Txn, FE> From<base::SparseBase<Txn, FE, f32>> for SparseBase<Txn, FE> {
    fn from(base: base::SparseBase<Txn, FE, f32>) -> Self {
        Self::F32(base)
    }
}

impl<Txn, FE> From<base::SparseBase<Txn, FE, f64>> for SparseBase<Txn, FE> {
    fn from(base: base::SparseBase<Txn, FE, f64>) -> Self {
        Self::F64(base)
    }
}

impl<Txn, FE> From<SparseBase<Txn, FE>> for SparseView<Txn, FE> {
    fn from(base: SparseBase<Txn, FE>) -> Self {
        match base {
            SparseBase::Bool(this) => SparseView::Bool(sparse_from(this.into())),
            SparseBase::C32((re, im)) => {
                SparseView::C32((sparse_from(re.into()), sparse_from(im.into())))
            }
            SparseBase::C64((re, im)) => {
                SparseView::C64((sparse_from(re.into()), sparse_from(im.into())))
            }
            SparseBase::F32(this) => SparseView::F32(sparse_from(this.into())),
            SparseBase::F64(this) => SparseView::F64(sparse_from(this.into())),
            SparseBase::I16(this) => SparseView::I16(sparse_from(this.into())),
            SparseBase::I32(this) => SparseView::I32(sparse_from(this.into())),
            SparseBase::I64(this) => SparseView::I64(sparse_from(this.into())),
            SparseBase::U8(this) => SparseView::U8(sparse_from(this.into())),
            SparseBase::U16(this) => SparseView::U16(sparse_from(this.into())),
            SparseBase::U32(this) => SparseView::U32(sparse_from(this.into())),
            SparseBase::U64(this) => SparseView::U64(sparse_from(this.into())),
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> fmt::Debug for SparseBase<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        base_dispatch!(
            self,
            this,
            this.fmt(f),
            write!(
                f,
                "a complex transactional sparse tensor of type {:?}",
                this.0.dtype()
            ),
            this.fmt(f)
        )
    }
}

struct SparseVisitor<Txn, FE> {
    txn: Txn,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> SparseVisitor<Txn, FE> {
    fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }
}

impl<Txn, FE> SparseVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
{
    async fn create_base<E: de::Error>(
        &self,
        dtype: NumberType,
        shape: Shape,
    ) -> Result<SparseBase<Txn, FE>, E> {
        let store = self.txn.context().clone();

        let txn_id = *self.txn.id();
        let store = fs::Dir::load(txn_id, store)
            .map_err(de::Error::custom)
            .await?;

        fs::Persist::create(txn_id, (dtype, shape).into(), store)
            .map_err(de::Error::custom)
            .await
    }
}

#[async_trait]
impl<Txn, FE> de::Visitor for SparseVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
{
    type Value = SparseBase<Txn, FE>;

    fn expecting() -> &'static str {
        "a sparse tensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let (dtype, shape) = seq.expect_next::<(ValueType, Shape)>(()).await?;
        let dtype = if let ValueType::Number(dtype) = dtype {
            Ok(dtype)
        } else {
            Err(de::Error::invalid_type(dtype, "a type of number"))
        }?;

        let txn = self.txn.clone();
        let shape_clone = shape.clone();
        match dtype {
            NumberType::Bool => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::Bool(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::Complex(ComplexType::C32) => {
                if let Some(visitor) = seq
                    .next_element::<base::SparseComplexBaseVisitor<Txn, FE, f32>>((txn, shape))
                    .await?
                {
                    visitor
                        .end()
                        .map_ok(SparseBase::C32)
                        .map_err(de::Error::custom)
                        .await
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::Complex(ComplexType::C64) => {
                if let Some(visitor) = seq
                    .next_element::<base::SparseComplexBaseVisitor<Txn, FE, f64>>((txn, shape))
                    .await?
                {
                    visitor
                        .end()
                        .map_ok(SparseBase::C64)
                        .map_err(de::Error::custom)
                        .await
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::Float(FloatType::F32) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::F32(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::Float(FloatType::F64) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::F64(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::Int(IntType::I16) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::I16(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::Int(IntType::I32) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::I32(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::Int(IntType::I64) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::I64(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::UInt(UIntType::U8) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::U8(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::UInt(UIntType::U16) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::U16(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::UInt(UIntType::U32) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::U32(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            NumberType::UInt(UIntType::U64) => {
                if let Some(base) = seq.next_element((txn, shape)).await? {
                    Ok(SparseBase::U64(base))
                } else {
                    self.create_base(dtype, shape_clone).await
                }
            }
            other => Err(de::Error::invalid_type(other, "a specific type of number")),
        }
    }
}

#[inline]
pub fn sparse_from<Txn, FE, A, T>(
    tensor: SparseTensor<Txn, FE, A>,
) -> SparseTensor<Txn, FE, SparseAccess<Txn, FE, T>>
where
    A: Into<SparseAccess<Txn, FE, T>>,
    T: CType,
{
    SparseTensor::from_access(tensor.into_inner())
}

#[inline]
fn unwrap_row<T>(mut row: Row<Number>) -> (Coord, T)
where
    Number: CastInto<T> + CastInto<u64>,
{
    let n = row.pop().expect("n").cast_into();
    let coord = row.into_iter().map(|i| i.cast_into()).collect();
    (coord, n)
}

#[inline]
fn table_range(range: &Range) -> Result<b_table::Range<usize, Number>, TCError> {
    let mut table_range = HashMap::with_capacity(range.len());

    for (x, bound) in range.iter().enumerate() {
        match bound {
            AxisRange::At(i) => {
                table_range.insert(x, b_table::ColumnRange::Eq(Number::from(*i)));
            }
            AxisRange::In(axis_range, 1) => {
                let start = Bound::Included(Number::from(axis_range.start));
                let stop = Bound::Excluded(Number::from(axis_range.end));
                table_range.insert(x, b_table::ColumnRange::In((start, stop)));
            }
            bound => {
                return Err(bad_request!(
                    "sparse tensor does not support axis bound {:?}",
                    bound
                ));
            }
        }
    }

    Ok(table_range.into())
}
