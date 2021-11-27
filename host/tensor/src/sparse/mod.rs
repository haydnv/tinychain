use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::pin::Pin;

use afarray::Array;
use async_trait::async_trait;
use destream::{de, en};
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use log::debug;
use safecast::{AsType, CastFrom};

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, File, Hash, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::{FloatType, Number, NumberClass, NumberInstance, NumberType};
use tcgeneric::{Instance, TCBoxTryFuture, TCBoxTryStream};

use super::dense::{BlockListSparse, DenseTensor, PER_BLOCK};
use super::transform;
use super::{
    Bounds, Coord, Phantom, Schema, Shape, Tensor, TensorAccess, TensorBoolean, TensorBooleanConst,
    TensorCompare, TensorCompareConst, TensorDualIO, TensorIO, TensorInstance, TensorMath,
    TensorMathConst, TensorReduce, TensorTransform, TensorType, TensorUnary, ERR_COMPLEX_EXPONENT,
};

use crate::stream::ReadValueAt;
use crate::{TensorDiagonal, TensorPersist};
use access::*;
pub use access::{DenseToSparse, SparseAccess, SparseAccessor, SparseWrite};
pub use table::SparseTable;

mod access;
mod combine;
mod table;

pub type SparseRow = (Coord, Number);
pub type SparseStream<'a> = Pin<Box<dyn Stream<Item = TCResult<SparseRow>> + Send + Unpin + 'a>>;

const ERR_NOT_SPARSE: &str = "The result of the requested operation would not be sparse;\
convert to a DenseTensor first.";

/// A `Tensor` stored as a `Table` of [`Coord`]s and [`Number`] values
#[derive(Clone)]
pub struct SparseTensor<FD, FS, D, T, A> {
    accessor: A,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseTensor<FD, FS, D, T, A> {
    /// Consume this [`SparseTensor`] and return its accessor.
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<FD, FS, D, T, A> Instance for SparseTensor<FD, FS, D, T, A>
where
    Self: Send + Sync,
{
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Sparse
    }
}

type Condensed<FD, FS, D, T, L, R> =
    DenseTensor<FD, FS, D, T, BlockListSparse<FD, FS, D, T, SparseCombinator<FD, FS, D, T, L, R>>>;

impl<FD, FS, D, T, A> SparseTensor<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    fn combine<R: SparseAccess<FD, FS, D, T>>(
        self,
        other: SparseTensor<FD, FS, D, T, R>,
        combinator: fn(Number, Number) -> Number,
    ) -> TCResult<SparseTensor<FD, FS, D, T, SparseCombinator<FD, FS, D, T, A, R>>> {
        if self.shape() != other.shape() {
            return Err(TCError::unsupported(format!(
                "cannot compare Tensors of different shapes: {}, {}",
                self.shape(),
                other.shape()
            )));
        }

        let accessor = SparseCombinator::new(self.accessor, other.accessor, combinator)?;

        Ok(SparseTensor {
            accessor,
            phantom: self.phantom,
        })
    }

    fn condense<R>(
        self,
        other: SparseTensor<FD, FS, D, T, R>,
        condensor: fn(Number, Number) -> Number,
    ) -> TCResult<Condensed<FD, FS, D, T, A, R>>
    where
        R: SparseAccess<FD, FS, D, T>,
    {
        if self.shape() != other.shape() {
            return Err(TCError::unsupported(format!(
                "cannot condense sparse Tensor of size {} with another of size {}",
                self.shape(),
                other.shape()
            )));
        }

        let accessor = SparseCombinator::new(self.accessor, other.accessor, condensor)?;

        let dense = BlockListSparse::from(accessor);
        Ok(dense.into())
    }

    fn left_combine<R>(
        self,
        other: SparseTensor<FD, FS, D, T, R>,
        combinator: fn(Number, Number) -> Number,
    ) -> TCResult<SparseTensor<FD, FS, D, T, SparseLeftCombinator<FD, FS, D, T, A, R>>>
    where
        R: SparseAccess<FD, FS, D, T>,
    {
        if self.shape() != other.shape() {
            return Err(TCError::unsupported(format!(
                "cannot compare Tensors of different shapes: {}, {}",
                self.shape(),
                other.shape()
            )));
        }

        let accessor = SparseLeftCombinator::new(self.accessor, other.accessor, combinator)?;

        Ok(SparseTensor {
            accessor,
            phantom: self.phantom,
        })
    }
}

impl<FD, FS, D, T> SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType>,
{
    /// Create a new `SparseTensor` with the given schema
    pub async fn create(dir: &D, schema: Schema, txn_id: TxnId) -> TCResult<Self> {
        SparseTable::create(dir, schema, txn_id)
            .map_ok(Self::from)
            .await
    }
}

impl<FD, FS, D, T> TensorPersist for SparseTensor<FD, FS, D, T, SparseAccessor<FD, FS, D, T>> {
    type Persistent = SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>;

    fn as_persistent(self) -> Option<Self::Persistent> {
        match self.accessor {
            SparseAccessor::Table(table) => Some(table.into()),
            _ => None,
        }
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn ndim(&self) -> usize {
        self.accessor.ndim()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }

    fn size(&self) -> u64 {
        self.accessor.size()
    }
}

impl<FD, FS, D, T, A> TensorInstance for SparseTensor<FD, FS, D, T, A> {
    type Dense = DenseTensor<FD, FS, D, T, BlockListSparse<FD, FS, D, T, A>>;
    type Sparse = Self;

    fn into_dense(self) -> Self::Dense {
        BlockListSparse::from(self.into_inner()).into()
    }

    fn into_sparse(self) -> Self::Sparse {
        self
    }
}

impl<FD, FS, D, T, L, R> TensorBoolean<SparseTensor<FD, FS, D, T, R>>
    for SparseTensor<FD, FS, D, T, L>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Combine = SparseTensor<FD, FS, D, T, SparseCombinator<FD, FS, D, T, L, R>>;
    type LeftCombine = SparseTensor<FD, FS, D, T, SparseLeftCombinator<FD, FS, D, T, L, R>>;

    fn and(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::LeftCombine> {
        self.left_combine(other, Number::and)
    }

    fn or(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Combine> {
        self.combine(other, Number::or)
    }

    fn xor(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Combine> {
        self.combine(other, Number::xor)
    }
}

impl<FD, FS, D, T, A> TensorBoolean<Tensor<FD, FS, D, T>> for SparseTensor<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Combine = Tensor<FD, FS, D, T>;
    type LeftCombine = Tensor<FD, FS, D, T>;

    fn and(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(other) => self.and(other.into_sparse()).map(Tensor::from),
            Tensor::Sparse(other) => self.and(other).map(Tensor::from),
        }
    }

    fn or(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(other) => self.into_dense().or(other).map(Tensor::from),
            Tensor::Sparse(other) => self.or(other).map(Tensor::from),
        }
    }

    fn xor(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(other) => self.into_dense().xor(other).map(Tensor::from),
            Tensor::Sparse(other) => self.xor(other).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, A> TensorBooleanConst for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Combine = SparseTensor<FD, FS, D, T, SparseConstCombinator<FD, FS, D, T, A>>;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        let access = SparseConstCombinator::new(self.accessor, other, Number::and);
        Ok(access.into())
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        let access = SparseConstCombinator::new(self.accessor, other, Number::or);
        Ok(access.into())
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        let access = SparseConstCombinator::new(self.accessor, other, Number::xor);
        Ok(access.into())
    }
}

impl<FD, FS, D, T, L, R> TensorCompare<SparseTensor<FD, FS, D, T, R>>
    for SparseTensor<FD, FS, D, T, L>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Compare = SparseTensor<FD, FS, D, T, SparseCombinator<FD, FS, D, T, L, R>>;
    type Dense = Condensed<FD, FS, D, T, L, R>;

    fn eq(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Dense> {
        fn eq(l: Number, r: Number) -> Number {
            (l == r).into()
        }

        self.condense(other, eq)
    }

    fn gt(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Compare> {
        fn gt(l: Number, r: Number) -> Number {
            (l > r).into()
        }

        self.combine(other, gt)
    }

    fn gte(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Dense> {
        fn gte(l: Number, r: Number) -> Number {
            (l >= r).into()
        }

        self.condense(other, gte)
    }

    fn lt(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Compare> {
        fn lt(l: Number, r: Number) -> Number {
            (l < r).into()
        }

        self.combine(other, lt)
    }

    fn lte(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Dense> {
        fn lte(l: Number, r: Number) -> Number {
            (l <= r).into()
        }

        self.condense(other, lte)
    }

    fn ne(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Compare> {
        fn ne(l: Number, r: Number) -> Number {
            (l != r).into()
        }

        self.combine(other, ne)
    }
}

impl<FD, FS, D, T, A> TensorCompare<Tensor<FD, FS, D, T>> for SparseTensor<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Compare = Tensor<FD, FS, D, T>;
    type Dense = Tensor<FD, FS, D, T>;

    fn eq(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(other) => self.eq(other.into_sparse()).map(Tensor::from),
            Tensor::Sparse(other) => self.eq(other).map(Tensor::from),
        }
    }

    fn gt(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(other) => self.into_dense().gt(other).map(Tensor::from),
            Tensor::Sparse(other) => self.gt(other).map(Tensor::from),
        }
    }

    fn gte(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(other) => self.into_dense().gte(other).map(Tensor::from),
            Tensor::Sparse(other) => self.gte(other).map(Tensor::from),
        }
    }

    fn lt(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(other) => self.into_dense().gt(other).map(Tensor::from),
            Tensor::Sparse(other) => self.gt(other).map(Tensor::from),
        }
    }

    fn lte(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(other) => self.into_dense().lte(other).map(Tensor::from),
            Tensor::Sparse(other) => self.lte(other).map(Tensor::from),
        }
    }

    fn ne(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(other) => self.ne(other.into_sparse()).map(Tensor::from),
            Tensor::Sparse(other) => self.ne(other).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, A> TensorCompareConst for SparseTensor<FD, FS, D, T, A> {
    type Compare = SparseTensor<FD, FS, D, T, SparseConstCombinator<FD, FS, D, T, A>>;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        fn eq(l: Number, r: Number) -> Number {
            (l == r).into()
        }

        Ok(SparseConstCombinator::new(self.accessor, other, eq).into())
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        fn gt(l: Number, r: Number) -> Number {
            (l > r).into()
        }

        Ok(SparseConstCombinator::new(self.accessor, other, gt).into())
    }

    fn gte_const(self, other: Number) -> TCResult<Self::Compare> {
        fn gte(l: Number, r: Number) -> Number {
            (l >= r).into()
        }

        Ok(SparseConstCombinator::new(self.accessor, other, gte).into())
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        fn lt(l: Number, r: Number) -> Number {
            (l < r).into()
        }

        Ok(SparseConstCombinator::new(self.accessor, other, lt).into())
    }

    fn lte_const(self, other: Number) -> TCResult<Self::Compare> {
        fn lte(l: Number, r: Number) -> Number {
            (l <= r).into()
        }

        Ok(SparseConstCombinator::new(self.accessor, other, lte).into())
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        fn ne(l: Number, r: Number) -> Number {
            (l != r).into()
        }

        Ok(SparseConstCombinator::new(self.accessor, other, ne).into())
    }
}

#[async_trait]
impl<FD, FS, D, T, A> TensorDiagonal<D> for SparseTensor<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    A: SparseAccess<FD, FS, D, T>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
    SparseTable<FD, FS, D, T>: ReadValueAt<D, Txn = T>,
{
    type Txn = T;
    type Diagonal = SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>;

    async fn diagonal(self, txn: Self::Txn) -> TCResult<Self::Diagonal> {
        if self.ndim() != 2 {
            return Err(TCError::not_implemented(format!(
                "diagonal of a {}-dimensional sparse Tensor",
                self.ndim()
            )));
        }

        let size = self.shape()[0];
        if size != self.shape()[1] {
            return Err(TCError::bad_request(
                "diagonal requires a square matrix but found",
                self.shape(),
            ));
        }

        let txn_id = *txn.id();
        let dir = txn.context().create_dir_unique(txn_id).await?;

        let shape = vec![size].into();
        let dtype = self.dtype();
        let schema = Schema { shape, dtype };
        let table = SparseTable::create(&dir, schema, txn_id).await?;

        let filled = self.accessor.filled(txn).await?;
        filled
            .try_filter_map(|(mut coord, value)| {
                future::ready(Ok({
                    debug_assert!(coord.len() == 2);
                    debug_assert_ne!(value, value.class().zero());

                    if coord.pop() == Some(coord[0]) {
                        Some((coord, value))
                    } else {
                        None
                    }
                }))
            })
            .map_ok(|(coord, value)| table.write_value(txn_id, coord, value))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), ()| future::ready(Ok(())))
            .await?;

        Ok(table.into())
    }
}

#[async_trait]
impl<FD, FS, D, T, L, R> TensorDualIO<D, SparseTensor<FD, FS, D, T, R>>
    for SparseTensor<FD, FS, D, T, L>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    L: SparseWrite<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    async fn write(
        self,
        txn: T,
        bounds: Bounds,
        other: SparseTensor<FD, FS, D, T, R>,
    ) -> TCResult<()> {
        let slice_shape = bounds.to_shape(self.shape())?;
        if &slice_shape != other.shape() {
            return Err(TCError::unsupported(format!(
                "cannot write tensor of shape {} to slice of shape {}",
                other.shape(),
                slice_shape,
            )));
        }

        let txn_id = *txn.id();
        let filled = other.accessor.filled(txn).await?;

        let rebase = transform::Slice::new(self.shape().clone(), bounds)?;
        filled
            .map_ok(move |(coord, value)| {
                let coord = rebase.invert_coord(&coord);
                (coord, value)
            })
            .map_ok(|(coord, value)| self.accessor.write_value(txn_id, coord, value))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T, A> TensorDualIO<D, Tensor<FD, FS, D, T>> for SparseTensor<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseWrite<FD, FS, D, T>,
{
    type Txn = T;

    async fn write(
        self,
        txn: Self::Txn,
        bounds: Bounds,
        other: Tensor<FD, FS, D, T>,
    ) -> TCResult<()> {
        let shape = bounds.to_shape(self.shape())?;
        let other = if other.shape() == &shape {
            other
        } else {
            other.broadcast(shape)?
        };

        match other {
            Tensor::Dense(other) => self.write(txn, bounds, other.into_sparse()).await,
            Tensor::Sparse(other) => self.write(txn, bounds, other).await,
        }
    }
}

#[async_trait]
impl<FD, FS, D, T, A> TensorIO<D> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseWrite<FD, FS, D, T>,
{
    type Txn = T;

    async fn read_value(self, txn: Self::Txn, coord: Coord) -> TCResult<Number> {
        self.accessor
            .read_value_at(txn, coord)
            .map_ok(|(_, value)| value)
            .await
    }

    async fn write_value(&self, txn_id: TxnId, mut bounds: Bounds, value: Number) -> TCResult<()> {
        if self.shape().is_empty() {
            return self.accessor.write_value(txn_id, vec![], value).await;
        }

        bounds.normalize(self.shape());
        debug!("SparseTensor::write_value {} to bounds, {}", value, bounds);
        stream::iter(bounds.affected())
            .map(|coord| self.accessor.write_value(txn_id, coord, value))
            .buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.accessor.write_value(txn_id, coord, value).await
    }
}

impl<FD, FS, D, T, L, R> TensorMath<D, SparseTensor<FD, FS, D, T, R>>
    for SparseTensor<FD, FS, D, T, L>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Combine = SparseTensor<FD, FS, D, T, SparseCombinator<FD, FS, D, T, L, R>>;
    type LeftCombine = SparseTensor<FD, FS, D, T, SparseLeftCombinator<FD, FS, D, T, L, R>>;

    fn add(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Combine> {
        debug!("SparseTensor::add");
        self.combine(other, Number::add)
    }

    fn div(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::LeftCombine> {
        debug!("SparseTensor::div");
        fn div(l: Number, r: Number) -> Number {
            // to prevent a divide-by-zero error, treat the right-hand side as if it doesn't exist
            if r == r.class().zero() {
                Ord::max(l.class(), r.class()).zero()
            } else {
                l / r
            }
        }

        self.left_combine(other, div)
    }

    fn mul(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::LeftCombine> {
        debug!("SparseTensor::mul");
        self.left_combine(other, Number::mul)
    }

    fn pow(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::LeftCombine> {
        if !other.dtype().is_real() {
            return Err(TCError::unsupported(ERR_COMPLEX_EXPONENT));
        }

        debug!("SparseTensor::pow");
        self.left_combine(other, Number::pow)
    }

    fn sub(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::Combine> {
        debug!("SparseTensor::sub");
        self.combine(other, Number::sub)
    }
}

impl<FD, FS, D, T, A> TensorMath<D, Tensor<FD, FS, D, T>> for SparseTensor<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Combine = Tensor<FD, FS, D, T>;
    type LeftCombine = Tensor<FD, FS, D, T>;

    fn add(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Sparse(sparse) => self.add(sparse).map(Tensor::from),
            Tensor::Dense(dense) => self.into_dense().add(dense).map(Tensor::from),
        }
    }

    fn div(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Sparse(sparse) => self.div(sparse).map(Tensor::from),
            Tensor::Dense(dense) => self.div(dense.into_sparse()).map(Tensor::from),
        }
    }

    fn mul(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Sparse(sparse) => self.mul(sparse).map(Tensor::from),
            Tensor::Dense(dense) => self.mul(dense.into_sparse()).map(Tensor::from),
        }
    }

    fn pow(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Sparse(sparse) => self.mul(sparse).map(Tensor::from),
            Tensor::Dense(dense) => self.mul(dense.into_sparse()).map(Tensor::from),
        }
    }

    fn sub(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Sparse(sparse) => self.sub(sparse).map(Tensor::from),
            Tensor::Dense(dense) => self.into_dense().sub(dense).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, A> TensorMathConst for SparseTensor<FD, FS, D, T, A> {
    type Combine = SparseTensor<FD, FS, D, T, SparseConstCombinator<FD, FS, D, T, A>>;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(SparseConstCombinator::new(self.accessor, other, Number::add).into())
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(SparseConstCombinator::new(self.accessor, other, Number::div).into())
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(SparseConstCombinator::new(self.accessor, other, Number::mul).into())
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        if !other.class().is_real() {
            return Err(TCError::unsupported(ERR_COMPLEX_EXPONENT));
        }

        Ok(SparseConstCombinator::new(self.accessor, other, Number::pow).into())
    }

    fn sub_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(SparseConstCombinator::new(self.accessor, other, Number::sub).into())
    }
}

impl<FD, FS, D, T, A> TensorReduce<D> for SparseTensor<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
    Self: TensorInstance,
    <Self as TensorInstance>::Dense: TensorReduce<D, Txn = T> + Send + Sync,
{
    type Txn = T;
    type Reduce = SparseTensor<FD, FS, D, T, SparseReduce<FD, FS, D, T>>;

    fn product(self, axis: usize) -> TCResult<Self::Reduce> {
        let accessor = SparseReduce::new(
            self.accessor.accessor(),
            axis,
            SparseTensor::<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>::product_all,
        )?;

        Ok(SparseTensor::from(accessor))
    }

    fn product_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        Box::pin(async move { self.clone().into_dense().product_all(txn).await })
    }

    fn sum(self, axis: usize) -> TCResult<Self::Reduce> {
        let accessor = SparseReduce::new(
            self.accessor.accessor(),
            axis,
            SparseTensor::<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>::sum_all,
        )?;

        Ok(SparseTensor::from(accessor))
    }

    fn sum_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let mut sum = self.dtype().zero();
            let mut filled = self.accessor.clone().filled(txn).await?;
            let mut buffer = Vec::with_capacity(PER_BLOCK);
            while let Some((_coord, value)) = filled.try_next().await? {
                buffer.push(value);

                if buffer.len() == PER_BLOCK {
                    sum += Array::from(buffer.to_vec()).sum();
                    buffer.clear()
                }
            }

            if !buffer.is_empty() {
                sum += Array::from(buffer).sum();
            }

            Ok(sum)
        })
    }
}

impl<FD, FS, D, T, A> TensorTransform for SparseTensor<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Broadcast = SparseTensor<FD, FS, D, T, SparseBroadcast<FD, FS, D, T, A>>;
    type Cast = SparseTensor<FD, FS, D, T, SparseCast<FD, FS, D, T, A>>;
    type Expand = SparseTensor<FD, FS, D, T, SparseExpand<FD, FS, D, T, A>>;
    type Flip = SparseTensor<FD, FS, D, T, SparseFlip<FD, FS, D, T, A>>;
    type Reshape = SparseTensor<FD, FS, D, T, SparseReshape<FD, FS, D, T, A>>;
    type Slice = SparseTensor<FD, FS, D, T, A::Slice>;
    type Transpose = SparseTensor<FD, FS, D, T, A::Transpose>;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        let accessor = SparseBroadcast::new(self.accessor, shape)?;
        Ok(accessor.into())
    }

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        let accessor = SparseCast::new(self.accessor, dtype);
        Ok(accessor.into())
    }

    fn expand_dims(self, axis: usize) -> TCResult<Self::Expand> {
        let accessor = SparseExpand::new(self.accessor, axis)?;
        Ok(accessor.into())
    }

    fn flip(self, axis: usize) -> TCResult<Self::Flip> {
        let accessor = SparseFlip::new(self.accessor, axis)?;
        Ok(accessor.into())
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        let accessor = SparseReshape::new(self.accessor, shape)?;
        Ok(accessor.into())
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let accessor = self.accessor.slice(bounds)?;
        Ok(accessor.into())
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let accessor = self.accessor.transpose(permutation)?;
        Ok(accessor.into())
    }
}

#[async_trait]
impl<FD, FS, D, T, A> TensorUnary<D> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;
    type Unary = SparseTensor<FD, FS, D, T, SparseUnary<FD, FS, D, T>>;

    fn abs(&self) -> TCResult<Self::Unary> {
        let source = self.accessor.clone().accessor();
        let transform = <Number as NumberInstance>::abs;

        let accessor = SparseUnary::new(source, transform, self.dtype());
        Ok(SparseTensor::from(accessor))
    }

    fn exp(&self) -> TCResult<Self::Unary> {
        fn exp(n: Number) -> Number {
            let n = f64::cast_from(n);
            n.exp().into()
        }

        let dtype = NumberType::Float(FloatType::F64);
        let source = self.accessor.clone().accessor();
        let accessor = SparseUnary::new(source, exp, dtype);
        Ok(SparseTensor::from(accessor))
    }

    async fn all(self, txn: Self::Txn) -> TCResult<bool> {
        let affected = stream::iter(Bounds::all(self.shape()).affected());
        let filled = self.accessor.filled(txn).await?;

        let mut coords = filled
            .map_ok(|(coord, _)| coord)
            .zip(affected)
            .map(|(r, expected)| r.map(|actual| (actual, expected)));

        while let Some((actual, expected)) = coords.try_next().await? {
            if actual != expected {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self, txn: Self::Txn) -> TCResult<bool> {
        let mut filled = self.accessor.filled(txn).await?;
        Ok(filled.next().await.is_some())
    }

    fn not(&self) -> TCResult<Self::Unary> {
        Err(TCError::unsupported(ERR_NOT_SPARSE))
    }
}

#[async_trait]
impl<FD, FS, D, T, A> CopyFrom<D, SparseTensor<FD, FS, D, T, A>>
    for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    async fn copy_from(
        instance: SparseTensor<FD, FS, D, T, A>,
        store: Self::Store,
        txn: &Self::Txn,
    ) -> TCResult<Self> {
        SparseTable::copy_from(instance, store, txn)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<'en, FD, FS, D, T, A> Hash<'en, D> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Item = SparseRow;
    type Txn = T;

    async fn hashable(&'en self, txn: &'en Self::Txn) -> TCResult<TCBoxTryStream<'en, SparseRow>> {
        self.accessor.clone().filled(txn.clone()).await
    }
}

#[async_trait]
impl<FD, FS, D, T> Persist<D> for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Schema = Schema;
    type Store = D;
    type Txn = T;

    fn schema(&self) -> &Self::Schema {
        self.accessor.schema()
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        SparseTable::load(txn, schema, store)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T> Restore<D> for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        self.accessor.restore(&backup.accessor, txn_id).await
    }
}

#[async_trait]
impl<FD, FS, D, T> Transact for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    Self: Send + Sync,
    SparseTable<FD, FS, D, T>: Transact + Send + Sync,
{
    async fn commit(&self, txn_id: &TxnId) {
        self.accessor.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.accessor.finalize(txn_id).await
    }
}

impl<FD, FS, D, T, A> From<A> for SparseTensor<FD, FS, D, T, A> {
    fn from(accessor: A) -> Self {
        Self {
            accessor,
            phantom: Phantom::default(),
        }
    }
}

impl<FD, FS, D, T, A> fmt::Display for SparseTensor<FD, FS, D, T, A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a sparse Tensor")
    }
}

#[async_trait]
impl<'en, FD, FS, D, T, A> IntoView<'en, D> for SparseTensor<FD, FS, D, T, A>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;
    type View = SparseTensorView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let shape = self.shape().clone();
        let dtype = self.dtype();

        Ok(SparseTensorView {
            schema: Schema { shape, dtype },
            filled: self.accessor.filled(txn).await?,
        })
    }
}

#[async_trait]
impl<FD, FS, D, T> de::FromStream for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Context = T;

    async fn from_stream<De: de::Decoder>(txn: T, decoder: &mut De) -> Result<Self, De::Error> {
        decoder.decode_seq(SparseTensorVisitor::new(txn)).await
    }
}

struct SparseTensorVisitor<FD, FS, D, T> {
    txn: T,
    dense: PhantomData<FD>,
    sparse: PhantomData<FS>,
    dir: PhantomData<D>,
}

impl<FD, FS, D, T> SparseTensorVisitor<FD, FS, D, T> {
    fn new(txn: T) -> Self {
        Self {
            txn,
            dense: PhantomData,
            sparse: PhantomData,
            dir: PhantomData,
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> de::Visitor for SparseTensorVisitor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Value = SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>;

    fn expecting() -> &'static str {
        "a SparseTensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let schema = seq.next_element(()).await?;
        let schema = schema.ok_or_else(|| de::Error::invalid_length(0, "tensor schema"))?;

        let txn_id = *self.txn.id();
        let table = SparseTable::create(self.txn.context(), schema, txn_id)
            .map_err(de::Error::custom)
            .await?;

        if let Some(table) = seq
            .next_element::<SparseTable<FD, FS, D, T>>((table.clone(), txn_id))
            .await?
        {
            Ok(SparseTensor::from(table))
        } else {
            Ok(SparseTensor::from(table))
        }
    }
}

pub struct SparseTensorView<'en> {
    schema: Schema,
    filled: SparseStream<'en>,
}

impl<'en> en::IntoStream<'en> for SparseTensorView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let filled = en::SeqStream::from(self.filled);
        (self.schema, filled).into_stream(encoder)
    }
}
