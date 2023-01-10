use std::convert::TryFrom;
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
use safecast::{CastFrom, CastInto};

use tc_error::*;
use tc_table::{Node, NodeId};
use tc_transact::fs::{CopyFrom, Dir, DirCreateFile, DirReadFile, File, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::{
    ComplexType, Float, FloatType, Number, NumberClass, NumberInstance, NumberType, Trigonometry,
    UIntType,
};
use tcgeneric::{Instance, TCBoxTryFuture};

use super::dense::{BlockListSparse, DenseTensor, PER_BLOCK};
use super::stream::ReadValueAt;
use super::transform;
use super::{
    coord_bounds, tile, trig_dtype, AxisBounds, Bounds, Coord, Phantom, Schema, Shape, Tensor,
    TensorAccess, TensorBoolean, TensorBooleanConst, TensorCompare, TensorCompareConst,
    TensorDiagonal, TensorDualIO, TensorIO, TensorIndex, TensorInstance, TensorMath,
    TensorMathConst, TensorPersist, TensorReduce, TensorTransform, TensorTrig, TensorType,
    TensorUnary, ERR_COMPLEX_EXPONENT,
};

use access::*;
pub use access::{DenseToSparse, SparseAccess, SparseAccessor, SparseWrite};
use combine::coord_to_offset;
pub use table::SparseTable;

mod access;
mod combine;
mod table;

pub type SparseRow = (Coord, Number);
pub type SparseStream<'a> = Pin<Box<dyn Stream<Item = TCResult<SparseRow>> + Send + Unpin + 'a>>;

const ERR_NOT_SPARSE: &str = "The result of the requested operation would not be sparse; \
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    /// Access the schema of this [`SparseTensor`]
    pub fn schema(&self) -> Schema {
        Schema::from((self.shape().clone(), self.dtype()))
    }

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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FD> + DirCreateFile<FS>,
    D::Store: From<D> + From<FS>,
{
    /// Tile the given `tensor` into a new `SparseTensor`
    pub async fn tile(
        txn: T,
        tensor: SparseTensor<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>,
        multiples: Vec<u64>,
    ) -> TCResult<Self> {
        if multiples.len() != tensor.ndim() {
            return Err(TCError::bad_request(
                "wrong number of multiples to tile a Tensor with shape",
                tensor.shape(),
            ))?;
        }

        let txn_id = *txn.id();
        let dir = txn.context().create_dir_unique(txn_id).await?;
        let dtype = tensor.dtype();
        let shape = tensor
            .shape()
            .iter()
            .zip(&multiples)
            .map(|(dim, m)| dim * m)
            .collect();

        let input = match tensor.accessor {
            SparseAccessor::Table(table) => table.into(),
            other => {
                let dir = txn.context().create_dir_unique(*txn.id()).await?;
                SparseTensor::copy_from(&txn, dir.into(), other.into()).await?
            }
        };

        let output = Self::create(txn_id, Schema { shape, dtype }, dir.into())?;
        tile(txn, input, output, multiples).await
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

    fn is_persistent(&self) -> bool {
        match &self.accessor {
            SparseAccessor::Table(_) => true,
            _ => false,
        }
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseTensor<FD, FS, D, T, A>
where
    A: TensorAccess,
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    Self: TensorInstance,
    <Self as TensorInstance>::Dense: TensorBooleanConst,
{
    type Combine = SparseTensor<FD, FS, D, T, SparseConstCombinator<FD, FS, D, T, A>>;
    type DenseCombine = <<Self as TensorInstance>::Dense as TensorBooleanConst>::DenseCombine;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        let access = SparseConstCombinator::new(self.accessor, other, Number::and);
        Ok(access.into())
    }

    fn or_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        self.into_dense().or_const(other)
    }

    fn xor_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        self.into_dense().xor_const(other)
    }
}

impl<FD, FS, D, T, L, R> TensorCompare<SparseTensor<FD, FS, D, T, R>>
    for SparseTensor<FD, FS, D, T, L>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS>,
    D::Store: From<D> + From<FS>,
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
        let table = SparseTable::create(txn_id, schema, dir.into())?;

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
impl<FD, FS, D, T, L> TensorDualIO<D, SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>>
    for SparseTensor<FD, FS, D, T, L>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    L: TensorAccess + SparseWrite,
    D::Write: DirCreateFile<FD>,
{
    type Txn = T;

    async fn write(
        self,
        txn: T,
        bounds: Bounds,
        other: SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>,
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

        let rebase = transform::Slice::new(self.accessor.shape().clone(), bounds)?;
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T> + SparseWrite,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
    D::Store: From<D> + From<FS>,
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
            Tensor::Dense(other) => {
                let dir = txn.context().create_dir_unique(*txn.id()).await?;
                let other = SparseTensor::copy_from(&txn, dir.into(), other.into_sparse()).await?;
                self.write(txn, bounds, other.into_sparse()).await
            }
            Tensor::Sparse(other) => match other.accessor {
                SparseAccessor::Table(table) => {
                    self.write(txn, bounds, SparseTensor::from(table)).await
                }
                other => {
                    let dir = txn.context().create_dir_unique(*txn.id()).await?;
                    let other = SparseTensor::copy_from(&txn, dir.into(), other.into()).await?;
                    self.write(txn, bounds, other).await
                }
            },
        }
    }
}

#[async_trait]
impl<FD, FS, D, T, A> TensorIndex<D> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T> + SparseWrite,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS>,
    D::Store: From<D> + From<FS>,
{
    type Txn = T;
    type Index = SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>;

    async fn argmax(self, txn: Self::Txn, axis: usize) -> TCResult<Self::Index> {
        if axis >= self.ndim() {
            return Err(TCError::unsupported(format!(
                "invalid argmax axis for tensor with {} dimensions: {}",
                self.ndim(),
                axis
            )));
        }

        let shape = {
            let mut shape = self.shape().clone();
            shape.remove(axis);
            shape
        };

        let schema = Schema {
            shape,
            dtype: NumberType::UInt(UIntType::U64),
        };

        let txn_id = *txn.id();
        let dir = txn.context().create_dir_unique(txn_id).await?;

        let table = SparseTable::create(txn_id, schema, dir.into())?;

        let dim = self.shape()[axis];
        let zero = self.dtype().zero();
        let axes = (0..self.ndim())
            .into_iter()
            .filter(|x| x != &axis)
            .collect();

        let mut filled = self.accessor.clone().filled_at(txn.clone(), axes).await?;
        while let Some(coords) = filled.try_next().await? {
            for coord in coords.to_vec() {
                let mut bounds: Bounds = coord.iter().cloned().map(AxisBounds::At).collect();
                bounds.insert(axis, AxisBounds::all(dim));

                let slice = self.accessor.clone().slice(bounds)?;
                debug_assert_eq!(slice.ndim(), 1);

                let filled = slice.filled(txn.clone()).await?;
                let filled = filled.map_ok(|(offset, value)| (offset[0], value));
                let imax = imax(filled, zero, dim).await?;
                table.write_value(txn_id, coord, imax.0.into()).await?;
            }
        }

        Ok(table.into())
    }

    async fn argmax_all(self, txn: Self::Txn) -> TCResult<u64> {
        let zero = self.dtype().zero();
        let size = self.size();
        let coord_bounds = coord_bounds(self.shape());
        let filled = self.accessor.filled(txn).await?;
        let filled =
            filled.map_ok(move |(coord, value)| (coord_to_offset(&coord, &coord_bounds), value));
        let imax = imax(filled, zero, size).await?;
        Ok(imax.0)
    }
}

async fn imax<F>(mut filled: F, zero: Number, size: u64) -> TCResult<(u64, Number)>
where
    F: Stream<Item = TCResult<(u64, Number)>> + Unpin,
{
    let mut first_empty = Some(0);
    let mut last = 0u64;
    let mut imax = None;
    while let Some((offset, value)) = filled.try_next().await? {
        if offset == 0 {
            first_empty = None;
        } else if first_empty.is_none() {
            if offset > (last + 1) {
                first_empty = Some(last + 1)
            }
        }

        if let Some((ref mut i, ref mut max)) = &mut imax {
            if value > *max {
                *i = offset;
                *max = value;
            }
        } else {
            imax = Some((offset, value));
        }

        last = offset;
    }

    if first_empty.is_none() && last < (size - 1) {
        if last == 0 && imax.is_none() {
            first_empty = Some(0);
        } else {
            first_empty = Some(last + 1);
        }
    }

    if let Some((i, max)) = imax {
        if max > zero {
            Ok((i, max))
        } else if let Some(first_empty) = first_empty {
            Ok((first_empty, zero))
        } else {
            Ok((i, max))
        }
    } else {
        Ok((0, zero))
    }
}

#[async_trait]
impl<FD, FS, D, T, A> TensorIO<D> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: TensorAccess + SparseWrite + ReadValueAt<D, Txn = T>,
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

        bounds.normalize(self.accessor.shape());
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

impl<FD, FS, D, T, L, R> TensorMath<SparseTensor<FD, FS, D, T, R>> for SparseTensor<FD, FS, D, T, L>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner>,
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

    fn log(self, base: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::LeftCombine> {
        if base.dtype().is_complex() {
            return Err(TCError::unsupported(ERR_COMPLEX_EXPONENT));
        }

        fn log(n: Number, base: Number) -> Number {
            n.log(Float::cast_from(base))
        }

        self.left_combine(base, log)
    }

    fn mul(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::LeftCombine> {
        debug!("SparseTensor::mul");
        self.left_combine(other, Number::mul)
    }

    fn pow(self, other: SparseTensor<FD, FS, D, T, R>) -> TCResult<Self::LeftCombine> {
        if other.dtype().is_complex() {
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

impl<FD, FS, D, T, A> TensorMath<Tensor<FD, FS, D, T>> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
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

    fn log(self, base: Tensor<FD, FS, D, T>) -> TCResult<Self::LeftCombine> {
        match base {
            Tensor::Sparse(sparse) => self.log(sparse).map(Tensor::from),
            Tensor::Dense(dense) => self.log(dense.into_sparse()).map(Tensor::from),
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

impl<FD, FS, D, T, A> TensorMathConst for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    Self: TensorInstance,
    <Self as TensorInstance>::Dense: TensorMathConst,
{
    type Combine = SparseTensor<FD, FS, D, T, SparseConstCombinator<FD, FS, D, T, A>>;
    type DenseCombine = <<Self as TensorInstance>::Dense as TensorMathConst>::DenseCombine;

    fn add_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        self.into_dense().add_const(other)
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        Ok(SparseConstCombinator::new(self.accessor, other, Number::div).into())
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        if base.class().is_complex() {
            return Err(TCError::unsupported(ERR_COMPLEX_EXPONENT));
        }

        fn log(n: Number, base: Number) -> Number {
            if let Number::Float(base) = base {
                n.log(base)
            } else {
                unreachable!("log with non-floating point base")
            }
        }

        let base = Number::Float(base.cast_into());
        Ok(SparseConstCombinator::new(self.accessor, base, log).into())
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

    fn sub_const(self, other: Number) -> TCResult<Self::DenseCombine> {
        self.into_dense().sub_const(other)
    }
}

impl<FD, FS, D, T, A> TensorReduce<D> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
    Self: TensorInstance,
    <Self as TensorInstance>::Dense: TensorReduce<D, Txn = T> + Send + Sync,
{
    type Txn = T;
    type Reduce = SparseTensor<FD, FS, D, T, SparseReduce<FD, FS, D, T>>;

    fn max(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce> {
        let accessor = SparseReduce::new(
            self.accessor.accessor(),
            axis,
            keepdims,
            SparseTensor::<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>::max_all,
        )?;

        Ok(SparseTensor::from(accessor))
    }

    fn max_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move { self.clone().into_dense().max_all(txn).await })
    }

    fn min(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce> {
        let accessor = SparseReduce::new(
            self.accessor.accessor(),
            axis,
            keepdims,
            SparseTensor::<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>::min_all,
        )?;

        Ok(SparseTensor::from(accessor))
    }

    fn min_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move { self.clone().into_dense().min_all(txn).await })
    }

    fn product(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce> {
        let accessor = SparseReduce::new(
            self.accessor.accessor(),
            axis,
            keepdims,
            SparseTensor::<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>::product_all,
        )?;

        Ok(SparseTensor::from(accessor))
    }

    fn product_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        Box::pin(async move { self.clone().into_dense().product_all(txn).await })
    }

    fn sum(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce> {
        let accessor = SparseReduce::new(
            self.accessor.accessor(),
            axis,
            keepdims,
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
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FD>,
{
    type Broadcast = SparseTensor<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>;
    type Cast = SparseTensor<FD, FS, D, T, SparseCast<FD, FS, D, T, A>>;
    type Expand = SparseTensor<FD, FS, D, T, SparseExpand<FD, FS, D, T, A>>;
    type Flip = SparseTensor<FD, FS, D, T, SparseFlip<FD, FS, D, T, A>>;
    type Reshape = SparseTensor<FD, FS, D, T, SparseReshape<FD, FS, D, T, A>>;
    type Slice = SparseTensor<FD, FS, D, T, A::Slice>;
    type Transpose = SparseTensor<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        if self.shape() == &shape {
            return Ok(self.into_inner().accessor().into());
        }

        let accessor = SparseBroadcast::new(self.accessor, shape)?;
        Ok(accessor.accessor().into())
    }

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        if self.dtype().is_complex() && dtype.is_real() {
            return Err(TCError::unsupported("cannot cast a complex Tensor into a real Tensor; consider the real, imag, or abs methods instead"));
        }

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
        debug!("SparseTensor::transpose {:?}", permutation);
        let accessor = self.accessor.transpose(permutation)?;
        Ok(accessor.into())
    }
}

macro_rules! trig {
    ($fun:ident) => {
        fn $fun(&self) -> TCResult<Self::Unary> {
            let dtype = trig_dtype(self.dtype());
            let source = self.accessor.clone().accessor();
            let accessor = SparseUnary::new(source, Number::$fun, dtype);
            Ok(SparseTensor::from(accessor))
        }
    };
}

#[async_trait]
impl<FD, FS, D, T, A> TensorTrig for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Unary = SparseTensor<FD, FS, D, T, SparseUnary<FD, FS, D, T>>;

    trig! {asin}
    trig! {sin}
    trig! {sinh}
    trig! {asinh}

    trig! {acos}
    trig! {cos}
    trig! {cosh}
    trig! {acosh}

    trig! {atan}
    trig! {tan}
    trig! {tanh}
    trig! {atanh}
}

#[async_trait]
impl<FD, FS, D, T, A> TensorUnary<D> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;
    type Unary = SparseTensor<FD, FS, D, T, SparseUnary<FD, FS, D, T>>;

    fn abs(&self) -> TCResult<Self::Unary> {
        let source = self.accessor.clone().accessor();
        let transform = <Number as NumberInstance>::abs;

        let accessor = SparseUnary::new(source, transform, self.dtype().one().abs().class());
        Ok(SparseTensor::from(accessor))
    }

    fn exp(&self) -> TCResult<Self::Unary> {
        fn exp(n: Number) -> Number {
            match n {
                Number::Complex(n) => n.exp().into(),
                Number::Float(n) => n.exp().into(),
                n => f64::cast_from(n).exp().into(),
            }
        }

        let dtype = if self.dtype().is_complex() {
            NumberType::Complex(ComplexType::C64)
        } else {
            NumberType::Float(FloatType::F64)
        };

        let source = self.accessor.clone().accessor();
        let accessor = SparseUnary::new(source, exp, dtype);
        Ok(SparseTensor::from(accessor))
    }

    fn ln(&self) -> TCResult<Self::Unary> {
        let dtype = self.dtype().one().ln().class();
        let source = self.accessor.clone().accessor();
        let accessor = SparseUnary::new(source, Number::ln, dtype);
        Ok(SparseTensor::from(accessor))
    }

    fn round(&self) -> TCResult<Self::Unary> {
        let dtype = self.dtype().one().ln().class();
        let source = self.accessor.clone().accessor();
        let accessor = SparseUnary::new(source, Number::round, dtype);
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

impl<FD, FS, D, T> Persist<D> for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS>,
    D::Store: From<FS>,
{
    type Txn = T;
    type Schema = Schema;

    fn create(txn_id: TxnId, schema: Self::Schema, store: D::Store) -> TCResult<Self> {
        SparseTable::create(txn_id, schema, store).map(Self::from)
    }

    fn load(txn_id: TxnId, schema: Self::Schema, store: D::Store) -> TCResult<Self> {
        SparseTable::load(txn_id, schema, store).map(Self::from)
    }

    fn dir(&self) -> D::Inner {
        self.accessor.dir()
    }
}

#[async_trait]
impl<FD, FS, D, T, A> CopyFrom<D, SparseTensor<FD, FS, D, T, A>>
    for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS>,
    D::Store: From<FS>,
{
    async fn copy_from(
        txn: &Self::Txn,
        store: D::Store,
        instance: SparseTensor<FD, FS, D, T, A>,
    ) -> TCResult<Self> {
        SparseTable::copy_from(txn, store, instance)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T> Restore<D> for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS>,
    D::Store: From<FS>,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        self.accessor.restore(txn_id, &backup.accessor).await
    }
}

#[async_trait]
impl<FD, FS, D, T> Transact for SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>
where
    Self: Send + Sync,
    SparseTable<FD, FS, D, T>: Transact + Send + Sync,
{
    type Commit = <SparseTable<FD, FS, D, T> as Transact>::Commit;

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        self.accessor.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.accessor.rollback(txn_id).await
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

impl<FD, FS, D, T, A> fmt::Display for SparseTensor<FD, FS, D, T, A>
where
    A: TensorAccess,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a Sparse tensor with dtype {} and shape {}",
            self.dtype(),
            self.shape()
        )
    }
}

#[async_trait]
impl<'en, FD, FS, D, T, A> IntoView<'en, D> for SparseTensor<FD, FS, D, T, A>
where
    FD: File<Key = u64, Block = Array, Inner = D::Inner>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS>,
    D::Store: From<D> + From<FS>,
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
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node, Inner = D::Inner> + TryFrom<D::Store, Error = TCError>,
    D: Dir + TryFrom<D::Store, Error = TCError>,
    T: Transaction<D>,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS>,
    D::Store: From<D> + From<FS>,
{
    type Value = SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>;

    fn expecting() -> &'static str {
        "a SparseTensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let schema = seq.next_element::<Schema>(()).await?;
        let schema = schema.ok_or_else(|| de::Error::invalid_length(0, "tensor schema"))?;
        schema.validate("load Sparse").map_err(de::Error::custom)?;

        let txn_id = *self.txn.id();
        let table = SparseTable::create(txn_id, schema, self.txn.context().clone().into())
            .map_err(de::Error::custom)?;

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
