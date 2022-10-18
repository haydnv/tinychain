use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

use afarray::{Array, ArrayExt, ArrayInstance, Complex as _Complex, CoordBlocks};
use arrayfire as af;
use async_trait::async_trait;
use collate::Collate;
use destream::{de, en};
use futures::future::{self, TryFutureExt};
use futures::stream::{Stream, StreamExt, TryStreamExt};
use log::{debug, warn};
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_table::{Node, NodeId};
use tc_transact::fs::{CopyFrom, Dir, DirCreateFile, DirReadFile, File, Persist, Restore};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tc_value::{
    ComplexType, Float, FloatType, Number, NumberClass, NumberCollator, NumberInstance, NumberType,
    Trigonometry, UIntType,
};
use tcgeneric::{Instance, TCBoxTryFuture, TCBoxTryStream};

use super::sparse::{DenseToSparse, SparseTensor};
use super::stream::{Read, ReadValueAt};
use super::{
    tile, trig_dtype, Bounds, Coord, Phantom, Schema, Shape, Tensor, TensorAccess, TensorBoolean,
    TensorBooleanConst, TensorCompare, TensorCompareConst, TensorDiagonal, TensorDualIO, TensorIO,
    TensorIndex, TensorInstance, TensorMath, TensorMathConst, TensorPersist, TensorReduce,
    TensorTransform, TensorTrig, TensorType, TensorUnary, ERR_COMPLEX_EXPONENT,
};

use access::*;
pub use access::{BlockListSparse, DenseAccess, DenseAccessor, DenseWrite};
pub use file::BlockListFile;

mod access;
mod file;
mod stream;

/// The number of bytes in one mebibyte.
const MEBIBYTE: usize = 1_048_576;

/// The number of elements per dense tensor block, equal to (1 mebibyte / 64 bits).
pub const PER_BLOCK: usize = 131_072;

/// A `Tensor` stored as a [`File`] of dense [`Array`] blocks
#[derive(Clone)]
pub struct DenseTensor<FD, FS, D, T, B> {
    blocks: B,

    #[allow(dead_code)]
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, B> DenseTensor<FD, FS, D, T, B> {
    /// Consume this `DenseTensor` handle and return its underlying [`DenseAccessor`]
    pub fn into_inner(self) -> B {
        self.blocks
    }
}

impl<FD, FS, D, T, B> DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    /// Access the schema of this [`DenseTensor`]
    pub fn schema(&self) -> Schema {
        Schema::from((self.shape().clone(), self.dtype()))
    }

    fn combine<OT: DenseAccess<FD, FS, D, T>>(
        self,
        other: DenseTensor<FD, FS, D, T, OT>,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, OT>>> {
        if self.shape() != other.shape() {
            return Err(TCError::unsupported(format!(
                "cannot combine tensors with different shapes: {}, {}",
                self.shape(),
                other.shape()
            )));
        }

        let blocks = BlockListCombine::new(
            self.blocks,
            other.blocks,
            combinator,
            value_combinator,
            dtype,
        )?;

        Ok(DenseTensor::from(blocks))
    }
}

impl<FD, FS, D, T, B> Instance for DenseTensor<FD, FS, D, T, B>
where
    Self: Send + Sync,
{
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        TensorType::Dense
    }
}

impl<FD, FS, D, T> DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    /// Create a new `DenseTensor` filled with the given `value`.
    pub async fn constant<S>(file: FD, txn_id: TxnId, shape: S, value: Number) -> TCResult<Self>
    where
        Shape: From<S>,
    {
        let schema = Schema {
            shape: shape.into(),
            dtype: value.class(),
        };

        schema.validate("create Dense constant")?;

        BlockListFile::constant(file, txn_id, schema.shape, value)
            .map_ok(Self::from)
            .await
    }

    /// Create a new `DenseTensor` filled with a range evenly distributed between `start` and `stop`.
    pub async fn range<S>(
        file: FD,
        txn_id: TxnId,
        shape: S,
        start: Number,
        stop: Number,
    ) -> TCResult<Self>
    where
        Shape: From<S>,
    {
        let schema = Schema {
            shape: shape.into(),
            dtype: Ord::max(start.class(), stop.class()),
        };

        schema.validate("create Dense range")?;

        BlockListFile::range(file, txn_id, schema.shape, start, stop)
            .map_ok(Self::from)
            .await
    }

    pub async fn tile(
        txn: T,
        tensor: DenseTensor<FD, FS, D, T, DenseAccessor<FD, FS, D, T>>,
        multiples: Vec<u64>,
    ) -> TCResult<Self> {
        if multiples.len() != tensor.ndim() {
            return Err(TCError::bad_request(
                "wrong number of multiples to tile a Tensor with shape",
                tensor.shape(),
            ));
        }

        let txn_id = *txn.id();
        let output_file = txn.context().create_file_unique(txn_id).await?;

        let shape: Shape = tensor
            .shape()
            .iter()
            .zip(&multiples)
            .map(|(dim, m)| dim * m)
            .collect();

        let dtype = tensor.dtype();
        let input = match tensor.blocks {
            DenseAccessor::File(file) => DenseTensor::from(file),
            other => {
                let input_file = txn.context().create_file_unique(txn_id).await?;
                DenseTensor::copy_from(DenseTensor::from(other), input_file, &txn).await?
            }
        };

        let output = Self::constant(output_file, txn_id, shape, dtype.zero()).await?;

        tile(txn, input, output, multiples).await
    }
}

impl<FD, FS, D, T> TensorPersist for DenseTensor<FD, FS, D, T, DenseAccessor<FD, FS, D, T>> {
    type Persistent = DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>;

    fn as_persistent(self) -> Option<Self::Persistent> {
        match self.into_inner() {
            DenseAccessor::File(file) => Some(file.into()),
            _ => None,
        }
    }

    fn is_persistent(&self) -> bool {
        match &self.blocks {
            DenseAccessor::File(_) => true,
            _ => false,
        }
    }
}

impl<FD, FS, D, T, B> TensorAccess for DenseTensor<FD, FS, D, T, B>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        self.blocks.dtype()
    }

    fn ndim(&self) -> usize {
        self.blocks.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.blocks.shape()
    }

    fn size(&self) -> u64 {
        self.blocks.size()
    }
}

impl<FD, FS, D, T, B> TensorInstance for DenseTensor<FD, FS, D, T, B> {
    type Dense = Self;
    type Sparse = SparseTensor<FD, FS, D, T, DenseToSparse<FD, FS, D, T, B>>;

    fn into_dense(self) -> Self::Dense {
        self
    }

    fn into_sparse(self) -> Self::Sparse {
        DenseToSparse::from(self.into_inner()).into()
    }
}

impl<FD, FS, D, T, B, O> TensorBoolean<DenseTensor<FD, FS, D, T, O>>
    for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    O: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Combine = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;
    type LeftCombine = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;

    fn and(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::and, Number::and, NumberType::Bool)
    }

    fn or(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::or, Number::or, NumberType::Bool)
    }

    fn xor(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        self.combine(other, Array::xor, Number::xor, NumberType::Bool)
    }
}

impl<FD, FS, D, T, B> TensorBoolean<Tensor<FD, FS, D, T>> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Combine = Tensor<FD, FS, D, T>;
    type LeftCombine = Tensor<FD, FS, D, T>;

    fn and(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.and(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.into_sparse().and(sparse).map(Tensor::from),
        }
    }

    fn or(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.or(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.or(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn xor(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.xor(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.and(sparse.into_dense()).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, B> TensorBooleanConst for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
{
    type Combine = DenseTensor<FD, FS, D, T, BlockListConst<FD, FS, D, T, B>>;
    type DenseCombine = DenseTensor<FD, FS, D, T, BlockListConst<FD, FS, D, T, B>>;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        fn array_and(l: Array, r: Number) -> Array {
            l.and_const(r)
        }

        let blocks = BlockListConst::new(self.blocks, other, array_and, Number::and);
        Ok(blocks.into())
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        fn array_or(l: Array, r: Number) -> Array {
            l.or_const(r)
        }

        let blocks = BlockListConst::new(self.blocks, other, array_or, Number::or);
        Ok(blocks.into())
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        fn array_xor(l: Array, r: Number) -> Array {
            l.xor_const(r)
        }

        let blocks = BlockListConst::new(self.blocks, other, array_xor, Number::xor);
        Ok(blocks.into())
    }
}

impl<FD, FS, D, T, B, O> TensorCompare<DenseTensor<FD, FS, D, T, O>>
    for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    O: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Compare = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;
    type Dense = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;

    fn eq(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Dense> {
        fn eq(l: Number, r: Number) -> Number {
            Number::from(l == r)
        }

        self.combine(other, Array::eq, eq, NumberType::Bool)
    }

    fn gt(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Compare> {
        fn gt(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::gt, gt, NumberType::Bool)
    }

    fn gte(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Dense> {
        fn gte(l: Number, r: Number) -> Number {
            Number::from(l >= r)
        }

        self.combine(other, Array::gte, gte, NumberType::Bool)
    }

    fn lt(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Compare> {
        fn lt(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::lt, lt, NumberType::Bool)
    }

    fn lte(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Dense> {
        fn lte(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::lte, lte, NumberType::Bool)
    }

    fn ne(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Compare> {
        fn ne(l: Number, r: Number) -> Number {
            Number::from(l > r)
        }

        self.combine(other, Array::ne, ne, NumberType::Bool)
    }
}

impl<FD, FS, D, T, B> TensorCompare<Tensor<FD, FS, D, T>> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Compare = Tensor<FD, FS, D, T>;
    type Dense = Tensor<FD, FS, D, T>;

    fn eq(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(dense) => self.eq(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.into_sparse().eq(sparse).map(Tensor::from),
        }
    }

    fn gt(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(dense) => self.gt(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.gt(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn gte(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(dense) => self.gte(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.gte(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn lt(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(dense) => self.lt(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.lt(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn lte(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Dense> {
        match other {
            Tensor::Dense(dense) => self.lte(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.lte(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn ne(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Compare> {
        match other {
            Tensor::Dense(dense) => self.ne(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.ne(sparse.into_dense()).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, B> TensorCompareConst for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
{
    type Compare = DenseTensor<FD, FS, D, T, BlockListConst<FD, FS, D, T, B>>;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        fn eq_array(l: Array, r: Number) -> Array {
            l.eq_const(r)
        }

        fn eq_number(l: Number, r: Number) -> Number {
            (l == r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, eq_array, eq_number).into())
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        fn gt_array(l: Array, r: Number) -> Array {
            l.gt_const(r)
        }

        fn gt_number(l: Number, r: Number) -> Number {
            (l > r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, gt_array, gt_number).into())
    }

    fn gte_const(self, other: Number) -> TCResult<Self::Compare> {
        fn gte_array(l: Array, r: Number) -> Array {
            l.gte_const(r)
        }

        fn gte_number(l: Number, r: Number) -> Number {
            (l >= r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, gte_array, gte_number).into())
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        fn lt_array(l: Array, r: Number) -> Array {
            l.lt_const(r)
        }

        fn lt_number(l: Number, r: Number) -> Number {
            (l < r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, lt_array, lt_number).into())
    }

    fn lte_const(self, other: Number) -> TCResult<Self::Compare> {
        fn lte_array(l: Array, r: Number) -> Array {
            l.lte_const(r)
        }

        fn lte_number(l: Number, r: Number) -> Number {
            (l <= r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, lte_array, lte_number).into())
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        fn ne_array(l: Array, r: Number) -> Array {
            l.ne_const(r)
        }

        fn ne_number(l: Number, r: Number) -> Number {
            (l != r).into()
        }

        Ok(BlockListConst::new(self.blocks, other, ne_array, ne_number).into())
    }
}

#[async_trait]
impl<FD, FS, D, T, B> TensorDiagonal<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseWrite<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;
    type Diagonal = DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>;

    async fn diagonal(self, txn: Self::Txn) -> TCResult<Self::Diagonal> {
        if self.ndim() < 2 {
            return Err(TCError::bad_request(
                "cannot take the diagonal of a Tensor with shape",
                self.shape(),
            ));
        }

        let size = self.shape()[self.ndim() - 1];
        if size != self.shape()[self.ndim() - 2] {
            return Err(TCError::bad_request(
                "diagonal requires a square matrix but found",
                self.shape(),
            ));
        }

        let txn_id = *txn.id();
        let file = txn.context().create_file_unique(txn_id).await?;

        let ndim = self.ndim();
        let dtype = self.dtype();

        if ndim == 2 {
            let blocks = self.blocks;

            // TODO: is is really necessary to allocate a new Vec for every Coord?
            let coords = futures::stream::iter((0..size).into_iter().map(|i| Ok(vec![i, i])));
            let values = CoordBlocks::new(coords, 2, PER_BLOCK)
                .map_ok(|coords| blocks.clone().read_values(txn.clone(), coords))
                .try_buffered(num_cpus::get());

            let shape = vec![size].into();
            let blocks =
                BlockListFile::from_blocks(file, txn_id, Some(shape), dtype, values).await?;

            return Ok(blocks.into());
        }

        let dest_shape = self.shape()[0..self.ndim() - 1].to_vec().into();
        let diagonal = BlockListFile::constant(file, *txn.id(), dest_shape, dtype.zero()).await?;
        let dest = diagonal.clone();

        let source_shape = self.shape().clone();
        futures::stream::iter((0..size).into_iter().map(move |i| {
            let mut source_bounds = Bounds::all(&source_shape);
            source_bounds[ndim - 2] = i.into();
            source_bounds[ndim - 1] = i.into();

            let dest_bounds = source_bounds[0..ndim - 1].to_vec().into();
            (source_bounds, dest_bounds)
        }))
        .map(move |(source_bounds, dest_bounds)| {
            let source = self.clone();
            let dest = dest.clone();
            let txn = txn.clone();

            async move {
                let slice = source.slice(source_bounds)?;
                dest.write(txn, dest_bounds, slice.blocks).await
            }
        })
        .buffer_unordered(num_cpus::get())
        .try_fold((), |_, _| future::ready(Ok(())))
        .await?;

        Ok(diagonal.into())
    }
}

#[async_trait]
impl<FD, FS, D, T, B> TensorIO<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseWrite<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;

    async fn read_value(self, txn: Self::Txn, coord: Coord) -> TCResult<Number> {
        self.blocks
            .read_value_at(txn, coord)
            .map_ok(|(_, val)| val)
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        self.blocks.write_value(txn_id, bounds, value).await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        debug!("DenseTensor::write_value_at");

        self.blocks
            .write_value(txn_id, Bounds::from(coord), value)
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T, B> TensorDualIO<D, DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>>
    for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseWrite<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;

    async fn write(
        self,
        txn: T,
        bounds: Bounds,
        other: DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>,
    ) -> TCResult<()> {
        debug!("write {} to dense {}", other, bounds);
        self.blocks.write(txn, bounds, other.blocks).await
    }
}

#[async_trait]
impl<FD, FS, D, T, B> TensorIndex<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseWrite<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;
    type Index = DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>;

    async fn argmax(self, txn: Self::Txn, axis: usize) -> TCResult<Self::Index> {
        if axis >= self.ndim() {
            return Err(TCError::unsupported(format!(
                "invalid argmax axis for tensor with {} dimensions: {}",
                self.ndim(),
                axis
            )));
        }

        let dtype = NumberType::UInt(UIntType::U64);
        let shape = {
            let mut shape = self.shape().clone();
            shape.remove(axis);
            shape
        };

        let txn_id = *txn.id();
        let file = txn.context().create_file_unique(txn_id).await?;

        let per_block = self.size() / shape.size();
        debug_assert_eq!(self.size() / per_block, shape.size());

        let blocks = self.blocks.value_stream(txn).await?; // TODO: just use self.block_stream directly
        let values = blocks
            .chunks(per_block as usize)
            .map(|values| values.into_iter().collect::<TCResult<Vec<Number>>>())
            .map_ok(Array::from)
            .map_ok(|array| {
                debug_assert!(array.len() as u64 == per_block);
                let (i, _max) = array.argmax();
                Number::from(i as u64)
            });

        BlockListFile::from_values(file, txn_id, shape, dtype, values)
            .map_ok(DenseTensor::from)
            .await
    }

    async fn argmax_all(self, txn: Self::Txn) -> TCResult<u64> {
        let mut offset = 0;
        let mut max_value = self.dtype().zero();
        let mut argmax = 0;

        let mut blocks = self.blocks.block_stream(txn).await?;
        while let Some(block) = blocks.try_next().await? {
            let (i, max) = block.argmax();
            if max > max_value {
                argmax = offset + (i as u64);
                max_value = max;
            }

            offset += block.len() as u64;
        }

        Ok(argmax)
    }
}

#[async_trait]
impl<FD, FS, D, T, B> TensorDualIO<D, Tensor<FD, FS, D, T>> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseWrite<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;

    async fn write(self, txn: T, bounds: Bounds, other: Tensor<FD, FS, D, T>) -> TCResult<()> {
        debug!("DenseTensor::write {} to {}", other, bounds);

        let shape = bounds.to_shape(self.shape())?;
        let other = if other.shape() == &shape {
            other
        } else {
            other.broadcast(shape)?
        };

        match other {
            Tensor::Dense(dense) => match dense.blocks {
                DenseAccessor::File(file) => self.write(txn, bounds, DenseTensor::from(file)).await,
                other => {
                    let file = txn.context().create_file_unique(*txn.id()).await?;

                    let other = DenseTensor::copy_from(other.into(), file, &txn).await?;
                    self.write(txn, bounds, other).await
                }
            },
            Tensor::Sparse(sparse) => {
                let file = txn.context().create_file_unique(*txn.id()).await?;

                let other = DenseTensor::copy_from(sparse.into_dense(), file, &txn).await?;
                self.write(txn, bounds, other).await
            }
        }
    }
}

impl<FD, FS, D, T, B, O> TensorMath<D, DenseTensor<FD, FS, D, T, O>>
    for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    O: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Combine = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;
    type LeftCombine = DenseTensor<FD, FS, D, T, BlockListCombine<FD, FS, D, T, B, O>>;

    fn add(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn add_array(l: &Array, r: &Array) -> Array {
            debug_assert_eq!(l.len(), r.len());
            l + r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, add_array, Add::add, dtype)
    }

    fn div(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn div_array(l: &Array, r: &Array) -> Array {
            if !r.all() {
                warn!("divide by zero in DenseTensor::div");
            }

            debug_assert_eq!(l.len(), r.len());
            l / r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, div_array, Div::div, dtype)
    }

    fn log(self, base: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::LeftCombine> {
        fn log(n: Number, base: Number) -> Number {
            n.log(Float::cast_from(base))
        }

        let dtype = self.dtype().one().ln().class();
        self.combine(base, Array::log, log, dtype)
    }

    fn mul(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn mul_array(l: &Array, r: &Array) -> Array {
            debug_assert_eq!(l.len(), r.len());
            l * r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, mul_array, Mul::mul, dtype)
    }

    fn pow(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        if !other.dtype().is_real() {
            return Err(TCError::unsupported(ERR_COMPLEX_EXPONENT));
        }

        fn pow_array(l: &Array, r: &Array) -> Array {
            debug_assert_eq!(l.len(), r.len());
            l.pow(r)
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, pow_array, Number::pow, dtype)
    }

    fn sub(self, other: DenseTensor<FD, FS, D, T, O>) -> TCResult<Self::Combine> {
        fn sub_array(l: &Array, r: &Array) -> Array {
            debug_assert_eq!(l.len(), r.len());
            l - r
        }

        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, sub_array, Sub::sub, dtype)
    }
}

impl<FD, FS, D, T, B> TensorMath<D, Tensor<FD, FS, D, T>> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Read: DirReadFile<FS>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Combine = Tensor<FD, FS, D, T>;
    type LeftCombine = Tensor<FD, FS, D, T>;

    fn add(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.add(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.add(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn div(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.div(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.div(sparse.into_dense()).map(Tensor::from),
        }
    }

    fn log(self, base: Tensor<FD, FS, D, T>) -> TCResult<Self::LeftCombine> {
        match base {
            Tensor::Dense(dense) => self.log(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.into_sparse().log(sparse).map(Tensor::from),
        }
    }

    fn mul(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.mul(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => sparse.mul(self.into_sparse()).map(Tensor::from),
        }
    }

    fn pow(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.pow(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => sparse.pow(self.into_sparse()).map(Tensor::from),
        }
    }

    fn sub(self, other: Tensor<FD, FS, D, T>) -> TCResult<Self::Combine> {
        match other {
            Tensor::Dense(dense) => self.sub(dense).map(Tensor::from),
            Tensor::Sparse(sparse) => self.sub(sparse.into_dense()).map(Tensor::from),
        }
    }
}

impl<FD, FS, D, T, B> TensorMathConst for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
{
    type Combine = DenseTensor<FD, FS, D, T, BlockListConst<FD, FS, D, T, B>>;
    type DenseCombine = DenseTensor<FD, FS, D, T, BlockListConst<FD, FS, D, T, B>>;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
        fn add_array(l: Array, r: Number) -> Array {
            &l + r
        }

        Ok(BlockListConst::new(self.blocks, other, add_array, Number::add).into())
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        fn div_array(l: Array, r: Number) -> Array {
            &l / r
        }

        Ok(BlockListConst::new(self.blocks, other, div_array, Number::div).into())
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        if base.class().is_complex() {
            return Err(TCError::unsupported(ERR_COMPLEX_EXPONENT));
        }

        let base = Number::Float(base.cast_into());

        fn log(n: Number, base: Number) -> Number {
            if let Number::Float(base) = base {
                n.log(base)
            } else {
                unreachable!("log with non-floating point base")
            }
        }

        fn log_array(l: Array, r: Number) -> Array {
            l.log_const(r)
        }

        Ok(BlockListConst::new(self.blocks, base, log_array, log).into())
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        fn mul_array(l: Array, r: Number) -> Array {
            &l * r
        }

        Ok(BlockListConst::new(self.blocks, other, mul_array, Number::mul).into())
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        if !other.class().is_real() {
            return Err(TCError::unsupported(ERR_COMPLEX_EXPONENT));
        }

        fn pow_array(l: Array, r: Number) -> Array {
            l.pow_const(r)
        }

        Ok(BlockListConst::new(self.blocks, other, pow_array, Number::pow).into())
    }

    fn sub_const(self, other: Number) -> TCResult<Self::Combine> {
        fn sub_array(l: Array, r: Number) -> Array {
            &l - r
        }

        Ok(BlockListConst::new(self.blocks, other, sub_array, Number::sub).into())
    }
}

impl<FD, FS, D, T, B> ReadValueAt<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        self.blocks.read_value_at(txn, coord)
    }
}

impl<FD, FS, D, T, B> TensorReduce<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;
    type Reduce = DenseTensor<FD, FS, D, T, BlockListReduce<FD, FS, D, T, B>>;

    fn max(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce> {
        BlockListReduce::max(self.blocks, axis, keepdims).map(DenseTensor::from)
    }

    fn max_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let zero = self.dtype().zero();
            let blocks = self.blocks.clone().block_stream(txn).await?;
            let collator = NumberCollator::default();

            blocks
                .map_ok(|array| array.max())
                .try_fold(zero, move |max, block_max| {
                    future::ready(Ok(match collator.compare(&max, &block_max) {
                        Ordering::Greater => max,
                        Ordering::Equal => max,
                        Ordering::Less => block_max,
                    }))
                })
                .await
        })
    }

    fn min(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce> {
        BlockListReduce::min(self.blocks, axis, keepdims).map(DenseTensor::from)
    }

    fn min_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let zero = self.dtype().zero();
            let blocks = self.blocks.clone().block_stream(txn).await?;
            let collator = NumberCollator::default();

            blocks
                .map_ok(|array| array.max())
                .try_fold(zero, move |min, block_min| {
                    future::ready(Ok(match collator.compare(&min, &block_min) {
                        Ordering::Less => min,
                        Ordering::Equal => min,
                        Ordering::Greater => block_min,
                    }))
                })
                .await
        })
    }

    fn product(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce> {
        BlockListReduce::product(self.blocks, axis, keepdims).map(DenseTensor::from)
    }

    fn product_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let zero = self.dtype().zero();
            let mut product = self.dtype().one();

            let blocks = self.blocks.clone().block_stream(txn).await?;
            let mut block_products = blocks.map_ok(|array| array.product());

            while let Some(block_product) = block_products.try_next().await? {
                if block_product == zero {
                    return Ok(zero);
                }

                product = product * block_product;
            }

            Ok(product)
        })
    }

    fn sum(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce> {
        BlockListReduce::sum(self.blocks, axis, keepdims).map(DenseTensor::from)
    }

    fn sum_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let zero = self.dtype().zero();
            let blocks = self.blocks.clone().block_stream(txn).await?;

            blocks
                .map_ok(|array| array.sum())
                .try_fold(zero, |sum, block_sum| future::ready(Ok(sum + block_sum)))
                .await
        })
    }
}

impl<FD, FS, D, T, B> TensorTransform for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Broadcast = DenseTensor<FD, FS, D, T, DenseAccessor<FD, FS, D, T>>;
    type Cast = DenseTensor<FD, FS, D, T, BlockListCast<FD, FS, D, T, B>>;
    type Expand = DenseTensor<FD, FS, D, T, BlockListExpand<FD, FS, D, T, B>>;
    type Flip = DenseTensor<FD, FS, D, T, BlockListFlip<FD, FS, D, T, B>>;
    type Reshape = DenseTensor<FD, FS, D, T, BlockListReshape<FD, FS, D, T, B>>;
    type Slice = DenseTensor<FD, FS, D, T, B::Slice>;
    type Transpose = DenseTensor<FD, FS, D, T, B::Transpose>;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        if self.shape() == &shape {
            return Ok(self.into_inner().accessor().into());
        }

        let blocks = BlockListBroadcast::new(self.blocks, shape)?;
        Ok(DenseTensor::from(blocks.accessor()))
    }

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        if self.dtype().is_complex() && dtype.is_real() {
            return Err(TCError::unsupported("cannot cast a complex Tensor into a real Tensor; consider the real, imag, or abs methods instead"));
        }

        let blocks = BlockListCast::new(self.blocks, dtype);
        Ok(DenseTensor::from(blocks))
    }

    fn expand_dims(self, axis: usize) -> TCResult<Self::Expand> {
        let blocks = BlockListExpand::new(self.blocks, axis)?;
        Ok(DenseTensor::from(blocks))
    }

    fn flip(self, axis: usize) -> TCResult<Self::Flip> {
        let blocks = BlockListFlip::new(self.blocks, axis)?;
        Ok(DenseTensor::from(blocks))
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        let blocks = BlockListReshape::new(self.blocks, shape)?;
        Ok(DenseTensor::from(blocks))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let blocks = self.blocks.slice(bounds)?;
        Ok(DenseTensor::from(blocks))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let blocks = self.blocks.transpose(permutation)?;
        Ok(DenseTensor::from(blocks))
    }
}

macro_rules! trig {
    ($fun:ident) => {
        fn $fun(&self) -> TCResult<Self::Unary> {
            let dtype = trig_dtype(self.dtype());
            let blocks = BlockListUnary::new(self.blocks.clone(), Array::$fun, Number::$fun, dtype);
            Ok(DenseTensor::from(blocks))
        }
    };
}

#[async_trait]
impl<FD, FS, D, T, B> TensorTrig for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
{
    type Unary = DenseTensor<FD, FS, D, T, BlockListUnary<FD, FS, D, T, B>>;

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
impl<FD, FS, D, T, B> TensorUnary<D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;
    type Unary = DenseTensor<FD, FS, D, T, BlockListUnary<FD, FS, D, T, B>>;

    fn abs(&self) -> TCResult<Self::Unary> {
        let blocks = BlockListUnary::new(
            self.blocks.clone(),
            Array::abs,
            <Number as NumberInstance>::abs,
            self.dtype().one().abs().class(),
        );

        Ok(DenseTensor::from(blocks))
    }

    fn exp(&self) -> TCResult<Self::Unary> {
        fn exp(n: Number) -> Number {
            match n {
                Number::Complex(n) => n.exp().into(),
                Number::Float(n) => n.exp().into(),
                n => f64::cast_from(n).exp().into(),
            }
        }

        debug!("{} is complex? {}", self.dtype(), self.dtype().is_complex());

        let dtype = if self.dtype().is_complex() {
            NumberType::Complex(ComplexType::C64)
        } else {
            NumberType::Float(FloatType::F64)
        };

        debug!("e**{} will have dtype {}", self, dtype);
        let blocks = BlockListUnary::new(self.blocks.clone(), Array::exp, exp, dtype);

        Ok(DenseTensor::from(blocks))
    }

    fn ln(&self) -> TCResult<Self::Unary> {
        let dtype = self.dtype().one().ln().class();
        let blocks = BlockListUnary::new(self.blocks.clone(), Array::ln, Number::ln, dtype);
        Ok(DenseTensor::from(blocks))
    }

    fn round(&self) -> TCResult<Self::Unary> {
        let dtype = self.dtype().one().round().class();
        let blocks = BlockListUnary::new(self.blocks.clone(), Array::round, Number::round, dtype);
        Ok(DenseTensor::from(blocks))
    }

    async fn all(self, txn: T) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(txn).await?;

        while let Some(array) = blocks.try_next().await? {
            if !array.all() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(self, txn: T) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(txn).await?;
        while let Some(array) = blocks.try_next().await? {
            if array.any() {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn not(&self) -> TCResult<Self::Unary> {
        let blocks = BlockListUnary::new(
            self.blocks.clone(),
            Array::not,
            Number::not,
            NumberType::Bool,
        );

        Ok(DenseTensor::from(blocks))
    }
}

#[async_trait]
impl<FD, FS, D, T> Transact for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array> + Transact,
    FS: File<Key = NodeId, Block = Node>,
    T: Transaction<D>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Commit = <BlockListFile<FD, FS, D, T> as Transact>::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        self.blocks.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.blocks.finalize(txn_id).await
    }
}

#[async_trait]
impl<FD, FS, D, T, B> CopyFrom<D, DenseTensor<FD, FS, D, T, B>>
    for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    async fn copy_from(
        instance: DenseTensor<FD, FS, D, T, B>,
        file: FD,
        txn: &T,
    ) -> TCResult<Self> {
        BlockListFile::copy_from(instance.blocks, file, txn)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T> Persist<D> for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    T: Transaction<D>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Schema = Schema;
    type Store = FD;
    type Txn = T;

    async fn create(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        BlockListFile::create(txn, schema, store)
            .map_ok(Self::from)
            .await
    }

    async fn load(txn: &T, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        BlockListFile::load(txn, schema, store)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T> Restore<D> for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    T: Transaction<D>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        self.blocks.restore(&backup.blocks, txn_id).await
    }
}

impl<FD, FS, D, T, B> From<B> for DenseTensor<FD, FS, D, T, B> {
    fn from(blocks: B) -> Self {
        Self {
            blocks,
            phantom: Phantom::default(),
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> de::FromStream for DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    T: Transaction<D>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Context = T;

    async fn from_stream<De: de::Decoder>(txn: T, decoder: &mut De) -> Result<Self, De::Error> {
        let txn_id = *txn.id();
        let file = txn
            .context()
            .create_file_unique(txn_id)
            .map_err(de::Error::custom)
            .await?;

        decoder
            .decode_seq(DenseTensorVisitor::new(txn_id, file))
            .await
    }
}

impl<FD, FS, D, T, B> fmt::Display for DenseTensor<FD, FS, D, T, B>
where
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a dense Tensor with dtype {} and shape {}",
            self.dtype(),
            self.shape()
        )
    }
}

struct DenseTensorVisitor<FD, FS, D, T> {
    txn_id: TxnId,
    file: FD,
    sparse: PhantomData<FS>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<FD, FS, D, T> DenseTensorVisitor<FD, FS, D, T> {
    fn new(txn_id: TxnId, file: FD) -> Self {
        Self {
            txn_id,
            file,
            sparse: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> de::Visitor for DenseTensorVisitor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    T: Transaction<D>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Value = DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>;

    fn expecting() -> &'static str {
        "a dense tensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        debug!("deserialize DenseTensor");

        let schema = seq.next_element::<Schema>(()).await?;
        let schema = schema.ok_or_else(|| de::Error::invalid_length(0, "a tensor schema"))?;
        schema.validate("load Dense").map_err(de::Error::custom)?;
        debug!("DenseTensor schema is {}", schema);

        let cxt = (self.txn_id, self.file, schema);
        let blocks = seq.next_element::<BlockListFile<FD, FS, D, T>>(cxt).await?;

        let blocks = blocks.ok_or_else(|| de::Error::invalid_length(1, "dense tensor data"))?;

        Ok(DenseTensor::from(blocks))
    }
}

#[async_trait]
impl<'en, FD, FS, D, T, B> IntoView<'en, D> for DenseTensor<FD, FS, D, T, B>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Key = u64, Block = Array>,
    FS: File<Key = NodeId, Block = Node>,
    B: DenseAccess<FD, FS, D, T>,
    D::Write: DirCreateFile<FS> + DirCreateFile<FD>,
{
    type Txn = T;
    type View = DenseTensorView<'en>;

    async fn into_view(self, txn: T) -> TCResult<DenseTensorView<'en>> {
        let shape = self.shape().clone();
        let dtype = self.dtype();
        let blocks = self.blocks.block_stream(txn).await?;

        Ok(DenseTensorView {
            schema: Schema { shape, dtype },
            blocks: BlockStreamView { dtype, blocks },
        })
    }
}

/// A view of a [`DenseTensor`] as of a specific [`TxnId`], used in serialization
pub struct DenseTensorView<'en> {
    schema: Schema,
    blocks: BlockStreamView<'en>,
}

#[async_trait]
impl<'en> en::IntoStream<'en> for DenseTensorView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeSeq;

        let mut seq = encoder.encode_seq(Some(2))?;
        seq.encode_element(self.schema)?;
        seq.encode_element(self.blocks)?;
        seq.end()
    }
}

struct BlockStreamView<'en> {
    dtype: NumberType,
    blocks: TCBoxTryStream<'en, Array>,
}

impl<'en> en::IntoStream<'en> for BlockStreamView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use tc_value::{
            ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT,
        };

        debug!("stream Tensor with dtype {}", self.dtype);

        fn encodable<'en, T>(blocks: TCBoxTryStream<'en, Array>) -> impl Stream<Item = Vec<T>> + 'en
        where
            T: af::HasAfEnum + Clone + Default + 'en,
            Array: AsType<ArrayExt<T>>,
        {
            // an error can't be encoded within an array
            // so in case of a read error, let the receiver figure out that the tensor
            // doesn't have enough elements

            #[cfg(not(debug_assertions))]
            {
                blocks
                    .take_while(|r| future::ready(r.is_ok()))
                    .map(|r| r.expect("tensor block"))
                    // TODO: explicitly catalog ArrayFire return types to avoid this cast
                    .map(|arr| arr.cast_into())
                    .map(|arr: ArrayExt<T>| arr.to_vec())
            }

            #[cfg(debug_assertions)]
            blocks
                .inspect_ok(|arr| {
                    debug!(
                        "{} is expected to have type {} but really has type {}",
                        arr,
                        std::any::type_name::<T>(),
                        arr.dtype(),
                    )
                })
                .map(|r| r.expect("tensor block"))
                // TODO: explicitly catalog ArrayFire return types to avoid this cast
                .map(|arr| arr.cast_into())
                .map(|arr: ArrayExt<T>| arr.to_vec())
        }

        match self.dtype {
            NT::Bool => encoder.encode_array_bool(encodable(self.blocks)),
            NT::Complex(ct) => match ct {
                CT::C32 => encoder.encode_array_f32(encodable_c32(self.blocks)),
                _ => encoder.encode_array_f64(encodable_c64(self.blocks)),
            },
            NT::Float(ft) => match ft {
                FT::F32 => encoder.encode_array_f32(encodable(self.blocks)),
                _ => encoder.encode_array_f64(encodable(self.blocks)),
            },
            NT::Int(it) => match it {
                IT::I8 | IT::I16 => encoder.encode_array_i16(encodable(self.blocks)),
                IT::I32 => encoder.encode_array_i32(encodable(self.blocks)),
                _ => encoder.encode_array_i64(encodable(self.blocks)),
            },
            NT::UInt(ut) => match ut {
                UT::U8 => encoder.encode_array_u8(encodable(self.blocks)),
                UT::U16 => encoder.encode_array_u16(encodable(self.blocks)),
                UT::U32 => encoder.encode_array_u32(encodable(self.blocks)),
                _ => encoder.encode_array_u64(encodable(self.blocks)),
            },
            NT::Number => Err(en::Error::custom(format!(
                "invalid Tensor data type: {}",
                NT::Number
            ))),
        }
    }
}

fn encodable_c32<'en>(blocks: TCBoxTryStream<'en, Array>) -> impl Stream<Item = Vec<f32>> + 'en {
    blocks
        .take_while(|r| future::ready(r.is_ok()))
        .map(|r| r.expect("tensor block"))
        .map(|arr| arr.type_cast())
        .map(|source: ArrayExt<_Complex<f32>>| {
            let re = source.re();
            assert_eq!(source.len(), re.len());
            let im = source.im();
            assert_eq!(source.len(), im.len());

            let mut i = 0;
            let mut dest = vec![0.; source.len() * 2];
            for (re, im) in re.to_vec().into_iter().zip(im.to_vec()) {
                dest[i] = re;
                dest[i + 1] = im;
                i += 2;
            }

            dest
        })
}

fn encodable_c64<'en>(blocks: TCBoxTryStream<'en, Array>) -> impl Stream<Item = Vec<f64>> + 'en {
    blocks
        .take_while(|r| future::ready(r.is_ok()))
        .map(|r| r.expect("tensor block"))
        .map(|arr| arr.type_cast())
        .map(|source: ArrayExt<_Complex<f64>>| {
            let re = source.re();
            assert_eq!(source.len(), re.len());
            let im = source.im();
            assert_eq!(source.len(), im.len());

            let mut i = 0;
            let mut dest = vec![0.; source.len() * 2];
            for (re, im) in re.to_vec().into_iter().zip(im.to_vec()) {
                dest[i] = re;
                dest[i + 1] = im;
                i += 2;
            }

            dest
        })
}

#[inline]
fn array_err(err: afarray::ArrayError) -> TCError {
    TCError::new(ErrorType::BadRequest, err.to_string())
}

#[inline]
fn div_ceil(l: u64, r: u64) -> u64 {
    if l % r == 0 {
        l / r
    } else {
        (l / r) + 1
    }
}
