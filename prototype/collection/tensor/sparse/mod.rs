use std::pin::Pin;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};

use crate::class::Instance;
use crate::collection::Collection;
use crate::error;
use crate::handler::*;
use crate::scalar::value::number::*;
use crate::scalar::{MethodType, PathSegment};
use crate::transaction::{Transact, Txn, TxnId};
use crate::{TCBoxTryFuture, TCResult};

use super::bounds::{Bounds, Shape};
use super::class::{Tensor, TensorInstance, TensorType};
use super::dense::{
    dense_constant, from_sparse, BlockListFile, BlockListSparse, DenseAccess, DenseTensor,
};
use super::stream::*;
use super::{
    broadcast, Coord, IntoView, TensorAccess, TensorBoolean, TensorCompare, TensorDualIO, TensorIO,
    TensorMath, TensorReduce, TensorTransform, TensorUnary, ERR_NONBIJECTIVE_WRITE,
};

mod access;
mod combine;
mod table;

pub use access::*;
pub use table::*;

pub type SparseRow = (Coord, Number);
pub type SparseStream<'a> = Pin<Box<dyn Stream<Item = TCResult<SparseRow>> + Send + Unpin + 'a>>;

const ERR_NOT_SPARSE: &str = "The result of the requested operation would not be sparse;\
convert to a DenseTensor first.";

#[derive(Clone)]
pub struct SparseTensor<T: Clone + SparseAccess> {
    accessor: T,
}

impl<T: Clone + SparseAccess> SparseTensor<T> {
    pub fn into_inner(self) -> T {
        self.accessor
    }

    pub async fn copy(&self, txn: &Txn) -> TCResult<SparseTensor<SparseTable>> {
        self.accessor
            .copy(txn)
            .await
            .map(|accessor| SparseTensor { accessor })
    }

    pub async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        self.accessor.filled(txn).await
    }

    fn combine<OT: Clone + SparseAccess>(
        &self,
        other: &SparseTensor<OT>,
        combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<SparseTensor<SparseCombinator<T, OT>>> {
        if self.shape() != other.shape() {
            return Err(error::unsupported(format!(
                "Cannot combine Tensors of different shapes: {}, {}",
                self.shape(),
                other.shape()
            )));
        }

        let accessor = SparseCombinator::new(
            self.accessor.clone(),
            other.accessor.clone(),
            combinator,
            dtype,
        )?;

        Ok(SparseTensor { accessor })
    }

    fn condense<'a, OT: Clone + SparseAccess>(
        &'a self,
        other: &'a SparseTensor<OT>,
        txn: &'a Txn,
        default: Number,
        condensor: fn(Number, Number) -> Number,
    ) -> TCBoxTryFuture<'a, DenseTensor<BlockListFile>> {
        Box::pin(async move {
            if self.shape() != other.shape() {
                let (this, that) = broadcast(self, other)?;
                return this.condense(&that, txn, default, condensor).await;
            }

            let accessor = SparseCombinator::new(
                self.accessor.clone(),
                other.accessor.clone(),
                condensor,
                default.class(),
            )?;

            let condensed = dense_constant(&txn, self.shape().clone(), default).await?;

            let txn_id = *txn.id();
            accessor
                .filled(txn)
                .await?
                .map_ok(|(coord, value)| condensed.write_value_at(txn_id, coord, value))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await?;

            Ok(condensed)
        })
    }
}

impl<T: Clone + SparseAccess> ReadValueAt for SparseTensor<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        self.accessor.read_value_at(txn, coord)
    }
}

impl<T: Clone + SparseAccess> Instance for SparseTensor<T> {
    type Class = TensorType;

    fn class(&self) -> TensorType {
        TensorType::Sparse
    }
}

impl<T: Clone + SparseAccess> TensorInstance for SparseTensor<T> {
    type Dense = DenseTensor<BlockListSparse<T>>;
    type Sparse = Self;

    fn into_dense(self) -> Self::Dense {
        from_sparse(self)
    }

    fn into_sparse(self) -> Self::Sparse {
        self
    }
}

impl<T: Clone + SparseAccess> IntoView for SparseTensor<T> {
    fn into_view(self) -> Tensor {
        let accessor = self.into_inner().accessor();
        Tensor::Sparse(SparseTensor { accessor })
    }
}

impl<T: Clone + SparseAccess> TensorAccess for SparseTensor<T> {
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn ndim(&self) -> usize {
        self.accessor.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.accessor.shape()
    }

    fn size(&self) -> u64 {
        self.accessor.size()
    }
}

#[async_trait]
impl<T: Clone + SparseAccess, OT: Clone + SparseAccess> TensorBoolean<SparseTensor<OT>>
    for SparseTensor<T>
{
    type Combine = SparseTensor<SparseCombinator<T, OT>>;

    fn and(&self, other: &SparseTensor<OT>) -> TCResult<Self::Combine> {
        // TODO: use a custom method for this, to only iterate over self.filled (not other.filled)
        self.combine(other, Number::and, NumberType::Bool)
    }

    fn or(&self, other: &SparseTensor<OT>) -> TCResult<Self::Combine> {
        self.combine(other, Number::or, NumberType::Bool)
    }

    fn xor(&self, _other: &SparseTensor<OT>) -> TCResult<Self::Combine> {
        Err(error::unsupported(ERR_NOT_SPARSE))
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> TensorUnary for SparseTensor<T> {
    type Unary = SparseTensor<SparseUnary>;

    fn abs(&self) -> TCResult<Self::Unary> {
        let source = self.accessor.clone().accessor();
        let transform = <Number as NumberInstance>::abs;

        let accessor = SparseUnary::new(source, transform, self.dtype());
        Ok(SparseTensor { accessor })
    }

    async fn all(&self, txn: &Txn) -> TCResult<bool> {
        let mut coords = self
            .accessor
            .filled(txn)
            .await?
            .map_ok(|(coord, _)| coord)
            .zip(stream::iter(Bounds::all(self.shape()).affected()))
            .map(|(r, expected)| r.map(|actual| (actual, expected)));

        while let Some(result) = coords.next().await {
            let (actual, expected) = result?;
            if actual != expected {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(&self, txn: &Txn) -> TCResult<bool> {
        let mut filled = self.accessor.filled(txn).await?;
        Ok(filled.next().await.is_some())
    }

    fn not(&self) -> TCResult<Self::Unary> {
        Err(error::unsupported(ERR_NOT_SPARSE))
    }
}

#[async_trait]
impl<T: Clone + SparseAccess, OT: Clone + SparseAccess> TensorCompare<SparseTensor<OT>>
    for SparseTensor<T>
{
    type Compare = SparseTensor<SparseCombinator<T, OT>>;
    type Dense = DenseTensor<BlockListFile>;

    async fn eq(&self, other: &SparseTensor<OT>, txn: &Txn) -> TCResult<Self::Dense> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::eq)
            .await
    }

    fn gt(&self, other: &SparseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(other, <Number as NumberInstance>::gt, NumberType::Bool)
    }

    async fn gte(&self, other: &SparseTensor<OT>, txn: &Txn) -> TCResult<Self::Dense> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::gte)
            .await
    }

    fn lt(&self, other: &SparseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(other, <Number as NumberInstance>::lt, NumberType::Bool)
    }

    async fn lte(&self, other: &SparseTensor<OT>, txn: &Txn) -> TCResult<Self::Dense> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::lte)
            .await
    }

    fn ne(&self, other: &SparseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(other, <Number as NumberInstance>::ne, NumberType::Bool)
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> TensorIO for SparseTensor<T> {
    async fn read_value(&self, txn: &Txn, coord: Coord) -> TCResult<Number> {
        self.accessor
            .read_value_at(txn, coord)
            .map_ok(|(_coord, value)| value)
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        if self.shape().is_empty() {
            self.write_value_at(txn_id, vec![], value).await
        } else {
            stream::iter(bounds.affected())
                .map(|coord| Ok(self.write_value_at(txn_id, coord, value.clone())))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        }
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.accessor.write_value(txn_id, coord, value).await
    }
}

#[async_trait]
impl<T: Clone + SparseAccess, OT: Clone + SparseAccess> TensorDualIO<SparseTensor<OT>>
    for SparseTensor<T>
{
    async fn mask(&self, txn: &Txn, other: SparseTensor<OT>) -> TCResult<()> {
        let zero = self.dtype().zero();
        let txn_id = *txn.id();

        other
            .filled(txn)
            .await?
            .map_ok(|(coord, _)| self.write_value_at(txn_id, coord, zero.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn write(&self, txn: &Txn, bounds: Bounds, other: SparseTensor<OT>) -> TCResult<()> {
        let slice = self.slice(bounds)?;
        if slice.shape() != other.shape() {
            return Err(error::unsupported(format!(
                "Cannot write Tensor with shape {} to slice with shape {}",
                other.shape(),
                slice.shape()
            )));
        }

        let txn_id = *txn.id();
        let filled = other.filled(txn).await?;
        filled
            .map_ok(|(coord, value)| slice.write_value_at(txn_id, coord, value))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> TensorDualIO<Tensor> for SparseTensor<T> {
    async fn mask(&self, txn: &Txn, other: Tensor) -> TCResult<()> {
        match other {
            Tensor::Dense(dense) => self.mask(txn, dense.into_sparse()).await,
            Tensor::Sparse(sparse) => self.mask(txn, sparse).await,
        }
    }

    async fn write(&self, txn: &Txn, bounds: Bounds, value: Tensor) -> TCResult<()> {
        match value {
            Tensor::Sparse(sparse) => self.write(txn, bounds, sparse).await,
            Tensor::Dense(dense) => self.write(txn, bounds, dense.into_sparse()).await,
        }
    }
}

impl<T: Clone + SparseAccess, OT: Clone + SparseAccess> TensorMath<SparseTensor<OT>>
    for SparseTensor<T>
{
    type Combine = SparseTensor<SparseCombinator<T, OT>>;

    fn add(&self, other: &SparseTensor<OT>) -> TCResult<Self::Combine> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, <Number as NumberInstance>::add, dtype)
    }

    fn multiply(&self, other: &SparseTensor<OT>) -> TCResult<Self::Combine> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, <Number as NumberInstance>::multiply, dtype)
    }
}

impl<T: Clone + SparseAccess> TensorReduce for SparseTensor<T> {
    type Reduce = SparseTensor<SparseReduce<T>>;

    fn product(&self, axis: usize) -> TCResult<Self::Reduce> {
        let accessor = SparseReduce::new(self.clone(), axis, SparseTensor::product_all)?;
        Ok(SparseTensor { accessor })
    }

    fn product_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            if self.all(&txn).await? {
                from_sparse(self.clone()).product_all(txn).await
            } else {
                Ok(self.dtype().zero())
            }
        })
    }

    fn sum(&self, axis: usize) -> TCResult<Self::Reduce> {
        let accessor = SparseReduce::new(self.clone(), axis, SparseTensor::sum_all)?;
        Ok(SparseTensor { accessor })
    }

    fn sum_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            if self.any(&txn).await? {
                from_sparse(self.clone()).sum_all(txn).await
            } else {
                Ok(self.dtype().zero())
            }
        })
    }
}

impl<T: Clone + SparseAccess> TensorTransform for SparseTensor<T> {
    type Cast = SparseTensor<SparseCast<T>>;
    type Broadcast = SparseTensor<SparseBroadcast<T>>;
    type Expand = SparseTensor<SparseExpand<T>>;
    type Slice = SparseTensor<<T as SparseAccess>::Slice>;
    type Transpose = SparseTensor<<T as SparseAccess>::Transpose>;

    fn as_type(&self, dtype: NumberType) -> TCResult<Self::Cast> {
        let accessor = SparseCast::new(self.accessor.clone(), dtype);
        Ok(accessor.into())
    }

    fn broadcast(&self, shape: Shape) -> TCResult<Self::Broadcast> {
        let accessor = SparseBroadcast::new(self.accessor.clone(), shape)?;
        Ok(accessor.into())
    }

    fn expand_dims(&self, axis: usize) -> TCResult<Self::Expand> {
        let accessor = SparseExpand::new(self.accessor.clone(), axis)?;
        Ok(accessor.into())
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Self::Slice> {
        let accessor = self.accessor.clone().slice(bounds)?;
        Ok(accessor.into())
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let accessor = self.accessor.clone().transpose(permutation)?;
        Ok(accessor.into())
    }
}

impl<T: Clone + SparseAccess> Route for SparseTensor<T> {
    fn route(
        &'_ self,
        method: MethodType,
        path: &'_ [PathSegment],
    ) -> Option<Box<dyn Handler + '_>> {
        super::handlers::route(self, method, path)
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> Transact for SparseTensor<T> {
    async fn commit(&self, txn_id: &TxnId) {
        self.accessor.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.accessor.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.accessor.finalize(txn_id).await
    }
}

impl<T: Clone + SparseAccess> From<T> for SparseTensor<T> {
    fn from(accessor: T) -> Self {
        Self { accessor }
    }
}

impl<T: Clone + SparseAccess> From<SparseTensor<T>> for Collection {
    fn from(sparse: SparseTensor<T>) -> Collection {
        let accessor = sparse.into_inner().accessor();
        Collection::Tensor(Tensor::Sparse(SparseTensor { accessor }))
    }
}

pub async fn create(
    txn: &Txn,
    shape: Shape,
    dtype: NumberType,
) -> TCResult<SparseTensor<SparseTable>> {
    SparseTable::create(txn, shape, dtype)
        .map_ok(|accessor| SparseTensor { accessor })
        .await
}

pub fn from_dense<T: Clone + DenseAccess>(
    source: DenseTensor<T>,
) -> SparseTensor<DenseToSparse<T>> {
    let accessor = DenseToSparse::new(source.into_inner());
    SparseTensor { accessor }
}
