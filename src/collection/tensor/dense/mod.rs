use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt, TryStreamExt};

use crate::class::Instance;
use crate::collection::{from_dense, Collection};
use crate::error;
use crate::general::{TCBoxTryFuture, TCResult, TCTryStream};
use crate::handler::*;
use crate::scalar::number::*;
use crate::scalar::{MethodType, PathSegment};
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{Bounds, Shape};
use super::class::{Tensor, TensorInstance, TensorType};
use super::sparse::{DenseToSparse, SparseAccess, SparseAccessor, SparseTensor};
use super::stream::*;
use super::{
    Coord, IntoView, TensorAccess, TensorBoolean, TensorCompare, TensorDualIO, TensorIO,
    TensorMath, TensorReduce, TensorTransform, TensorUnary,
};

mod access;
mod array;
mod file;

pub use access::*;
pub use array::Array;
pub use file::*;

#[async_trait]
pub trait DenseAccess: ReadValueAt + TensorAccess + Transact + 'static {
    type Slice: Clone + DenseAccess;
    type Transpose: Clone + DenseAccess;

    fn accessor(self) -> DenseAccessor;

    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let dtype = self.dtype();
            let blocks = self
                .value_stream(txn)
                .await?
                .chunks(file::PER_BLOCK)
                .map(|values| values.into_iter().collect::<TCResult<Vec<Number>>>())
                .and_then(move |values| future::ready(Array::try_from_values(values, dtype)));

            let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let values = self
                .block_stream(txn)
                .await?
                .and_then(|array| future::ready(Ok(array.into_values())))
                .map_ok(|values| {
                    values
                        .into_iter()
                        .map(Ok)
                        .collect::<Vec<TCResult<Number>>>()
                })
                .map_ok(stream::iter)
                .try_flatten();

            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice>;

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose>;

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()>;

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()>;
}

#[derive(Clone)]
pub enum DenseAccessor {
    Broadcast(Box<BlockListBroadcast<DenseAccessor>>),
    Cast(Box<BlockListCast<DenseAccessor>>),
    Combine(Box<BlockListCombine<DenseAccessor, DenseAccessor>>),
    Expand(Box<BlockListExpand<DenseAccessor>>),
    File(BlockListFile),
    Reduce(Box<BlockListReduce<DenseAccessor>>),
    Slice(Box<BlockListSlice<DenseAccessor>>),
    Sparse(Box<BlockListSparse<SparseAccessor>>),
    Transpose(Box<BlockListTranspose<DenseAccessor>>),
    Unary(Box<BlockListUnary<DenseAccessor>>),
}

impl TensorAccess for DenseAccessor {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Broadcast(broadcast) => broadcast.dtype(),
            Self::Cast(cast) => cast.dtype(),
            Self::Combine(combine) => combine.dtype(),
            Self::Expand(expand) => expand.dtype(),
            Self::File(file) => file.dtype(),
            Self::Reduce(reduce) => reduce.dtype(),
            Self::Slice(slice) => slice.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
            Self::Transpose(transpose) => transpose.dtype(),
            Self::Unary(unary) => unary.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Broadcast(broadcast) => broadcast.ndim(),
            Self::Cast(cast) => cast.ndim(),
            Self::Combine(combine) => combine.ndim(),
            Self::Expand(expand) => expand.ndim(),
            Self::File(file) => file.ndim(),
            Self::Reduce(reduce) => reduce.ndim(),
            Self::Slice(slice) => slice.ndim(),
            Self::Sparse(sparse) => sparse.ndim(),
            Self::Transpose(transpose) => transpose.ndim(),
            Self::Unary(unary) => unary.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Broadcast(broadcast) => broadcast.shape(),
            Self::Cast(cast) => cast.shape(),
            Self::Combine(combine) => combine.shape(),
            Self::Expand(expand) => expand.shape(),
            Self::File(file) => file.shape(),
            Self::Reduce(reduce) => reduce.shape(),
            Self::Slice(slice) => slice.shape(),
            Self::Sparse(sparse) => sparse.shape(),
            Self::Transpose(transpose) => transpose.shape(),
            Self::Unary(unary) => unary.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Broadcast(broadcast) => broadcast.size(),
            Self::Cast(cast) => cast.size(),
            Self::Combine(combine) => combine.size(),
            Self::Expand(expand) => expand.size(),
            Self::File(file) => file.size(),
            Self::Reduce(reduce) => reduce.size(),
            Self::Slice(slice) => slice.size(),
            Self::Sparse(sparse) => sparse.size(),
            Self::Transpose(transpose) => transpose.size(),
            Self::Unary(unary) => unary.size(),
        }
    }
}

#[async_trait]
impl DenseAccess for DenseAccessor {
    type Slice = Self;
    type Transpose = Self;

    fn accessor(self) -> Self {
        self
    }

    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.block_stream(txn),
            Self::Cast(cast) => cast.block_stream(txn),
            Self::Combine(combine) => combine.block_stream(txn),
            Self::Expand(expand) => expand.block_stream(txn),
            Self::File(file) => file.block_stream(txn),
            Self::Reduce(reduce) => reduce.block_stream(txn),
            Self::Slice(slice) => slice.block_stream(txn),
            Self::Sparse(sparse) => sparse.block_stream(txn),
            Self::Transpose(transpose) => transpose.block_stream(txn),
            Self::Unary(unary) => unary.block_stream(txn),
        }
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.value_stream(txn),
            Self::Cast(cast) => cast.value_stream(txn),
            Self::Combine(combine) => combine.value_stream(txn),
            Self::Expand(expand) => expand.value_stream(txn),
            Self::File(file) => file.value_stream(txn),
            Self::Reduce(reduce) => reduce.value_stream(txn),
            Self::Slice(slice) => slice.value_stream(txn),
            Self::Sparse(sparse) => sparse.value_stream(txn),
            Self::Transpose(transpose) => transpose.value_stream(txn),
            Self::Unary(unary) => unary.value_stream(txn),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::Broadcast(broadcast) => broadcast.slice(bounds).map(DenseAccess::accessor),
            Self::Cast(cast) => cast.slice(bounds).map(DenseAccess::accessor),
            Self::Combine(combine) => combine.slice(bounds).map(DenseAccess::accessor),
            Self::Expand(expand) => expand.slice(bounds).map(DenseAccess::accessor),
            Self::File(file) => file.slice(bounds).map(DenseAccess::accessor),
            Self::Reduce(reduce) => reduce.slice(bounds).map(DenseAccess::accessor),
            Self::Slice(slice) => slice.slice(bounds).map(DenseAccess::accessor),
            Self::Sparse(sparse) => sparse.slice(bounds).map(DenseAccess::accessor),
            Self::Transpose(transpose) => transpose.slice(bounds).map(DenseAccess::accessor),
            Self::Unary(unary) => unary.slice(bounds).map(DenseAccess::accessor),
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        match self {
            Self::Broadcast(broadcast) => {
                broadcast.transpose(permutation).map(DenseAccess::accessor)
            }
            Self::Cast(cast) => cast.transpose(permutation).map(DenseAccess::accessor),
            Self::Combine(combine) => combine.transpose(permutation).map(DenseAccess::accessor),
            Self::Expand(expand) => expand.transpose(permutation).map(DenseAccess::accessor),
            Self::File(file) => file.transpose(permutation).map(DenseAccess::accessor),
            Self::Reduce(reduce) => reduce.transpose(permutation).map(DenseAccess::accessor),
            Self::Slice(slice) => slice.transpose(permutation).map(DenseAccess::accessor),
            Self::Sparse(sparse) => sparse.transpose(permutation).map(DenseAccess::accessor),
            Self::Transpose(transpose) => {
                transpose.transpose(permutation).map(DenseAccess::accessor)
            }
            Self::Unary(unary) => unary.transpose(permutation).map(DenseAccess::accessor),
        }
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        match self {
            Self::Broadcast(broadcast) => broadcast.write_value(txn_id, bounds, number).await,
            Self::Cast(cast) => cast.write_value(txn_id, bounds, number).await,
            Self::Combine(combine) => combine.write_value(txn_id, bounds, number).await,
            Self::Expand(expand) => expand.write_value(txn_id, bounds, number).await,
            Self::File(file) => file.write_value(txn_id, bounds, number).await,
            Self::Reduce(reduce) => reduce.write_value(txn_id, bounds, number).await,
            Self::Slice(slice) => slice.write_value(txn_id, bounds, number).await,
            Self::Sparse(sparse) => sparse.write_value(txn_id, bounds, number).await,
            Self::Transpose(transpose) => transpose.write_value(txn_id, bounds, number).await,
            Self::Unary(unary) => unary.write_value(txn_id, bounds, number).await,
        }
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        match self {
            Self::Broadcast(broadcast) => broadcast.write_value_at(txn_id, coord, value),
            Self::Cast(cast) => cast.write_value_at(txn_id, coord, value),
            Self::Combine(combine) => combine.write_value_at(txn_id, coord, value),
            Self::Expand(expand) => expand.write_value_at(txn_id, coord, value),
            Self::File(file) => file.write_value_at(txn_id, coord, value),
            Self::Reduce(reduce) => reduce.write_value_at(txn_id, coord, value),
            Self::Slice(slice) => slice.write_value_at(txn_id, coord, value),
            Self::Sparse(sparse) => sparse.write_value_at(txn_id, coord, value),
            Self::Transpose(transpose) => transpose.write_value_at(txn_id, coord, value),
            Self::Unary(unary) => unary.write_value_at(txn_id, coord, value),
        }
    }
}

impl ReadValueAt for DenseAccessor {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        match self {
            Self::Broadcast(broadcast) => broadcast.read_value_at(txn, coord),
            Self::Cast(cast) => cast.read_value_at(txn, coord),
            Self::Combine(combine) => combine.read_value_at(txn, coord),
            Self::Expand(expand) => expand.read_value_at(txn, coord),
            Self::File(file) => file.read_value_at(txn, coord),
            Self::Reduce(reduce) => reduce.read_value_at(txn, coord),
            Self::Slice(slice) => slice.read_value_at(txn, coord),
            Self::Sparse(sparse) => sparse.read_value_at(txn, coord),
            Self::Transpose(transpose) => transpose.read_value_at(txn, coord),
            Self::Unary(unary) => unary.read_value_at(txn, coord),
        }
    }
}

#[async_trait]
impl Transact for DenseAccessor {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Broadcast(broadcast) => broadcast.commit(txn_id).await,
            Self::Cast(cast) => cast.commit(txn_id).await,
            Self::Combine(combine) => combine.commit(txn_id).await,
            Self::Expand(expand) => expand.commit(txn_id).await,
            Self::File(file) => file.commit(txn_id).await,
            Self::Reduce(reduce) => reduce.commit(txn_id).await,
            Self::Slice(slice) => slice.commit(txn_id).await,
            Self::Sparse(sparse) => sparse.commit(txn_id).await,
            Self::Transpose(transpose) => transpose.commit(txn_id).await,
            Self::Unary(unary) => unary.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Broadcast(broadcast) => broadcast.rollback(txn_id).await,
            Self::Cast(cast) => cast.rollback(txn_id).await,
            Self::Combine(combine) => combine.rollback(txn_id).await,
            Self::Expand(expand) => expand.rollback(txn_id).await,
            Self::File(file) => file.rollback(txn_id).await,
            Self::Reduce(reduce) => reduce.rollback(txn_id).await,
            Self::Slice(slice) => slice.rollback(txn_id).await,
            Self::Sparse(sparse) => sparse.rollback(txn_id).await,
            Self::Transpose(transpose) => transpose.rollback(txn_id).await,
            Self::Unary(unary) => unary.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Broadcast(broadcast) => broadcast.finalize(txn_id).await,
            Self::Cast(cast) => cast.finalize(txn_id).await,
            Self::Combine(combine) => combine.finalize(txn_id).await,
            Self::Expand(expand) => expand.finalize(txn_id).await,
            Self::File(file) => file.finalize(txn_id).await,
            Self::Reduce(reduce) => reduce.finalize(txn_id).await,
            Self::Slice(slice) => slice.finalize(txn_id).await,
            Self::Sparse(sparse) => sparse.finalize(txn_id).await,
            Self::Transpose(transpose) => transpose.finalize(txn_id).await,
            Self::Unary(unary) => unary.finalize(txn_id).await,
        }
    }
}

impl From<BlockListFile> for DenseAccessor {
    fn from(file: BlockListFile) -> Self {
        Self::File(file)
    }
}

impl<T: Clone + SparseAccess> From<SparseTensor<T>> for DenseAccessor {
    fn from(sparse: SparseTensor<T>) -> Self {
        BlockListSparse::new(sparse).accessor()
    }
}

#[derive(Clone)]
pub struct DenseTensor<T: Clone + DenseAccess> {
    blocks: T,
}

impl<T: Clone + DenseAccess> DenseTensor<T> {
    pub fn into_inner(self) -> T {
        self.blocks
    }

    pub async fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCResult<TCTryStream<'a, Number>> {
        self.blocks.value_stream(txn).await
    }

    fn combine<OT: Clone + DenseAccess>(
        &self,
        other: &DenseTensor<OT>,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<DenseTensor<BlockListCombine<T, OT>>> {
        if self.shape() != other.shape() {
            return Err(error::unsupported(format!(
                "Cannot combine tensors with different shapes: {}, {}",
                self.shape(),
                other.shape()
            )));
        }

        let blocks = BlockListCombine::new(
            self.blocks.clone(),
            other.blocks.clone(),
            combinator,
            value_combinator,
            dtype,
        )?;

        Ok(DenseTensor { blocks })
    }
}

impl<T: Clone + DenseAccess> Instance for DenseTensor<T> {
    type Class = TensorType;

    fn class(&self) -> TensorType {
        TensorType::Dense
    }
}

impl<T: Clone + DenseAccess> TensorInstance for DenseTensor<T> {
    type Dense = Self;
    type Sparse = SparseTensor<DenseToSparse<T>>;

    fn into_dense(self) -> Self::Dense {
        self
    }

    fn into_sparse(self) -> Self::Sparse {
        from_dense(self)
    }
}

impl<T: Clone + DenseAccess> IntoView for DenseTensor<T> {
    fn into_view(self) -> Tensor {
        let blocks = self.into_inner().accessor();
        Tensor::Dense(DenseTensor { blocks })
    }
}

impl<T: Clone + DenseAccess> ReadValueAt for DenseTensor<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        self.blocks.read_value_at(txn, coord)
    }
}

impl<T: Clone + DenseAccess> TensorAccess for DenseTensor<T> {
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

#[async_trait]
impl<T: Clone + DenseAccess, OT: Clone + DenseAccess> TensorBoolean<DenseTensor<OT>>
    for DenseTensor<T>
{
    type Combine = DenseTensor<BlockListCombine<T, OT>>;

    fn and(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        self.combine(other, Array::and, Number::and, NumberType::Bool)
    }

    fn or(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        self.combine(other, Array::or, Number::or, NumberType::Bool)
    }

    fn xor(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        self.combine(other, Array::xor, Number::xor, NumberType::Bool)
    }
}

#[async_trait]
impl<T: Clone + DenseAccess> TensorUnary for DenseTensor<T> {
    type Unary = DenseTensor<BlockListUnary<T>>;

    fn abs(&self) -> TCResult<Self::Unary> {
        let blocks = BlockListUnary::new(
            self.blocks.clone(),
            Array::abs,
            <Number as NumberInstance>::abs,
            NumberType::Bool,
        );
        Ok(DenseTensor { blocks })
    }

    async fn all(&self, txn: &Txn) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(txn).await?;

        while let Some(array) = blocks.next().await {
            if !array?.all() {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn any(&self, txn: &Txn) -> TCResult<bool> {
        let mut blocks = self.blocks.block_stream(txn).await?;
        while let Some(array) = blocks.next().await {
            if array?.any() {
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
        Ok(DenseTensor { blocks })
    }
}

#[async_trait]
impl<T: Clone + DenseAccess, OT: Clone + DenseAccess> TensorCompare<DenseTensor<OT>>
    for DenseTensor<T>
{
    type Compare = DenseTensor<BlockListCombine<T, OT>>;
    type Dense = DenseTensor<BlockListCombine<T, OT>>;

    async fn eq(&self, other: &DenseTensor<OT>, _txn: &Txn) -> TCResult<Self::Dense> {
        self.combine(
            other,
            Array::eq,
            <Number as NumberInstance>::eq,
            NumberType::Bool,
        )
    }

    fn gt(&self, other: &DenseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(
            other,
            Array::gt,
            <Number as NumberInstance>::gt,
            NumberType::Bool,
        )
    }

    async fn gte(&self, other: &DenseTensor<OT>, _txn: &Txn) -> TCResult<Self::Dense> {
        self.combine(
            other,
            Array::gte,
            <Number as NumberInstance>::gte,
            NumberType::Bool,
        )
    }

    fn lt(&self, other: &DenseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(
            other,
            Array::lt,
            <Number as NumberInstance>::lt,
            NumberType::Bool,
        )
    }

    async fn lte(&self, other: &DenseTensor<OT>, _txn: &Txn) -> TCResult<Self::Dense> {
        self.combine(
            other,
            Array::lte,
            <Number as NumberInstance>::lte,
            NumberType::Bool,
        )
    }

    fn ne(&self, other: &DenseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(
            other,
            Array::ne,
            <Number as NumberInstance>::ne,
            NumberType::Bool,
        )
    }
}

impl<T: Clone + DenseAccess, OT: Clone + DenseAccess> TensorMath<DenseTensor<OT>>
    for DenseTensor<T>
{
    type Combine = DenseTensor<BlockListCombine<T, OT>>;

    fn add(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, Array::add, <Number as NumberInstance>::add, dtype)
    }

    fn multiply(&self, other: &DenseTensor<OT>) -> TCResult<Self::Combine> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(
            other,
            Array::multiply,
            <Number as NumberInstance>::multiply,
            dtype,
        )
    }
}

#[async_trait]
impl<T: Clone + DenseAccess> TensorIO for DenseTensor<T> {
    async fn read_value(&self, txn: &Txn, coord: Coord) -> TCResult<Number> {
        self.blocks
            .read_value_at(txn, coord.to_vec())
            .map_ok(|(_, val)| val)
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        self.blocks.clone().write_value(txn_id, bounds, value).await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.blocks.write_value_at(txn_id, coord, value).await
    }
}

#[async_trait]
impl<T: Clone + DenseAccess, OT: Clone + DenseAccess> TensorDualIO<DenseTensor<OT>>
    for DenseTensor<T>
{
    async fn mask(&self, _txn: &Txn, _other: DenseTensor<OT>) -> TCResult<()> {
        Err(error::not_implemented("DenseTensor::mask"))
    }

    async fn write(&self, txn: &Txn, bounds: Bounds, other: DenseTensor<OT>) -> TCResult<()> {
        let slice = self.slice(bounds)?;
        let other = other
            .broadcast(slice.shape().clone())?
            .as_type(self.dtype())?;

        let coords = stream::iter(Bounds::all(slice.shape()).affected().map(TCResult::Ok));
        let values: TCTryStream<(Coord, Number)> =
            Box::pin(ValueReader::new(coords, txn, &other.blocks));

        values
            .map_ok(|(coord, value)| slice.write_value_at(*txn.id(), coord, value))
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(())
    }
}

#[async_trait]
impl<T: Clone + DenseAccess> TensorDualIO<Tensor> for DenseTensor<T> {
    async fn mask(&self, txn: &Txn, other: Tensor) -> TCResult<()> {
        match other {
            Tensor::Sparse(sparse) => self.mask(txn, sparse.into_dense()).await,
            Tensor::Dense(dense) => self.mask(txn, dense).await,
        }
    }

    async fn write(&self, txn: &Txn, bounds: Bounds, other: Tensor) -> TCResult<()> {
        match other {
            Tensor::Sparse(sparse) => self.write(txn, bounds, sparse.into_dense()).await,
            Tensor::Dense(dense) => self.write(txn, bounds, dense).await,
        }
    }
}

impl<T: Clone + DenseAccess> TensorReduce for DenseTensor<T> {
    type Reduce = DenseTensor<BlockListReduce<T>>;

    fn product(&self, _axis: usize) -> TCResult<Self::Reduce> {
        Err(error::not_implemented("DenseTensor::product"))
    }

    fn product_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let blocks = self.blocks.block_stream(&txn).await?;

            let mut block_products = blocks.map_ok(|array| array.product());

            let zero = self.dtype().zero();
            let mut product = self.dtype().one();
            while let Some(block_product) = block_products.try_next().await? {
                if block_product == zero {
                    return Ok(zero);
                }

                product = product * block_product;
            }

            Ok(product)
        })
    }

    fn sum(&self, _axis: usize) -> TCResult<Self::Reduce> {
        Err(error::not_implemented("DenseTensor::sum"))
    }

    fn sum_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            let blocks = self.blocks.block_stream(&txn).await?;

            blocks
                .map_ok(|array| array.sum())
                .try_fold(self.dtype().zero(), |sum, block_sum| {
                    future::ready(Ok(sum + block_sum))
                })
                .await
        })
    }
}

impl<T: Clone + DenseAccess> TensorTransform for DenseTensor<T> {
    type Cast = DenseTensor<BlockListCast<T>>;
    type Broadcast = DenseTensor<BlockListBroadcast<T>>;
    type Expand = DenseTensor<BlockListExpand<T>>;
    type Slice = DenseTensor<BlockListSlice<T>>;
    type Transpose = DenseTensor<BlockListTranspose<T>>;

    fn as_type(&self, dtype: NumberType) -> TCResult<Self::Cast> {
        let blocks = BlockListCast::new(self.blocks.clone(), dtype);
        Ok(DenseTensor { blocks })
    }

    fn broadcast(&self, shape: Shape) -> TCResult<Self::Broadcast> {
        let blocks = BlockListBroadcast::new(self.blocks.clone(), shape)?;
        Ok(DenseTensor { blocks })
    }

    fn expand_dims(&self, axis: usize) -> TCResult<Self::Expand> {
        let blocks = BlockListExpand::new(self.blocks.clone(), axis)?;
        Ok(DenseTensor { blocks })
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Self::Slice> {
        let blocks = BlockListSlice::new(self.blocks.clone(), bounds)?;
        Ok(DenseTensor { blocks })
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let blocks = BlockListTranspose::new(self.blocks.clone(), permutation)?;
        Ok(DenseTensor { blocks })
    }
}

impl<T: Clone + DenseAccess> Route for DenseTensor<T> {
    fn route(
        &'_ self,
        method: MethodType,
        path: &'_ [PathSegment],
    ) -> Option<Box<dyn Handler + '_>> {
        super::handlers::route(self, method, path)
    }
}

#[async_trait]
impl<T: Clone + DenseAccess> Transact for DenseTensor<T> {
    async fn commit(&self, txn_id: &TxnId) {
        self.blocks.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.blocks.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.blocks.finalize(txn_id).await
    }
}

impl<T: Clone + DenseAccess> From<T> for DenseTensor<T> {
    fn from(blocks: T) -> DenseTensor<T> {
        DenseTensor { blocks }
    }
}

impl<T: Clone + DenseAccess> From<DenseTensor<T>> for Collection {
    fn from(dense: DenseTensor<T>) -> Collection {
        let blocks = dense.into_inner().accessor();
        Collection::Tensor(Tensor::Dense(blocks.into()))
    }
}

pub async fn dense_constant(
    txn: &Txn,
    shape: Shape,
    value: Number,
) -> TCResult<DenseTensor<BlockListFile>> {
    let blocks = BlockListFile::constant(txn, shape, value).await?;
    Ok(DenseTensor { blocks })
}

pub fn from_sparse<T: Clone + SparseAccess>(
    sparse: SparseTensor<T>,
) -> DenseTensor<BlockListSparse<T>> {
    let blocks = BlockListSparse::new(sparse);
    DenseTensor { blocks }
}
