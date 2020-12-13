use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};

use crate::class::{Instance, TCBoxTryFuture, TCResult, TCTryStream};
use crate::collection::schema::{Column, IndexSchema};
use crate::collection::table::{self, ColumnBound, Table, TableIndex, TableInstance};
use crate::error;
use crate::scalar::value::number::*;
use crate::scalar::{label, Bound, Id, Label, Value, ValueType};
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{AxisBounds, Bounds, Shape};
use super::class::{TensorInstance, TensorType, TensorViewType};
use super::dense::{dense_constant, from_sparse, BlockList, BlockListFile, DenseTensor};
use super::transform;
use super::{
    broadcast, IntoView, TensorAccessor, TensorBoolean, TensorCompare, TensorDualIO, TensorIO,
    TensorMath, TensorReduce, TensorTransform, TensorUnary, TensorView, ERR_NONBIJECTIVE_WRITE,
};

mod access;
mod combine;

pub use access::*;
use combine::SparseCombine;

const VALUE: Label = label("value");

pub type SparseRow = (Vec<u64>, Number);
pub type SparseStream = TCTryStream<SparseRow>;

const ERR_NOT_SPARSE: &str = "The result of the requested operation would not be sparse;\
convert to a DenseTensor first.";

const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

#[derive(Clone)]
pub struct SparseTable {
    table: TableIndex,
    shape: Shape,
    dtype: NumberType,
}

impl SparseTable {
    pub async fn constant(txn: &Txn, shape: Shape, value: Number) -> TCResult<SparseTable> {
        let bounds = Bounds::all(&shape);
        let table = Self::create(txn, shape, value.class()).await?;

        if value != value.class().zero() {
            stream::iter(bounds.affected())
                .map(|coord| Ok(table.write_value(txn.id().clone(), coord, value.clone())))
                .try_buffer_unordered(2usize)
                .try_fold((), |(), ()| future::ready(Ok(())))
                .await?;
        }

        Ok(table)
    }

    pub async fn create(txn: &Txn, shape: Shape, dtype: NumberType) -> TCResult<Self> {
        let table = Self::create_table(txn, shape.len(), dtype).await?;
        Ok(Self {
            table,
            dtype,
            shape,
        })
    }

    pub async fn create_table(txn: &Txn, ndim: usize, dtype: NumberType) -> TCResult<TableIndex> {
        let key: Vec<Column> = Self::key(ndim);
        let value: Vec<Column> = vec![(VALUE.into(), ValueType::Number(dtype)).into()];
        let indices = (0..ndim).map(|axis| (axis.into(), vec![axis.into()]));
        Table::create(txn, (IndexSchema::from((key, value)), indices).into()).await
    }

    pub async fn from_values<S: Stream<Item = Number>>(
        txn: &Txn,
        shape: Shape,
        dtype: NumberType,
        values: S,
    ) -> TCResult<Self> {
        let zero = dtype.zero();
        let bounds = Bounds::all(&shape);

        let table = Self::create(txn, shape, dtype).await?;

        let txn_id = *txn.id();
        stream::iter(bounds.affected())
            .zip(values)
            .filter(|(_, value)| future::ready(value != &zero))
            .map(|(coord, value)| Ok(table.write_value(txn_id, coord, value)))
            .try_buffer_unordered(2usize)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(table)
    }

    pub fn try_from_table(table: TableIndex, shape: Shape) -> TCResult<SparseTable> {
        let expected_key = Self::key(shape.len());
        let actual_key = table.key();

        for (expected, actual) in actual_key.iter().zip(expected_key.iter()) {
            if expected != actual {
                let key: Vec<String> = table.key().iter().map(|c| c.to_string()).collect();
                return Err(error::bad_request(
                    "Table has invalid key for SparseTable",
                    format!("[{}]", key.join(", ")),
                ));
            }
        }

        let actual_value = table.values();
        if actual_value.len() != 1 {
            let actual_value: Vec<String> = actual_value.iter().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Table has invalid value for SparseTable",
                format!("[{}]", actual_value.join(", ")),
            ));
        }

        let dtype = actual_value[0].dtype();
        if let ValueType::Number(dtype) = dtype {
            Ok(SparseTable {
                table,
                shape,
                dtype,
            })
        } else {
            Err(error::bad_request(
                "Table has invalid data type for SparseTable",
                dtype,
            ))
        }
    }

    fn key(ndim: usize) -> Vec<Column> {
        let u64_type = NumberType::uint64();
        (0..ndim).map(|axis| (axis, u64_type).into()).collect()
    }
}

impl TensorAccessor for SparseTable {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.shape.size()
    }
}

#[async_trait]
impl SparseAccess for SparseTable {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let rows = self.table.clone().stream(txn.id().clone()).await?;
        let filled = rows.map(unwrap_row);
        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        let columns: Vec<Id> = axes.iter().map(|x| (*x).into()).collect();
        let filled_at = self
            .table
            .group_by(columns.to_vec())?
            .stream(txn.id().clone())
            .await?
            .map(|coord| unwrap_coord(&coord));

        let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
        Ok(filled_at)
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.table.count(txn.id().clone()).await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let source = slice_table(self.table.clone().into(), &bounds).await?;
        let filled_in = source.stream(txn.id().clone()).await?.map(unwrap_row);
        let filled_in: SparseStream = Box::pin(filled_in);
        Ok(filled_in)
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        if !self.shape().contains_coord(coord) {
            return Err(error::bad_request(
                "Coordinate out of bounds",
                Bounds::from(coord),
            ));
        }

        let selector: HashMap<Id, ColumnBound> = coord
            .iter()
            .enumerate()
            .map(|(axis, at)| (axis.into(), u64_to_value(*at).into()))
            .collect();

        let mut slice = self
            .table
            .slice(selector.into())?
            .select(vec![VALUE.into()])?
            .stream(txn.id().clone())
            .await?;

        match slice.next().await {
            Some(mut number) if number.len() == 1 => number.pop().unwrap().try_into(),
            None => Ok(self.dtype().zero()),
            _ => Err(error::internal(ERR_CORRUPT)),
        }
    }

    async fn write_value(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        let value = value.into_type(self.dtype);

        let key = coord
            .into_iter()
            .map(Number::from)
            .map(Value::Number)
            .collect();

        self.table
            .upsert(&txn_id, key, vec![Value::Number(value)])
            .await
    }
}

#[async_trait]
impl Transact for SparseTable {
    async fn commit(&self, txn_id: &TxnId) {
        self.table.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.table.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.table.finalize(txn_id).await
    }
}

#[derive(Clone)]
pub struct SparseUnary {
    source: Arc<dyn SparseAccess>,
    transform: fn(Number) -> Number,
    dtype: NumberType,
}

impl TensorAccessor for SparseUnary {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl SparseAccess for SparseUnary {
    async fn filled(self: Arc<Self>, txn: Txn) -> TCResult<SparseStream> {
        let transform = self.transform;
        let filled = self.source.clone().filled(txn).await?;
        let cast = filled.map_ok(move |(coord, value)| (coord, transform(value)));
        let cast: SparseStream = Box::pin(cast);
        Ok(cast)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCResult<TCTryStream<Vec<u64>>> {
        self.source.clone().filled_at(txn, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn: Txn) -> TCResult<u64> {
        self.source.clone().filled_count(txn).await
    }

    async fn filled_in(self: Arc<Self>, txn: Txn, bounds: Bounds) -> TCResult<SparseStream> {
        let transform = self.transform;
        let source = self.source.clone().filled_in(txn, bounds).await?;
        let filled_in = source.map_ok(move |(coord, value)| (coord, transform(value)));
        let filled_in: SparseStream = Box::pin(filled_in);
        Ok(filled_in)
    }

    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        let dtype = self.dtype;
        self.source
            .read_value(txn, coord)
            .map_ok(move |value| value.into_type(dtype))
            .await
    }

    async fn write_value(&self, _txn_id: TxnId, _coord: Vec<u64>, _value: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

#[async_trait]
impl Transact for SparseUnary {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}

#[derive(Clone)]
pub struct SparseTensor<T: Clone + SparseAccess> {
    accessor: Arc<T>,
}

impl<T: Clone + SparseAccess> SparseTensor<T> {
    fn into_dyn(self) -> SparseTensor<SparseAccessorDyn> {
        let accessor = Arc::new(SparseAccessorDyn::new(self.clone_into()));
        SparseTensor { accessor }
    }

    pub fn clone_into(&self) -> T {
        (*self.accessor).clone()
    }

    pub async fn copy(&self, txn: Txn) -> TCResult<SparseTensor<SparseTable>> {
        self.accessor
            .clone()
            .copy(txn)
            .await
            .map(Arc::new)
            .map(|accessor| SparseTensor { accessor })
    }

    pub fn filled(&'_ self, txn: Txn) -> TCBoxTryFuture<'_, SparseStream> {
        self.accessor.clone().filled(txn)
    }

    fn filled_at(
        &'_ self,
        txn: Txn,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'_, TCTryStream<Vec<u64>>> {
        self.accessor.clone().filled_at(txn, axes)
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
        )
        .map(Arc::new)?;

        Ok(SparseTensor { accessor })
    }

    fn condense<'a, OT: Clone + SparseAccess>(
        &'a self,
        other: &'a SparseTensor<OT>,
        txn: Txn,
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
            )
            .map(Arc::new)?;

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

impl<T: Clone + SparseAccess> Instance for SparseTensor<T> {
    type Class = TensorType;

    fn class(&self) -> TensorType {
        TensorType::View(TensorViewType::Sparse)
    }
}

impl<T: Clone + SparseAccess> TensorInstance for SparseTensor<T> {}

impl<T: Clone + SparseAccess> IntoView for SparseTensor<T> {
    fn into_view(self) -> TensorView {
        let accessor = Arc::new(SparseAccessorDyn::new(self.clone_into()));
        TensorView::Sparse(SparseTensor { accessor })
    }
}

impl<T: Clone + SparseAccess> TensorAccessor for SparseTensor<T> {
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
        let accessor = Arc::new(SparseUnary {
            source: self.accessor.clone(),
            transform: <Number as NumberInstance>::abs,
            dtype: NumberType::Bool,
        });

        Ok(SparseTensor { accessor })
    }

    async fn all(&self, txn: Txn) -> TCResult<bool> {
        let mut coords = self
            .accessor
            .clone()
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

    async fn any(&self, txn: Txn) -> TCResult<bool> {
        let mut filled = self.accessor.clone().filled(txn).await?;
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

    async fn eq(&self, other: &SparseTensor<OT>, txn: Txn) -> TCResult<Self::Dense> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::eq)
            .await
    }

    fn gt(&self, other: &SparseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(other, <Number as NumberInstance>::gt, NumberType::Bool)
    }

    async fn gte(&self, other: &SparseTensor<OT>, txn: Txn) -> TCResult<Self::Dense> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::gte)
            .await
    }

    fn lt(&self, other: &SparseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(other, <Number as NumberInstance>::lt, NumberType::Bool)
    }

    async fn lte(&self, other: &SparseTensor<OT>, txn: Txn) -> TCResult<Self::Dense> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::lte)
            .await
    }

    fn ne(&self, other: &SparseTensor<OT>) -> TCResult<Self::Compare> {
        self.combine(other, <Number as NumberInstance>::ne, NumberType::Bool)
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> TensorIO for SparseTensor<T> {
    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        self.accessor.read_value(txn, coord).await
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

    async fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
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
            .filled(txn.clone())
            .await?
            .map_ok(|(coord, _)| self.write_value_at(txn_id, coord, zero.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn write(&self, txn: Txn, bounds: Bounds, other: SparseTensor<OT>) -> TCResult<()> {
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
        let accessor =
            SparseReduce::new(self.clone(), axis, SparseTensor::product_all).map(Arc::new)?;
        Ok(SparseTensor { accessor })
    }

    fn product_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            if self.all(txn.clone()).await? {
                from_sparse(self.clone()).product_all(txn).await
            } else {
                Ok(self.dtype().zero())
            }
        })
    }

    fn sum(&self, axis: usize) -> TCResult<Self::Reduce> {
        let accessor =
            SparseReduce::new(self.clone(), axis, SparseTensor::sum_all).map(Arc::new)?;
        Ok(SparseTensor { accessor })
    }

    fn sum_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        Box::pin(async move {
            if self.any(txn.clone()).await? {
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
    type Slice = SparseTensor<SparseSlice>;
    type Reshape = SparseTensor<SparseReshape>;
    type Transpose = SparseTensor<SparseTranspose<T>>;

    fn as_type(&self, dtype: NumberType) -> TCResult<Self::Cast> {
        let source = self.accessor.clone();
        let accessor = Arc::new(SparseCast::new(source, dtype));

        Ok(accessor.into())
    }

    fn broadcast(&self, shape: Shape) -> TCResult<Self::Broadcast> {
        let rebase = transform::Broadcast::new(self.shape().clone(), shape)?;
        let accessor = Arc::new(SparseBroadcast::new(self.accessor.clone(), rebase));

        Ok(accessor.into())
    }

    fn expand_dims(&self, axis: usize) -> TCResult<Self::Expand> {
        let rebase = transform::Expand::new(self.shape().clone(), axis)?;
        let accessor = Arc::new(SparseExpand::new(self.accessor.clone(), rebase));

        Ok(accessor.into())
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Self::Slice> {
        let rebase = transform::Slice::new(self.shape().clone(), bounds)?;
        let accessor = Arc::new(SparseSlice::new(self.accessor.clone(), rebase));

        Ok(accessor.into())
    }

    fn reshape(&self, shape: Shape) -> TCResult<Self::Reshape> {
        let rebase = transform::Reshape::new(self.shape().clone(), shape)?;
        let accessor = Arc::new(SparseReshape::new(self.accessor.clone(), rebase));

        Ok(accessor.into())
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let rebase = transform::Transpose::new(self.shape().clone(), permutation)?;
        let accessor = Arc::new(SparseTranspose::new(self.accessor.clone(), rebase));

        Ok(accessor.into())
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
        let accessor = Arc::new(accessor);
        accessor.into()
    }
}

impl<T: Clone + SparseAccess> From<Arc<T>> for SparseTensor<T> {
    fn from(accessor: Arc<T>) -> Self {
        Self { accessor }
    }
}

pub fn from_dense<T: Clone + BlockList>(source: DenseTensor<T>) -> SparseTensor<DenseAccessor<T>> {
    let accessor = Arc::new(DenseAccessor::new(source));
    SparseTensor { accessor }
}

async fn slice_table(mut table: Table, bounds: &'_ Bounds) -> TCResult<Table> {
    use AxisBounds::*;

    for (axis, axis_bound) in bounds.to_vec().into_iter().enumerate() {
        let axis: Id = axis.into();
        let column_bound = match axis_bound {
            At(x) => table::ColumnBound::Is(u64_to_value(x)),
            In(range) => {
                let start = Bound::In(u64_to_value(range.start));
                let end = Bound::Ex(u64_to_value(range.end));
                (start, end).into()
            }
            _ => todo!(),
        };

        let bounds: HashMap<Id, ColumnBound> = iter::once((axis, column_bound)).collect();
        table = table.slice(bounds.into())?
    }

    Ok(table)
}

fn u64_to_value(u: u64) -> Value {
    Value::Number(Number::UInt(UInt::U64(u)))
}

fn unwrap_coord(coord: &[Value]) -> TCResult<Vec<u64>> {
    coord.iter().map(|val| unwrap_u64(val)).collect()
}

fn unwrap_row(mut row: Vec<Value>) -> TCResult<(Vec<u64>, Number)> {
    let coord = unwrap_coord(&row[0..row.len() - 1])?;
    if let Some(value) = row.pop() {
        Ok((coord, value.try_into()?))
    } else {
        Err(error::internal(ERR_CORRUPT))
    }
}

fn unwrap_u64(value: &Value) -> TCResult<u64> {
    if let Value::Number(Number::UInt(UInt::U64(unwrapped))) = value {
        Ok(*unwrapped)
    } else {
        Err(error::bad_request("Expected u64 but found", value))
    }
}
