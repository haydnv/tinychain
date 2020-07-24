use std::collections::HashMap;
use std::convert::TryInto;
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, BoxFuture, Future, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};

use crate::error;
use crate::state::table::schema::*;
use crate::state::table::{Selection, Table, TableBase};
use crate::transaction::{Txn, TxnId};
use crate::value::class::{NumberClass, NumberImpl, NumberType, UIntType, ValueType};
use crate::value::{Number, TCResult, TCStream, UInt, Value, ValueId};

use super::base::*;
use super::bounds::{AxisBounds, Bounds, Shape};
use super::dense::{BlockTensor, DenseTensorView};
use super::stream::ValueBlockStream;
use super::TensorView;

const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

#[async_trait]
pub trait SparseTensorView: TensorView + 'static {
    async fn filter_map<
        O: SparseTensorView,
        F: Fn(Vec<u64>, Number, Number) -> Option<Vec<u64>> + Send + Sync,
    >(
        this: Self,
        that: O,
        txn: Arc<Txn>,
        filter: F,
        value: Number,
    ) -> TCResult<SparseTensor> {
        if this.shape() != that.shape() {
            return Err(error::bad_request(
                "Cannot compare tensor with shape",
                that.shape(),
            ));
        }

        let txn_id = txn.id().clone();
        let txn_id_clone = txn_id.clone();
        let per_block = super::dense::per_block(NumberType::Bool);
        let sparse = SparseTensor::create(txn, this.shape().clone(), NumberType::Bool).await?;

        let that_clone = that.clone();
        let these_values = this
            .clone()
            .filled(txn_id.clone())
            .await?
            .map(|(coord, left)| {
                that.read_value(txn_id.clone(), coord.to_vec())
                    .and_then(move |right| future::ready(Ok((coord, left, right))))
            })
            .buffer_unordered(per_block)
            .try_filter_map(|(coord, left, right)| future::ready(Ok(filter(coord, left, right))));

        let those_values = that_clone
            .filled(txn_id.clone())
            .await?
            .map(|(coord, right)| {
                this.read_value(txn_id.clone(), coord.to_vec())
                    .and_then(move |left| future::ready(Ok((coord, left, right))))
            })
            .buffer_unordered(per_block)
            .try_filter_map(|(coord, left, right)| future::ready(Ok(filter(coord, left, right))));

        these_values
            .chain(those_values)
            .map_ok(|coord| sparse.write_value(txn_id_clone.clone(), coord, value.clone()))
            .try_buffer_unordered(per_block)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(sparse)
    }

    async fn filter_map_dense<
        O: SparseTensorView,
        F: Fn(Vec<u64>, Number, Number) -> Option<Vec<u64>> + Send + Sync,
    >(
        this: Self,
        that: O,
        txn: Arc<Txn>,
        filter: F,
        value: Number,
    ) -> TCResult<BlockTensor>
    where
        Self: Slice,
        <Self as Slice>::Slice: SparseTensorView + Slice,
    {
        if this.shape() != that.shape() {
            return Err(error::bad_request(
                "Cannot compare tensor with shape",
                that.shape(),
            ));
        }

        let txn_id = txn.id().clone();
        let shape = this.shape().clone();
        let per_block = super::dense::per_block(NumberType::Bool);

        let blocks = BlockTensor::sparse_into_blocks(txn.id().clone(), this.clone())
            .and_then(|block| future::ready(block.into_type(NumberType::Bool)))
            .map_ok(|block| block.not());
        let dense =
            BlockTensor::from_blocks(txn, shape, NumberType::Bool, Box::pin(blocks)).await?;

        that.filled(txn_id.clone())
            .await?
            .map(|(coord, right)| {
                this.read_value(txn_id.clone(), coord.to_vec())
                    .and_then(|left| future::ready(Ok((coord, left, right))))
            })
            .buffer_unordered(per_block)
            .try_filter_map(|(coord, left, right)| future::ready(Ok(filter(coord, left, right))))
            .map_ok(|coord| {
                dense
                    .clone()
                    .write_value(txn_id.clone(), coord.into(), value.clone())
            })
            .try_buffer_unordered(per_block)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(dense)
    }

    async fn filled(self, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>>;

    async fn filled_at(&self, txn: Arc<Txn>, axes: &[usize]) -> TCResult<TCStream<Vec<u64>>>;

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64>;

    fn read_value<'a>(&'a self, txn_id: TxnId, coord: Vec<u64>) -> BoxFuture<'a, TCResult<Number>>;

    async fn write_sparse<T: SparseTensorView>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        value: T,
    ) -> TCResult<()>
    where
        Self: Slice,
        <Self as Slice>::Slice: SparseTensorView,
    {
        let dest = self.slice(bounds)?;
        value
            .filled(txn_id.clone())
            .await?
            .map(|(coord, number)| dest.write_value(txn_id.clone(), coord, number))
            .buffer_unordered(dest.size() as usize)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> BoxFuture<'a, TCResult<()>>;
}

#[derive(Clone)]
pub struct SparseTensor {
    dtype: NumberType,
    shape: Shape,
    table: TableBase,
}

impl SparseTensor {
    pub async fn create(txn: Arc<Txn>, shape: Shape, dtype: NumberType) -> TCResult<SparseTensor> {
        let u64_type = ValueType::Number(NumberType::UInt(UIntType::U64));
        let key: Vec<Column> = (0..shape.len())
            .map(|axis| Column {
                name: axis.into(),
                dtype: u64_type,
                max_len: None,
            })
            .collect();

        let value = vec![Column {
            name: "value".parse()?,
            dtype: dtype.into(),
            max_len: None,
        }];

        let schema = Schema::new(key, value);
        let table = TableBase::create(txn, schema).await?;

        Ok(SparseTensor {
            dtype,
            shape,
            table,
        })
    }

    async fn slice_table(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<Table> {
        let mut table: Table = self.table.clone().into();
        use AxisBounds::*;
        for (axis, axis_bound) in bounds.to_vec().into_iter().enumerate() {
            let axis: ValueId = axis.into();
            table = match axis_bound {
                At(x) => {
                    let column_bound = ColumnBound::Is(u64_to_value(x));
                    table
                        .slice(txn_id, vec![(axis, column_bound)].into())
                        .await?
                }
                In(range, 1) => {
                    let start = Bound::Included(u64_to_value(range.start));
                    let end = Bound::Excluded(u64_to_value(range.end));
                    let column_bound = ColumnBound::In(start, end);
                    table
                        .slice(txn_id, vec![(axis, column_bound)].into())
                        .await?
                }
                _ => unimplemented!(),
            };
        }

        Ok(table)
    }
}

impl TensorView for SparseTensor {
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
impl SparseTensorView for SparseTensor {
    async fn filled(self, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let filled = self.table.stream(txn_id).await?.map(unwrap_row);

        Ok(Box::pin(filled))
    }

    async fn filled_at(&self, txn: Arc<Txn>, axes: &[usize]) -> TCResult<TCStream<Vec<u64>>> {
        let txn_id = txn.id().clone();
        let columns: Vec<ValueId> = axes.iter().map(|x| (*x).into()).collect();
        Ok(Box::pin(
            self.table
                .group_by(txn, columns)
                .await?
                .stream(txn_id)
                .await?
                .map(|coord| unwrap_coord(&coord)),
        ))
    }

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.table.count(txn_id).await
    }

    fn read_value<'a>(&'a self, txn_id: TxnId, coord: Vec<u64>) -> BoxFuture<'a, TCResult<Number>> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(error::bad_request(
                    "Coordinate out of bounds",
                    Bounds::from(coord),
                ));
            }

            let selector: HashMap<ValueId, Value> = coord
                .iter()
                .enumerate()
                .map(|(axis, at)| (axis.into(), u64_to_value(*at)))
                .collect();

            let mut slice = self
                .table
                .slice(&txn_id, selector.into())
                .await?
                .select(vec!["value".parse()?])?
                .stream(txn_id)
                .await?;

            match slice.next().await {
                Some(mut number) if number.len() == 1 => number.pop().unwrap().try_into(),
                None => Ok(self.dtype().zero()),
                _ => Err(error::internal(ERR_CORRUPT)),
            }
        })
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        mut coord: Vec<u64>,
        value: Number,
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let mut row: HashMap<ValueId, Value> = coord
                .drain(..)
                .enumerate()
                .map(|(x, v)| (x.into(), Value::Number(Number::UInt(UInt::U64(v)))))
                .collect();

            row.insert("value".parse()?, value.into());
            self.table.upsert(&txn_id, row).await
        })
    }
}

impl Slice for SparseTensor {
    type Slice = TensorSlice<SparseTensor>;

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        TensorSlice::new(self, bounds)
    }
}

#[async_trait]
impl SparseTensorView for TensorSlice<SparseTensor> {
    async fn filled(self, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let stream = self
            .source()
            .clone()
            .slice_table(&txn_id, self.bounds())
            .await?
            .stream(txn_id)
            .await?
            .map(unwrap_row)
            .map(move |(coord, value)| (self.map_bounds(coord.into()).into_coord(), value));

        Ok(Box::pin(stream))
    }

    async fn filled_at(&self, txn: Arc<Txn>, axes: &[usize]) -> TCResult<TCStream<Vec<u64>>> {
        let table = self
            .source()
            .clone()
            .slice_table(txn.id(), self.bounds())
            .await?;
        let txn_id = txn.id().clone();
        let columns: Vec<ValueId> = axes.iter().map(|x| (*x).into()).collect();
        Ok(Box::pin(
            table
                .group_by(txn, columns)
                .await?
                .stream(txn_id)
                .await?
                .map(|coord| unwrap_coord(&coord)),
        ))
    }

    fn read_value<'a>(&'a self, txn_id: TxnId, coord: Vec<u64>) -> BoxFuture<'a, TCResult<Number>> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(error::bad_request(
                    "Coordinate out of bounds",
                    Bounds::from(coord),
                ));
            }

            let source_coord = self.invert_bounds(coord.into()).into_coord();
            self.source().read_value(txn_id, source_coord).await
        })
    }

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.source()
            .clone()
            .slice_table(&txn_id, self.bounds())
            .await?
            .count(txn_id)
            .await
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> BoxFuture<'a, TCResult<()>> {
        self.source()
            .write_value(txn_id, self.invert_bounds(coord.into()).into_coord(), value)
    }
}

#[async_trait]
impl<T: SparseTensorView> AnyAll for T {
    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        let mut values = self.filled(txn_id).await?;
        Ok(values.next().await.is_some())
    }

    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        self.filled_count(txn_id)
            .await
            .map(|count| count == self.size())
    }
}

#[async_trait]
impl<T: SparseTensorView + Slice> SparseTensorUnary for T
where
    <T as Slice>::Slice: SparseTensorUnary,
{
    async fn as_dtype(self, txn: Arc<Txn>, dtype: NumberType) -> TCResult<SparseTensor> {
        let txn_id = txn.id().clone();
        let cast = SparseTensor::create(txn, self.shape().clone(), dtype).await?;
        self.filled(txn_id.clone())
            .await?
            .map(|(coord, value)| cast.write_value(txn_id.clone(), coord, value.into_type(dtype)))
            .buffer_unordered(super::dense::per_block(dtype))
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(cast)
    }

    async fn copy(self, txn: Arc<Txn>) -> TCResult<SparseTensor> {
        let dtype = self.dtype();
        let txn_id = txn.id().clone();
        let copy = SparseTensor::create(txn, self.shape().clone(), dtype).await?;
        self.filled(txn_id.clone())
            .await?
            .map(|(coord, value)| copy.write_value(txn_id.clone(), coord, value))
            .buffer_unordered(super::dense::per_block(dtype))
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(copy)
    }

    async fn abs(self, txn: Arc<Txn>) -> TCResult<SparseTensor> {
        let dtype = self.dtype();
        let txn_id = txn.id().clone();
        let copy = SparseTensor::create(txn, self.shape().clone(), dtype).await?;
        self.filled(txn_id.clone())
            .await?
            .map(|(coord, value)| copy.write_value(txn_id.clone(), coord, value.abs()))
            .buffer_unordered(super::dense::per_block(dtype))
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(copy)
    }

    async fn sum(self, txn: Arc<Txn>, axis: usize) -> TCResult<SparseTensor> {
        let txn_id = txn.id().clone();
        if axis == 0 {
            let reduce = |slice: <Self as Slice>::Slice| slice.sum_all(txn_id.clone());
            reduce_axis0(txn, self, reduce).await
        } else {
            let txn_clone = txn.clone();
            let reduce = |slice: <Self as Slice>::Slice| slice.sum(txn_clone.clone(), 0);
            reduce_axis(txn, self, reduce, axis).await
        }
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        let dtype = self.dtype();
        let values = self.filled(txn_id).await?.map(|(_, value)| Ok(value));
        ValueBlockStream::new(values, dtype, super::dense::per_block(dtype))
            .try_fold(dtype.zero(), |sum, block| {
                future::ready(Ok(sum + block.sum()))
            })
            .await
    }

    async fn product(self, txn: Arc<Txn>, axis: usize) -> TCResult<SparseTensor> {
        let txn_id = txn.id().clone();
        if axis == 0 {
            let reduce = |slice: <Self as Slice>::Slice| slice.product_all(txn_id.clone());
            reduce_axis0(txn, self, reduce).await
        } else {
            let txn_clone = txn.clone();
            let reduce = |slice: <Self as Slice>::Slice| slice.product(txn_clone.clone(), 0);
            reduce_axis(txn, self, reduce, axis).await
        }
    }

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number> {
        let dtype = self.dtype();
        if !self.clone().all(txn_id.clone()).await? {
            return Ok(dtype.zero());
        }

        let values = self.filled(txn_id).await?.map(|(_, value)| Ok(value));
        ValueBlockStream::new(values, dtype, super::dense::per_block(dtype))
            .try_fold(dtype.one(), |product, block| {
                future::ready(Ok(product * block.product()))
            })
            .await
    }

    async fn not(self, txn: Arc<Txn>) -> TCResult<SparseTensor> {
        let dtype = self.dtype();
        let txn_id = txn.id().clone();
        let not = SparseTensor::create(txn, self.shape().clone(), NumberType::Bool).await?;
        self.filled(txn_id.clone())
            .await?
            .map(|(coord, value)| (coord, value.into_type(NumberType::Bool)))
            .map(|(coord, value)| {
                let r: TCResult<bool> = value.try_into();
                r.map(|v| (coord, Number::Bool(!v)))
            })
            .map_ok(|(coord, value)| not.write_value(txn_id.clone(), coord, value))
            .try_buffer_unordered(super::dense::per_block(dtype))
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(not)
    }
}

#[async_trait]
impl<T: SparseTensorView + Slice, O: SparseTensorView> SparseTensorCompare<O> for T
where
    <T as Slice>::Slice: SparseTensorView + Slice,
{
    async fn equals(self, other: O, txn: Arc<Txn>) -> TCResult<BlockTensor> {
        let filter = |coord, left, right| {
            if left == right {
                None
            } else {
                Some(coord)
            }
        };

        SparseTensorView::filter_map_dense(self, other, txn, filter, Number::Bool(false)).await
    }

    async fn gt(self, other: O, txn: Arc<Txn>) -> TCResult<SparseTensor> {
        let filter = |coord, left, right| {
            if left > right {
                Some(coord)
            } else {
                None
            }
        };

        SparseTensorView::filter_map(self, other, txn, filter, Number::Bool(true)).await
    }

    async fn gte(self, other: O, txn: Arc<Txn>) -> TCResult<BlockTensor> {
        let filter = |coord, left, right| {
            if left >= right {
                None
            } else {
                Some(coord)
            }
        };

        SparseTensorView::filter_map_dense(self, other, txn, filter, Number::Bool(false)).await
    }

    async fn lt(self, other: O, txn: Arc<Txn>) -> TCResult<SparseTensor> {
        let filter = |coord, left, right| {
            if left < right {
                Some(coord)
            } else {
                None
            }
        };

        SparseTensorView::filter_map(self, other, txn, filter, Number::Bool(true)).await
    }

    async fn lte(self, other: O, txn: Arc<Txn>) -> TCResult<BlockTensor> {
        let filter = |coord, left, right| {
            if left <= right {
                None
            } else {
                Some(coord)
            }
        };

        SparseTensorView::filter_map_dense(self, other, txn, filter, Number::Bool(false)).await
    }
}

async fn reduce_axis0<
    S: SparseTensorView + Slice,
    F: Future<Output = TCResult<Number>>,
    R: Fn(<S as Slice>::Slice) -> F + Send + Sync,
>(
    txn: Arc<Txn>,
    source: S,
    reduce: R,
) -> TCResult<SparseTensor>
where
    <S as Slice>::Slice: SparseTensorUnary,
{
    let dtype = source.dtype();
    let txn_id = txn.id().clone();
    let per_block = super::dense::per_block(dtype);
    let mut shape = source.shape().clone();
    let axis_bound = AxisBounds::all(shape[0]);
    shape.remove(0);
    let reduced = SparseTensor::create(txn.clone(), shape, dtype).await?;

    let axes: Vec<usize> = (1..source.ndim()).collect();
    source
        .filled_at(txn.subcontext_tmp().await?, &axes)
        .await?
        .map(|coord| {
            let bounds: Bounds = (axis_bound.clone(), coord.to_vec()).into();
            (coord, bounds)
        })
        .map(|(coord, source_bounds)| {
            source.clone().slice(source_bounds).map(|slice| {
                reduce(slice).and_then(|reduced_value| future::ready(Ok((coord, reduced_value))))
            })
        })
        .try_buffer_unordered(2)
        .map_ok(|(coord, reduced_value)| reduced.write_value(txn_id.clone(), coord, reduced_value))
        .try_buffer_unordered(per_block)
        .try_fold((), |_, _| future::ready(Ok(())))
        .await?;

    Ok(reduced)
}

async fn reduce_axis<
    S: SparseTensorView + Slice,
    O: SparseTensorView,
    F: Future<Output = TCResult<O>>,
    R: Fn(<S as Slice>::Slice) -> F + Send + Sync,
>(
    txn: Arc<Txn>,
    source: S,
    reduce: R,
    axis: usize,
) -> TCResult<SparseTensor> {
    if axis >= source.ndim() {
        return Err(error::bad_request("Axis out of range", axis));
    }

    let txn_id = txn.id().clone();
    let mut shape = source.shape().clone();
    shape.remove(axis);
    let reduced = SparseTensor::create(txn.clone(), shape, source.dtype()).await?;

    let axes: Vec<usize> = (0..axis).collect();
    source
        .filled_at(txn.subcontext_tmp().await?, &axes)
        .await?
        .map(|prefix| {
            source.clone().slice(prefix.to_vec().into()).map(|slice| {
                reduce(slice).and_then(|reduced_value| future::ready(Ok((prefix, reduced_value))))
            })
        })
        .try_buffer_unordered(2)
        .map_ok(|(prefix, reduced_value)| {
            reduced
                .clone()
                .write_sparse(txn_id.clone(), prefix.into(), reduced_value)
        })
        .try_buffer_unordered(2)
        .try_fold((), |_, _| future::ready(Ok(())))
        .await?;

    Ok(reduced)
}

fn u64_to_value(u: u64) -> Value {
    Value::Number(Number::UInt(UInt::U64(u)))
}

fn unwrap_coord(coord: &[Value]) -> Vec<u64> {
    coord.iter().map(|val| unwrap_u64(val)).collect()
}

fn unwrap_row(mut row: Vec<Value>) -> (Vec<u64>, Number) {
    let coord = unwrap_coord(&row[0..row.len() - 1]);
    let value = row.pop().unwrap().try_into().unwrap();
    (coord, value)
}

fn unwrap_u64(value: &Value) -> u64 {
    if let Value::Number(Number::UInt(UInt::U64(unwrapped))) = value {
        *unwrapped
    } else {
        panic!("Expected u64 but found {}", value)
    }
}
