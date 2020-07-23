use std::collections::HashMap;
use std::convert::TryInto;
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};

use crate::error;
use crate::state::table::schema::*;
use crate::state::table::{Selection, Table, TableBase};
use crate::transaction::{Txn, TxnId};
use crate::value::class::{NumberClass, NumberType, UIntType, ValueType};
use crate::value::{Number, TCResult, TCStream, UInt, Value, ValueId};

use super::base::*;
use super::bounds::{AxisBounds, Bounds, Shape};
use super::dense::{BlockTensor, DenseTensorView};

const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

#[async_trait]
pub trait SparseTensorView: TensorView + 'static {
    async fn filter_map_dense<
        O: SparseTensorView,
        F: Fn(Vec<u64>, Number, Number) -> Option<Vec<u64>> + Send + Sync,
    >(
        txn: Arc<Txn>,
        this: Arc<Self>,
        that: Arc<O>,
        filter: F,
        value: Number,
    ) -> TCResult<Arc<BlockTensor>>
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
                this.clone()
                    .read_value(txn_id.clone(), coord.to_vec())
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

    async fn filled(self: Arc<Self>, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>>;

    async fn filled_at(&self, txn: Arc<Txn>, axes: &[usize]) -> TCResult<TCStream<Vec<u64>>>;

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64>;

    async fn read_value(self: Arc<Self>, txn_id: TxnId, coord: Vec<u64>) -> TCResult<Number>;

    async fn write_sparse<T: SparseTensorView>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        value: Arc<T>,
    ) -> TCResult<()>
    where
        Self: Slice,
        <Self as Slice>::Slice: SparseTensorView,
    {
        let dest = self.slice(bounds)?;
        value
            .filled(txn_id.clone())
            .await?
            .map(|(coord, number)| Ok(dest.clone().write_value(txn_id.clone(), coord, number)))
            .try_buffer_unordered(dest.size() as usize)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCResult<()>;
}

pub struct SparseTensor {
    dtype: NumberType,
    shape: Shape,
    table: TableBase,
}

impl SparseTensor {
    pub async fn create(
        txn: Arc<Txn>,
        shape: Shape,
        dtype: NumberType,
    ) -> TCResult<Arc<SparseTensor>> {
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

        Ok(Arc::new(SparseTensor {
            dtype,
            shape,
            table,
        }))
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
    async fn filled(self: Arc<Self>, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>> {
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

    async fn read_value(self: Arc<Self>, txn_id: TxnId, coord: Vec<u64>) -> TCResult<Number> {
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
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        mut coord: Vec<u64>,
        value: Number,
    ) -> TCResult<()> {
        let mut row: HashMap<ValueId, Value> = coord
            .drain(..)
            .enumerate()
            .map(|(x, v)| (x.into(), Value::Number(Number::UInt(UInt::U64(v)))))
            .collect();

        row.insert("value".parse()?, value.into());
        self.table.upsert(&txn_id, row).await
    }
}

impl Slice for SparseTensor {
    type Slice = TensorSlice<SparseTensor>;

    fn slice(self: Arc<Self>, bounds: Bounds) -> TCResult<Arc<Self::Slice>> {
        Ok(Arc::new(TensorSlice::new(self, bounds)?))
    }
}

#[async_trait]
impl SparseTensorView for TensorSlice<SparseTensor> {
    async fn filled(self: Arc<Self>, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let stream = self
            .source()
            .slice_table(&txn_id, self.bounds())
            .await?
            .stream(txn_id)
            .await?
            .map(unwrap_row)
            .map(move |(coord, value)| (self.map_bounds(coord.into()).into_coord(), value));

        Ok(Box::pin(stream))
    }

    async fn filled_at(&self, txn: Arc<Txn>, axes: &[usize]) -> TCResult<TCStream<Vec<u64>>> {
        let table = self.source().slice_table(txn.id(), self.bounds()).await?;
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

    async fn read_value(self: Arc<Self>, txn_id: TxnId, coord: Vec<u64>) -> TCResult<Number> {
        if !self.shape().contains_coord(&coord) {
            return Err(error::bad_request(
                "Coordinate out of bounds",
                Bounds::from(coord),
            ));
        }

        let source_coord = self.invert_bounds(coord.into()).into_coord();
        self.source().read_value(txn_id, source_coord).await
    }

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.source()
            .slice_table(&txn_id, self.bounds())
            .await?
            .count(txn_id)
            .await
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCResult<()> {
        self.source()
            .write_value(txn_id, self.invert_bounds(coord.into()).into_coord(), value)
            .await
    }
}

#[async_trait]
impl<T: SparseTensorView + Slice, O: SparseTensorView> SparseTensorCompare<O> for T
where
    <T as Slice>::Slice: SparseTensorView + Slice,
{
    async fn equals(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<BlockTensor>> {
        let filter = |coord, left, right| {
            if left == right {
                None
            } else {
                Some(coord)
            }
        };

        SparseTensorView::filter_map_dense(txn, self, other, filter, Number::Bool(false)).await
    }

    async fn gt(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<SparseTensor>> {
        if self.shape() != other.shape() {
            return Err(error::bad_request(
                "Cannot compare tensor with shape",
                other.shape(),
            ));
        }

        let txn_id = txn.id().clone();
        let txn_id_clone = txn_id.clone();
        let per_block = super::dense::per_block(NumberType::Bool);
        let gt = SparseTensor::create(txn, self.shape().clone(), NumberType::Bool).await?;

        let other_clone = other.clone();
        let this = self
            .clone()
            .filled(txn_id.clone())
            .await?
            .map(|(coord, left)| {
                other
                    .clone()
                    .read_value(txn_id.clone(), coord.to_vec())
                    .and_then(move |right| future::ready(Ok((coord, left, right))))
            })
            .buffer_unordered(per_block)
            .try_filter_map(|(coord, left, right)| {
                if left > right {
                    future::ready(Ok(Some(coord)))
                } else {
                    future::ready(Ok(None))
                }
            });

        let that = other_clone
            .filled(txn_id.clone())
            .await?
            .map(|(coord, right)| {
                self.clone()
                    .read_value(txn_id.clone(), coord.to_vec())
                    .and_then(move |left| future::ready(Ok((coord, left, right))))
            })
            .buffer_unordered(per_block)
            .try_filter_map(|(coord, left, right)| {
                if left > right {
                    future::ready(Ok(Some(coord)))
                } else {
                    future::ready(Ok(None))
                }
            });

        this.chain(that)
            .map_ok(|coord| {
                gt.clone()
                    .write_value(txn_id_clone.clone(), coord, Number::Bool(true))
            })
            .try_buffer_unordered(per_block)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(gt)
    }

    async fn gte(self: Arc<Self>, other: Arc<O>, txn: Arc<Txn>) -> TCResult<Arc<BlockTensor>> {
        let filter = |coord, left, right| {
            if left >= right {
                None
            } else {
                Some(coord)
            }
        };

        SparseTensorView::filter_map_dense(txn, self, other, filter, Number::Bool(false)).await
    }

    async fn lt(self: Arc<Self>, _other: Arc<O>, _txn: Arc<Txn>) -> TCResult<Arc<SparseTensor>> {
        Err(error::not_implemented())
    }

    async fn lte(self: Arc<Self>, _other: Arc<O>, _txn: Arc<Txn>) -> TCResult<Arc<BlockTensor>> {
        Err(error::not_implemented())
    }
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
