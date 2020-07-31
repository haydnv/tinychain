use std::collections::HashMap;
use std::convert::TryInto;
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt};

use crate::error;
use crate::state::table::schema::*;
use crate::state::table::{Selection, Table, TableBase};
use crate::transaction::{Txn, TxnId};
use crate::value::class::{NumberClass, NumberImpl, NumberType, UIntType, ValueType};
use crate::value::{Number, TCBoxTryFuture, TCResult, TCStream, UInt, Value, ValueId};

use super::bounds::{AxisBounds, Bounds, Shape};
use super::*;

const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

#[async_trait]
trait SparseAccessor: TensorView + 'static {
    async fn filled(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>>;

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>>;

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64>;

    async fn filled_in(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>>;

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number>;

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()>;
}

struct SparseBroadcast {
    source: Arc<dyn SparseAccessor>,
    rebase: transform::Broadcast,
}

impl TensorView for SparseBroadcast {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl SparseAccessor for SparseBroadcast {
    async fn filled(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let rebase = self.rebase.clone();
        let source = self
            .source
            .clone()
            .filled(txn_id, order.map(|axes| rebase.invert_axes(axes)))
            .await?;
        let filled = source
            .map(move |(coord, value)| {
                let broadcast = rebase
                    .map_bounds(coord.into())
                    .affected()
                    .map(move |coord| (coord, value.clone()));
                stream::iter(broadcast)
            })
            .flatten();

        Ok(Box::pin(filled))
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        let filled = self.source.clone().filled(txn_id, None).await?;
        let rebase = self.rebase.clone();
        Ok(filled
            .fold(0u64, |count, (coord, _)| {
                future::ready(count + rebase.map_bounds(coord.into()).size())
            })
            .await)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        self.source
            .clone()
            .filled_at(txn_id, self.rebase.invert_axes(axes))
            .await
    }

    async fn filled_in(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let bounds = self.rebase.invert_bounds(bounds);
        let order = order.map(|axes| self.rebase.invert_axes(axes));
        self.source.clone().filled_in(txn_id, bounds, order).await
    }

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            self.source
                .read_value(txn_id, &self.rebase.invert_coord(coord))
                .await
        })
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.source
            .write_value(txn_id, self.rebase.invert_coord(&coord), value)
    }
}

struct SparseCast {
    source: Arc<dyn SparseAccessor>,
    dtype: NumberType,
}

impl TensorView for SparseCast {
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
impl SparseAccessor for SparseCast {
    async fn filled(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let dtype = self.dtype;
        let filled = self.source.clone().filled(txn_id, order).await?;
        let cast = filled.map(move |(coord, value)| (coord, value.into_type(dtype)));
        Ok(Box::pin(cast))
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        self.source.clone().filled_at(txn_id, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.source.clone().filled_count(txn_id).await
    }

    async fn filled_in(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let dtype = self.dtype;
        let source = self.source.clone().filled_in(txn_id, bounds, order).await?;
        Ok(Box::pin(source.map(move |(coord, value)| {
            (coord, value.into_type(dtype))
        })))
    }

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<Number> {
        let dtype = self.dtype;
        Box::pin(
            self.source
                .read_value(txn_id, coord)
                .map_ok(move |value| value.into_type(dtype)),
        )
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<()> {
        self.source.write_value(txn_id, coord, value)
    }
}

struct SparseTranspose {
    source: Arc<dyn SparseAccessor>,
    rebase: transform::Transpose,
}

impl TensorView for SparseTranspose {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl SparseAccessor for SparseTranspose {
    async fn filled(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let order = if let Some(order) = order {
            self.rebase.invert_axes(order)
        } else {
            self.rebase.invert_axes((0..self.ndim()).collect())
        };

        self.source.clone().filled(txn_id, Some(order)).await
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        self.source
            .clone()
            .filled_at(txn_id, self.rebase.invert_axes(axes))
            .await
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.source.clone().filled_count(txn_id).await
    }

    async fn filled_in(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let bounds = self.rebase.invert_bounds(bounds);

        let order = if let Some(order) = order {
            self.rebase.invert_axes(order)
        } else {
            self.rebase.invert_axes((0..self.ndim()).collect())
        };

        self.source
            .clone()
            .filled_in(txn_id, bounds, Some(order))
            .await
    }

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let coord = self.rebase.invert_coord(coord);
            self.source.read_value(txn_id, &coord).await
        })
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.source
            .write_value(txn_id, self.rebase.invert_coord(&coord), value)
    }
}

struct SparseSlice {
    source: Arc<dyn SparseAccessor>,
    rebase: transform::Slice,
}

impl TensorView for SparseSlice {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.rebase.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.rebase.size()
    }
}

#[async_trait]
impl SparseAccessor for SparseSlice {
    async fn filled(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let order = order.map(|axes| self.rebase.invert_axes(axes));
        self.source
            .clone()
            .filled_in(txn_id, self.rebase.bounds().clone(), order)
            .await
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        if axes.len() > self.ndim() {
            let axes: Vec<String> = axes.iter().map(|x| x.to_string()).collect();
            return Err(error::bad_request("Axis out of bounds", axes.join(", ")));
        }

        let axes_clone = axes.to_vec();
        let left = self
            .clone()
            .filled(txn_id.clone(), None)
            .await?
            .map(move |(coord, _)| axes_clone.iter().map(|x| coord[*x]).collect::<Vec<u64>>());

        let mut right = self
            .clone()
            .filled(txn_id, None)
            .await?
            .map(move |(coord, _)| axes.iter().map(|x| coord[*x]).collect::<Vec<u64>>());

        if right.next().await.is_none() {
            return Ok(Box::pin(stream::empty()));
        }

        let filled_at = left.zip(right).filter_map(|(l, r)| {
            if l == r {
                future::ready(Some(l))
            } else {
                future::ready(None)
            }
        });

        Ok(Box::pin(filled_at))
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        let count = self
            .filled(txn_id, None)
            .await?
            .fold(0u64, |count, _| future::ready(count + 1))
            .await;
        Ok(count)
    }

    async fn filled_in(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let bounds = self.rebase.invert_bounds(bounds);
        let order = order.map(|axes| self.rebase.invert_axes(axes));
        self.source.clone().filled_in(txn_id, bounds, order).await
    }

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let coord = self.rebase.invert_coord(coord);
            self.source.read_value(txn_id, &coord).await
        })
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let coord = self.rebase.invert_coord(&coord);
            self.source.write_value(txn_id, coord, value).await
        })
    }
}

struct SparseTable {
    table: TableBase,
    shape: Shape,
    dtype: NumberType,
}

impl SparseTable {
    pub async fn create(txn: Arc<Txn>, shape: Shape, dtype: NumberType) -> TCResult<SparseTable> {
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

        Ok(SparseTable {
            table,
            dtype,
            shape,
        })
    }
}

impl TensorView for SparseTable {
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
impl SparseAccessor for SparseTable {
    async fn filled(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let rows = if let Some(mut order) = order {
            let order: Vec<ValueId> = order.drain(..).map(ValueId::from).collect();
            self.table
                .order_by(&txn_id, order, false)
                .await?
                .stream(txn_id)
                .await?
        } else {
            self.table.clone().stream(txn_id).await?
        };
        let filled = rows.map(unwrap_row);
        Ok(Box::pin(filled))
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        let columns: Vec<ValueId> = axes.iter().map(|x| (*x).into()).collect();
        let filled_at = self
            .table
            .group_by(txn_id.clone(), columns)
            .await?
            .stream(txn_id)
            .await?
            .map(|coord| unwrap_coord(&coord));

        Ok(Box::pin(filled_at))
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.table.count(txn_id).await
    }

    async fn filled_in(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let source = slice_table(self.table.clone().into(), &txn_id, &bounds, order).await?;
        let filled_in = source.stream(txn_id).await?.map(unwrap_row);
        Ok(Box::pin(filled_in))
    }

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            if !self.shape().contains_coord(coord) {
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
                .stream(txn_id.clone())
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
    ) -> TCBoxTryFuture<'a, ()> {
        let value = value.into_type(self.dtype);

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

#[derive(Clone)]
pub struct SparseTensor {
    accessor: Arc<dyn SparseAccessor>,
}

impl TensorView for SparseTensor {
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

impl TensorTransform for SparseTensor {
    fn as_type(&self, dtype: NumberType) -> TCResult<Self> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }

        let accessor = Arc::new(SparseCast {
            source: self.accessor.clone(),
            dtype: self.dtype(),
        });

        Ok(SparseTensor { accessor })
    }

    fn broadcast(&self, shape: Shape) -> TCResult<Self> {
        if &shape == self.shape() {
            return Ok(self.clone());
        }

        let rebase = transform::Broadcast::new(self.shape().clone(), shape)?;
        let accessor = Arc::new(SparseBroadcast {
            source: self.accessor.clone(),
            rebase,
        });

        Ok(SparseTensor { accessor })
    }

    fn expand_dims(&self, _axis: usize) -> TCResult<Self> {
        Err(error::not_implemented())
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Self> {
        if bounds == Bounds::all(self.shape()) {
            return Ok(self.clone());
        }

        let rebase = transform::Slice::new(self.shape().clone(), bounds)?;
        let accessor = Arc::new(SparseSlice {
            source: self.accessor.clone(),
            rebase,
        });

        Ok(SparseTensor { accessor })
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        if permutation == Some((0..self.ndim()).collect()) {
            return Ok(self.clone());
        }

        let rebase = transform::Transpose::new(self.shape().clone(), permutation)?;
        let accessor = Arc::new(SparseTranspose {
            source: self.accessor.clone(),
            rebase,
        });

        Ok(SparseTensor { accessor })
    }
}

async fn slice_table(
    mut table: Table,
    txn_id: &TxnId,
    bounds: &Bounds,
    order: Option<Vec<usize>>,
) -> TCResult<Table> {
    use AxisBounds::*;

    if let Some(order) = order {
        let order = order.iter().map(|x| ValueId::from(*x)).collect();
        table = table.order_by(txn_id, order, false).await?;
    }

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
            _ => todo!(),
        };
    }

    Ok(table)
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
