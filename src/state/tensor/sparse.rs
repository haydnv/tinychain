use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};

use crate::error;
use crate::state::table::{self, Selection, Table, TableBase};
use crate::transaction::{Txn, TxnId};
use crate::value::class::{NumberClass, NumberImpl, NumberType, UIntType, ValueType};
use crate::value::{Number, TCBoxTryFuture, TCResult, TCStream, UInt, Value, ValueId};

use super::bounds::{AxisBounds, Bounds, Shape};
use super::*;

const ERR_BROADCAST_WRITE: &str = "Cannot write to a broadcasted tensor since it is not a \
bijection of its source. Consider copying the broadcast, or writing directly to the source Tensor.";
const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

pub type SparseStream = TCStream<(Vec<u64>, Number)>;

#[async_trait]
trait SparseAccessor: TensorView + 'static {
    fn filled<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream>;

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>>;

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64>;

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream>;

    fn filled_range<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        start: Vec<u64>,
        end: Vec<u64>,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream>;

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

impl SparseBroadcast {
    fn broadcast(&self, coord: Vec<u64>, value: Number) -> impl Stream<Item = (Vec<u64>, Number)> {
        let broadcast = self
            .rebase
            .map_bounds(coord.into())
            .affected()
            .map(move |coord| (coord, value.clone()));

        stream::iter(broadcast)
    }
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
    fn filled<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let filled = self
                .source
                .clone()
                .filled(txn_id, order.map(|axes| self.rebase.invert_axes(axes)))
                .await?
                .map(move |(coord, value)| self.broadcast(coord, value))
                .flatten();

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        let filled = self.source.clone().filled(txn_id, None).await?;
        let rebase = self.rebase.clone();
        let count = filled
            .fold(0u64, |count, (coord, _)| {
                future::ready(count + rebase.map_bounds(coord.into()).size())
            })
            .await;

        Ok(count)
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        group_axes(self, txn_id, axes).await
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let order = order.map(|axes| self.rebase.invert_axes(axes));
            let filled_in = self
                .source
                .clone()
                .filled_in(txn_id, bounds, order)
                .await?
                .map(move |(coord, value)| self.broadcast(coord, value))
                .flatten();

            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
        })
    }

    fn filled_range<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        start: Vec<u64>,
        end: Vec<u64>,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let filled_range = self
                .source
                .clone()
                .filled_range(
                    txn_id,
                    self.rebase.invert_coord(&start),
                    self.rebase.invert_coord(&end),
                    order.map(|axes| self.rebase.invert_axes(axes)),
                )
                .await?
                .map(move |(coord, value)| self.broadcast(coord, value))
                .flatten();

            let filled_range: SparseStream = Box::pin(filled_range);
            Ok(filled_range)
        })
    }

    fn read_value<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(future::ready(Err(error::unsupported(ERR_BROADCAST_WRITE))))
    }

    fn write_value<'a>(
        &'a self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(ERR_BROADCAST_WRITE))))
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
    fn filled<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let dtype = self.dtype;
            let filled = self.source.clone().filled(txn_id, order).await?;
            let cast = filled.map(move |(coord, value)| (coord, value.into_type(dtype)));
            let cast: SparseStream = Box::pin(cast);
            Ok(cast)
        })
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

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let dtype = self.dtype;
            let source = self.source.clone().filled_in(txn_id, bounds, order).await?;
            let filled_in = source.map(move |(coord, value)| (coord, value.into_type(dtype)));
            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
        })
    }

    fn filled_range<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        start: Vec<u64>,
        end: Vec<u64>,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let dtype = self.dtype;
            let filled_range = self
                .source
                .clone()
                .filled_range(txn_id, start, end, order)
                .await?
                .map(move |(coord, value)| (coord, value.into_type(dtype)));
            let filled_range: SparseStream = Box::pin(filled_range);
            Ok(filled_range)
        })
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

struct SparseExpand {
    source: Arc<dyn SparseAccessor>,
    rebase: transform::Expand,
}

impl TensorView for SparseExpand {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim() + 1
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.shape().size()
    }
}

#[async_trait]
impl SparseAccessor for SparseExpand {
    fn filled<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let filled = self
                .source
                .clone()
                .filled(txn_id, order.map(|axes| self.rebase.invert_axes(axes)))
                .await?
                .map(move |(coord, value)| (self.rebase.map_coord(coord), value));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        group_axes(self, txn_id, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.source.clone().filled_count(txn_id).await
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let order = order.map(|axes| self.rebase.invert_axes(axes));
            let filled_in = self
                .source
                .clone()
                .filled_in(txn_id, bounds, order)
                .await?
                .map(move |(coord, value)| (self.rebase.map_coord(coord), value));

            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
        })
    }

    fn filled_range<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        start: Vec<u64>,
        end: Vec<u64>,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let filled_range = self
                .source
                .clone()
                .filled_range(
                    txn_id,
                    rebase.invert_coord(&start),
                    rebase.invert_coord(&end),
                    order.map(|axes| rebase.invert_axes(axes)),
                )
                .await?
                .map(move |(coord, value)| (rebase.map_coord(coord), value));
            let filled_range: SparseStream = Box::pin(filled_range);
            Ok(filled_range)
        })
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
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value(txn_id, coord, value)
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
    fn filled<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let order = order.map(|axes| rebase.invert_axes(axes));
            let filled = self
                .source
                .clone()
                .filled_in(txn_id, rebase.bounds().clone(), order)
                .await?
                .map(move |(coord, value)| (rebase.map_coord(coord), value));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        group_axes(self, txn_id, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        let count = self
            .filled(txn_id, None)
            .await?
            .fold(0u64, |count, _| future::ready(count + 1))
            .await;
        Ok(count)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let order = order.map(|axes| self.rebase.invert_axes(axes));
            let filled_in = self
                .source
                .clone()
                .filled_in(txn_id, bounds, order)
                .await?
                .map(move |(coord, value)| (self.rebase.map_coord(coord), value));

            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
        })
    }

    fn filled_range<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        start: Vec<u64>,
        end: Vec<u64>,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let filled_range = self
                .source
                .clone()
                .filled_range(
                    txn_id,
                    rebase.invert_coord(&start),
                    rebase.invert_coord(&end),
                    order.map(|axes| rebase.invert_axes(axes)),
                )
                .await?
                // TODO: use a filter to handle the case of a slice with step > 1
                .map(move |(coord, value)| (rebase.map_coord(coord), value));
            let filled_range: SparseStream = Box::pin(filled_range);
            Ok(filled_range)
        })
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
    fn filled<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        let order = if let Some(order) = order {
            self.rebase.invert_axes(order)
        } else {
            self.rebase.invert_axes((0..self.ndim()).collect())
        };

        self.source.clone().filled(txn_id, Some(order))
    }

    async fn filled_at(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCResult<TCStream<Vec<u64>>> {
        group_axes(self, txn_id, axes).await
    }

    async fn filled_count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.source.clone().filled_count(txn_id).await
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);

            let order = if let Some(order) = order {
                self.rebase.invert_axes(order)
            } else {
                self.rebase.invert_axes((0..self.ndim()).collect())
            };

            let source = self
                .source
                .clone()
                .filled_in(txn_id, bounds, Some(order))
                .await?;
            let filled_in = source.map(move |(coord, value)| (self.rebase.map_coord(coord), value));
            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
        })
    }

    fn filled_range<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        start: Vec<u64>,
        end: Vec<u64>,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rebase = self.rebase.clone();

            let order = if let Some(order) = order {
                rebase.invert_axes(order)
            } else {
                rebase.invert_axes((0..self.ndim()).collect())
            };

            let filled_range = self
                .source
                .clone()
                .filled_range(
                    txn_id,
                    rebase.invert_coord(&start),
                    rebase.invert_coord(&end),
                    Some(order),
                )
                .await?
                .map(move |(coord, value)| (rebase.map_coord(coord), value));
            let filled_range: SparseStream = Box::pin(filled_range);
            Ok(filled_range)
        })
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

struct SparseTable {
    table: TableBase,
    shape: Shape,
    dtype: NumberType,
}

impl SparseTable {
    pub async fn create(txn: Arc<Txn>, shape: Shape, dtype: NumberType) -> TCResult<SparseTable> {
        let u64_type = ValueType::Number(NumberType::UInt(UIntType::U64));
        let key: Vec<table::schema::Column> = (0..shape.len())
            .map(|axis| table::schema::Column {
                name: axis.into(),
                dtype: u64_type,
                max_len: None,
            })
            .collect();

        let value = vec![table::schema::Column {
            name: "value".parse()?,
            dtype: dtype.into(),
            max_len: None,
        }];

        let schema = table::schema::Schema::new(key, value);
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
    fn filled<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
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
            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
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

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let source = slice_table(self.table.clone().into(), &txn_id, &bounds, order).await?;
            let filled_in = source.stream(txn_id).await?.map(unwrap_row);
            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
        })
    }

    fn filled_range<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        start: Vec<u64>,
        end: Vec<u64>,
        order: Option<Vec<usize>>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let table_bounds = coords_to_table_bounds(start, end);

            let table = if let Some(mut order) = order {
                let order: Vec<ValueId> = order.drain(..).map(ValueId::from).collect();
                self.table
                    .order_by(&txn_id, order, false)
                    .await?
                    .slice(&txn_id, table_bounds)
                    .await?
            } else {
                self.table.slice(&txn_id, table_bounds).await?
            };

            let filled_range = table.stream(txn_id).await?.map(unwrap_row);

            let filled_range: SparseStream = Box::pin(filled_range);
            Ok(filled_range)
        })
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

impl SparseTensor {
    pub fn filled_range<'a>(
        &'a self,
        txn_id: TxnId,
        start: Vec<u64>,
        end: Vec<u64>,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        self.accessor.clone().filled_range(txn_id, start, end, None)
    }
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

impl TensorIO for SparseTensor {
    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        self.accessor.read_value(txn_id, coord)
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            stream::iter(bounds.affected())
                .map(|coord| Ok(self.write_value_at(txn_id.clone(), coord, value.clone())))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.accessor.write_value(txn_id, coord, value)
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

    fn expand_dims(&self, axis: usize) -> TCResult<Self> {
        let rebase = transform::Expand::new(self.shape().clone(), axis)?;
        let accessor = Arc::new(SparseExpand {
            source: self.accessor.clone(),
            rebase,
        });

        Ok(SparseTensor { accessor })
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

async fn group_axes(
    accessor: Arc<dyn SparseAccessor>,
    txn_id: TxnId,
    axes: Vec<usize>,
) -> TCResult<TCStream<Vec<u64>>> {
    if axes.len() > accessor.ndim() {
        let axes: Vec<String> = axes.iter().map(|x| x.to_string()).collect();
        return Err(error::bad_request("Axis out of bounds", axes.join(", ")));
    }

    let axes_clone = axes.to_vec();
    let left = accessor
        .clone()
        .filled(txn_id.clone(), None)
        .await?
        .map(move |(coord, _)| axes_clone.iter().map(|x| coord[*x]).collect::<Vec<u64>>());

    let mut right = accessor
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

fn slice_table<'a>(
    mut table: Table,
    txn_id: &'a TxnId,
    bounds: &'a Bounds,
    order: Option<Vec<usize>>,
) -> TCBoxTryFuture<'a, Table> {
    use AxisBounds::*;

    Box::pin(async move {
        if let Some(order) = order {
            let order = order.iter().map(|x| ValueId::from(*x)).collect();
            table = table.order_by(txn_id, order, false).await?;
        }

        for (axis, axis_bound) in bounds.to_vec().into_iter().enumerate() {
            let axis: ValueId = axis.into();
            table = match axis_bound {
                At(x) => {
                    let column_bound = table::schema::ColumnBound::Is(u64_to_value(x));
                    table
                        .slice(txn_id, iter::once((axis, column_bound)).collect())
                        .await?
                }
                In(range, 1) => {
                    let start = Bound::Included(u64_to_value(range.start));
                    let end = Bound::Excluded(u64_to_value(range.end));
                    let column_bound = table::schema::ColumnBound::In(start, end);
                    table
                        .slice(txn_id, iter::once((axis, column_bound)).collect())
                        .await?
                }
                _ => todo!(),
            };
        }

        Ok(table)
    })
}

fn coords_to_table_bounds(mut start: Vec<u64>, mut end: Vec<u64>) -> table::schema::Bounds {
    assert!(start.len() == end.len());

    start
        .drain(..)
        .map(u64_to_value)
        .zip(end.drain(..).map(u64_to_value))
        .map(|(s, e)| table::schema::ColumnBound::In(Bound::Included(s), Bound::Excluded(e)))
        .enumerate()
        .map(|(axis, bound)| (ValueId::from(axis), bound))
        .collect()
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
