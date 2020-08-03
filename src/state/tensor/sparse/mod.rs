use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::ops::Bound;
use std::sync::Arc;

use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;

use crate::error;
use crate::state::table::{self, Selection, Table, TableBase};
use crate::transaction::{Txn, TxnId};
use crate::value::class::{NumberClass, NumberImpl, NumberType, UIntType, ValueType};
use crate::value::{Number, TCBoxTryFuture, TCResult, TCStream, UInt, Value, ValueId};

use super::bounds::{AxisBounds, Bounds, Shape};
use super::*;

mod combinator;
mod combine;

use combine::SparseCombine;

pub type SparseRow = (Vec<u64>, Number);
pub type SparseStream = TCStream<SparseRow>;

const ERR_BROADCAST_WRITE: &str = "Cannot write to a broadcasted tensor since it is not a \
bijection of its source. Consider copying the broadcast into a new Tensor, \
or writing directly to the source Tensor.";

const ERR_PRODUCT_WRITE: &str = "Cannot write to a product of two Tensors. \
Consider copying the product into a new Tensor, or writing to the source Tensors directly.";

const ERR_NOT_SPARSE: &str = "The result of the requested operation would not be sparse;\
convert to a DenseTensor first.";

const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

trait SparseAccessor: TensorView + 'static {
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream>;

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>>;

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64>;

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream>;

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number>;

    fn read_value_owned<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        coord: Vec<u64>,
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move { self.read_value(&txn_id, &coord).await })
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()>;
}

struct DenseAccessor {
    source: DenseTensor,
}

impl TensorView for DenseAccessor {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
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
impl SparseAccessor for DenseAccessor {
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let source = self.source.clone();
            // TODO: remove this .unwrap() and have SparseStream return a Result
            let values = source.value_stream(txn_id).await?.map(|r| r.unwrap());

            let zero = self.dtype().zero();
            let filled = stream::iter(Bounds::all(self.shape()).affected())
                .zip(values)
                .filter(move |(_, value)| future::ready(value != &zero));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
        Box::pin(async move {
            let shape = self.shape();
            let source = self.source.clone();
            let bounds: HashMap<usize, AxisBounds> = shape
                .to_vec()
                .drain(..)
                .map(|dim| AxisBounds::In(0..dim))
                .enumerate()
                .collect();

            let filled_at = stream::iter(
                Bounds::all(&Shape::from(
                    axes.iter().map(|x| shape[*x]).collect::<Vec<u64>>(),
                ))
                .affected(),
            )
            .filter_map(move |at| {
                let source = source.clone();
                let txn_id = txn_id.clone();
                let axes = axes.to_vec();
                let mut bounds = bounds.clone();

                Box::pin(async move {
                    for (axis, coord) in axes.iter().zip(at.iter()) {
                        bounds.insert(*axis, AxisBounds::At(*coord));
                    }
                    let bounds: Bounds = (0..bounds.len())
                        .map(|x| bounds.remove(&x).unwrap())
                        .collect::<Vec<AxisBounds>>()
                        .into();

                    let slice = source.slice(bounds).unwrap(); // TODO: remove this call to .unwrap()
                    if slice.any(txn_id.clone()).await.unwrap() {
                        // TODO: remove this call to .unwrap()
                        Some(at)
                    } else {
                        None
                    }
                })
            });

            let filled_at: TCStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            self.source
                .value_stream(txn_id)
                .await?
                .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
                .await
        })
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        match self.source.slice(bounds) {
            Ok(source) => {
                let slice = Arc::new(DenseAccessor { source });
                slice.filled(txn_id)
            }
            Err(cause) => Box::pin(future::ready(Err(cause))),
        }
    }

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        self.source.read_value(txn_id, coord)
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        self.source.write_value(txn_id, coord.into(), value)
    }
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
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let filled = self
                .source
                .clone()
                .filled(txn_id)
                .await?
                .map(move |(coord, value)| self.broadcast(coord, value))
                .flatten();

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            let filled = self.source.clone().filled(txn_id).await?;
            let rebase = self.rebase.clone();
            let count = filled
                .fold(0u64, |count, (coord, _)| {
                    future::ready(count + rebase.map_bounds(coord.into()).size())
                })
                .await;

            Ok(count)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
        group_axes(self, txn_id, axes)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let filled_in = self
                .source
                .clone()
                .filled_in(txn_id, bounds)
                .await?
                .map(move |(coord, value)| self.broadcast(coord, value))
                .flatten();

            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
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

impl SparseAccessor for SparseCast {
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let dtype = self.dtype;
            let filled = self.source.clone().filled(txn_id).await?;
            let cast = filled.map(move |(coord, value)| (coord, value.into_type(dtype)));
            let cast: SparseStream = Box::pin(cast);
            Ok(cast)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
        self.source.clone().filled_at(txn_id, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64> {
        self.source.clone().filled_count(txn_id)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let dtype = self.dtype;
            let source = self.source.clone().filled_in(txn_id, bounds).await?;
            let filled_in = source.map(move |(coord, value)| (coord, value.into_type(dtype)));
            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
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

struct SparseCombinator {
    left: Arc<dyn SparseAccessor>,
    right: Arc<dyn SparseAccessor>,
    combinator: fn(Option<Number>, Option<Number>) -> Option<Number>,
    zero: Number,
}

impl SparseCombinator {
    fn new(
        left: Arc<dyn SparseAccessor>,
        right: Arc<dyn SparseAccessor>,
        combinator: fn(Option<Number>, Option<Number>) -> Option<Number>,
    ) -> TCResult<SparseCombinator> {
        if left.shape() != right.shape() {
            return Err(error::internal(
                "Tried to combine SparseTensors with different shapes",
            ));
        }

        let zero = left.dtype().zero() * right.dtype().zero();

        Ok(SparseCombinator {
            left,
            right,
            combinator,
            zero,
        })
    }

    fn filled_inner(&self, left: SparseStream, right: SparseStream) -> SparseStream {
        let combinator = self.combinator;
        let zero = self.zero.clone();
        let combined = SparseCombine::new(left, right).filter_map(move |(coord, l, r)| {
            let row = if let Some(value) = combinator(l, r) {
                if value == zero {
                    None
                } else {
                    Some((coord, value))
                }
            } else {
                None
            };

            future::ready(row)
        });

        Box::pin(combined)
    }
}

impl TensorView for SparseCombinator {
    fn dtype(&self) -> NumberType {
        self.left.dtype()
    }

    fn ndim(&self) -> usize {
        self.left.ndim() + 1
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.left.shape()
    }

    fn size(&self) -> u64 {
        self.left.size()
    }
}

#[async_trait]
impl SparseAccessor for SparseCombinator {
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let left = self.left.clone().filled(txn_id.clone());
            let right = self.right.clone().filled(txn_id);
            let (left, right) = try_join!(left, right)?;
            Ok(self.filled_inner(left, right))
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
        group_axes(self, txn_id, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            let count = self
                .filled(txn_id)
                .await?
                .fold(0u64, |count, _| future::ready(count + 1))
                .await;

            Ok(count)
        })
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let left = self.left.clone().filled_in(txn_id.clone(), bounds.clone());

            let right = self.right.clone().filled_in(txn_id, bounds);

            let (left, right) = try_join!(left, right)?;

            Ok(self.filled_inner(left, right))
        })
    }

    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        Box::pin(async move {
            let left = self.left.read_value(txn_id, coord);
            let right = self.right.read_value(txn_id, coord);
            let (left, right) = try_join!(left, right)?;
            let combinator = self.combinator;
            Ok(combinator(Some(left), Some(right)).unwrap_or_else(|| self.zero.clone()))
        })
    }

    fn write_value<'a>(
        &'a self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(ERR_PRODUCT_WRITE))))
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
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let filled = self
                .source
                .clone()
                .filled(txn_id)
                .await?
                .map(move |(coord, value)| (self.rebase.map_coord(coord), value));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
        group_axes(self, txn_id, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64> {
        self.source.clone().filled_count(txn_id)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let filled_in = self
                .source
                .clone()
                .filled_in(txn_id, bounds)
                .await?
                .map(move |(coord, value)| (self.rebase.map_coord(coord), value));

            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
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
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let filled = self
                .source
                .clone()
                .filled_in(txn_id, rebase.bounds().clone())
                .await?
                .map(move |(coord, value)| (rebase.map_coord(coord), value));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
        group_axes(self, txn_id, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            let count = self
                .filled(txn_id)
                .await?
                .fold(0u64, |count, _| future::ready(count + 1))
                .await;

            Ok(count)
        })
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let filled_in = self
                .source
                .clone()
                .filled_in(txn_id, bounds)
                .await?
                .map(move |(coord, value)| (self.rebase.map_coord(coord), value));

            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
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

impl SparseAccessor for SparseTranspose {
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let ndim = self.ndim();
            let filled = self
                .clone()
                .filled_at(txn_id.clone(), (0..ndim).collect())
                .await?
                .then(move |coord| {
                    self.clone()
                        .read_value_owned(txn_id.clone(), coord.to_vec())
                        .map_ok(|value| (coord, value))
                })
                .map(|r| r.unwrap()); // TODO: remove this call to .unwrap()

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let filled_at = self
                .source
                .clone()
                .filled_at(txn_id, rebase.invert_axes(axes))
                .await?
                .map(move |coord| rebase.map_coord(coord));

            let filled_at: TCStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64> {
        self.source.clone().filled_count(txn_id)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let slice_rebase = transform::Slice::new(
                self.source.shape().clone(),
                self.rebase.invert_bounds(bounds),
            )?;
            let slice = SparseSlice {
                source: self.source.clone(),
                rebase: slice_rebase,
            };
            Arc::new(slice).filled(txn_id).await
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

impl SparseAccessor for SparseTable {
    fn filled<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rows = self.table.clone().stream(txn_id).await?;
            let filled = rows.map(unwrap_row);
            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
        Box::pin(async move {
            let columns: Vec<ValueId> = axes.iter().map(|x| (*x).into()).collect();
            let filled_at = self
                .table
                .group_by(txn_id.clone(), columns)
                .await?
                .stream(txn_id)
                .await?
                .map(|coord| unwrap_coord(&coord));

            let filled_at: TCStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, txn_id: TxnId) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move { self.table.count(txn_id).await })
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let source = slice_table(self.table.clone().into(), &txn_id, &bounds).await?;
            let filled_in = source.stream(txn_id).await?.map(unwrap_row);
            let filled_in: SparseStream = Box::pin(filled_in);
            Ok(filled_in)
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
    pub fn filled(&'_ self, txn_id: TxnId) -> TCBoxTryFuture<'_, SparseStream> {
        self.accessor.clone().filled(txn_id)
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

#[async_trait]
impl TensorBoolean for SparseTensor {
    async fn all(&self, txn_id: TxnId) -> TCResult<bool> {
        let filled_count = self.accessor.clone().filled_count(txn_id).await?;
        Ok(filled_count == self.size())
    }

    fn any(&'_ self, txn_id: TxnId) -> TCBoxTryFuture<'_, bool> {
        Box::pin(async move {
            let mut filled = self.accessor.clone().filled(txn_id).await?;
            Ok(filled.next().await.is_some())
        })
    }

    async fn and(&self, other: &Self) -> TCResult<Self> {
        let accessor = SparseCombinator::new(
            self.accessor.clone(),
            other.accessor.clone(),
            combinator::and,
        )
        .map(Arc::new)?;

        Ok(SparseTensor { accessor })
    }

    async fn not(&self) -> TCResult<Self> {
        Err(error::unsupported(ERR_NOT_SPARSE))
    }

    async fn or(&self, other: &Self) -> TCResult<Self> {
        let accessor = SparseCombinator::new(
            self.accessor.clone(),
            other.accessor.clone(),
            combinator::or,
        )
        .map(Arc::new)?;

        Ok(SparseTensor { accessor })
    }

    async fn xor(&self, _other: &Self) -> TCResult<Self> {
        Err(error::unsupported(ERR_NOT_SPARSE))
    }
}

impl TensorIO for SparseTensor {
    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        self.accessor.read_value(txn_id, coord)
    }

    fn write_value(
        &'_ self,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCBoxTryFuture<'_, ()> {
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

    fn reshape(&self, shape: Shape) -> TCResult<Self> {
        if &shape == self.shape() {
            return Ok(self.clone());
        }

        let _rebase = transform::Reshape::new(self.shape().clone(), shape);

        Err(error::not_implemented())
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

fn group_axes<'a>(
    accessor: Arc<dyn SparseAccessor>,
    txn_id: TxnId,
    axes: Vec<usize>,
) -> TCBoxTryFuture<'a, TCStream<Vec<u64>>> {
    Box::pin(async move {
        if axes.len() > accessor.ndim() {
            let axes: Vec<String> = axes.iter().map(|x| x.to_string()).collect();
            return Err(error::bad_request("Axis out of bounds", axes.join(", ")));
        }

        let axes_clone = axes.to_vec();
        let left = accessor
            .clone()
            .filled(txn_id.clone())
            .await?
            .map(move |(coord, _)| axes_clone.iter().map(|x| coord[*x]).collect::<Vec<u64>>());

        let mut right = accessor
            .clone()
            .filled(txn_id)
            .await?
            .map(move |(coord, _)| axes.iter().map(|x| coord[*x]).collect::<Vec<u64>>());

        if right.next().await.is_none() {
            let filled_at: TCStream<Vec<u64>> = Box::pin(stream::empty());
            return Ok(filled_at);
        }

        let filled_at = left.zip(right).filter_map(|(l, r)| {
            if l == r {
                future::ready(Some(l))
            } else {
                future::ready(None)
            }
        });

        let filled_at: TCStream<Vec<u64>> = Box::pin(filled_at);
        Ok(filled_at)
    })
}

fn slice_table<'a>(
    mut table: Table,
    txn_id: &'a TxnId,
    bounds: &'a Bounds,
) -> TCBoxTryFuture<'a, Table> {
    use AxisBounds::*;

    Box::pin(async move {
        for (axis, axis_bound) in bounds.to_vec().into_iter().enumerate() {
            let axis: ValueId = axis.into();
            table = match axis_bound {
                At(x) => {
                    let column_bound = table::schema::ColumnBound::Is(u64_to_value(x));
                    table
                        .slice(txn_id, iter::once((axis, column_bound)).collect())
                        .await?
                }
                In(range) => {
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
