use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::ops::Bound;
use std::sync::Arc;

use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;

use crate::error;
use crate::state::btree;
use crate::state::table::{self, Selection, Table, TableBase};
use crate::transaction::{Txn, TxnId};
use crate::value::class::{Instance, NumberClass, NumberInstance, NumberType, UIntType, ValueType};
use crate::value::{Number, TCBoxTryFuture, TCResult, TCTryStream, UInt, Value, ValueId};

use super::bounds::{AxisBounds, Bounds, Shape};
use super::*;

mod combine;

use combine::SparseCombine;

pub type SparseRow = (Vec<u64>, Number);
pub type SparseStream = TCTryStream<SparseRow>;

const ERR_NONBIJECTIVE_WRITE: &str = "Cannot write to a derived Tensor which is not a \
bijection of its source. Consider copying first, or writing directly to the source Tensor.";

const ERR_NOT_SPARSE: &str = "The result of the requested operation would not be sparse;\
convert to a DenseTensor first.";

const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

trait SparseAccessor: TensorView + 'static {
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream>;

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>>;

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64>;

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
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
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let source = self.source.clone();
            let values = source.value_stream(txn).await?;

            let zero = self.dtype().zero();
            let filled = stream::iter(Bounds::all(self.shape()).affected())
                .zip(values)
                .map(|(coord, r)| r.map(|value| (coord, value)))
                .try_filter(move |(_, value)| future::ready(value != &zero));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        _txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        Box::pin(async move {
            let shape = self.shape();
            let filled_at = stream::iter(
                Bounds::all(&Shape::from(
                    axes.iter().map(|x| shape[*x]).collect::<Vec<u64>>(),
                ))
                .affected(),
            )
            .map(|at| Ok(at));

            let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            self.source
                .value_stream(txn)
                .await?
                .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
                .await
        })
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        match self.source.slice(bounds) {
            Ok(source) => {
                let slice = Arc::new(DenseAccessor { source });
                slice.filled(txn)
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
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let filled = self
                .source
                .clone()
                .filled(txn)
                .await?
                .map_ok(move |(coord, value)| self.broadcast(coord, value).map(Ok))
                .try_flatten();

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            let filled = self.source.clone().filled(txn).await?;
            let rebase = self.rebase.clone();
            filled
                .try_fold(0u64, |count, (coord, _)| {
                    future::ready(Ok(count + rebase.map_bounds(coord.into()).size()))
                })
                .await
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let filled_in = self
                .source
                .clone()
                .filled_in(txn, bounds)
                .await?
                .map_ok(move |(coord, value)| self.broadcast(coord, value).map(Ok))
                .try_flatten();

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
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
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
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let dtype = self.dtype;
            let filled = self.source.clone().filled(txn).await?;
            let cast = filled.map_ok(move |(coord, value)| (coord, value.into_type(dtype)));
            let cast: SparseStream = Box::pin(cast);
            Ok(cast)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        self.source.clone().filled_at(txn, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        self.source.clone().filled_count(txn)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let dtype = self.dtype;
            let source = self.source.clone().filled_in(txn, bounds).await?;
            let filled_in = source.map_ok(move |(coord, value)| (coord, value.into_type(dtype)));
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
    left_zero: Number,
    right: Arc<dyn SparseAccessor>,
    right_zero: Number,
    combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
}

impl SparseCombinator {
    fn new(
        left: Arc<dyn SparseAccessor>,
        right: Arc<dyn SparseAccessor>,
        combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<SparseCombinator> {
        if left.shape() != right.shape() {
            return Err(error::internal(
                "Tried to combine SparseTensors with different shapes",
            ));
        }

        let left_zero = left.dtype().zero();
        let right_zero = right.dtype().zero();
        Ok(SparseCombinator {
            left,
            left_zero,
            right,
            right_zero,
            combinator,
            dtype,
        })
    }

    fn filled_inner(&self, left: SparseStream, right: SparseStream) -> SparseStream {
        let combinator = self.combinator;
        let left_zero = self.left_zero.clone();
        let right_zero = self.right_zero.clone();

        let combined = SparseCombine::new(left, right).try_filter_map(move |(coord, l, r)| {
            let l = l.unwrap_or_else(|| left_zero.clone());
            let r = r.unwrap_or_else(|| right_zero.clone());
            let value = combinator(l, r);
            let row = if value == value.class().zero() {
                None
            } else {
                Some((coord, value))
            };

            future::ready(Ok(row))
        });

        Box::pin(combined)
    }
}

impl TensorView for SparseCombinator {
    fn dtype(&self) -> NumberType {
        self.dtype
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
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let left = self.left.clone().filled(txn.clone());
            let right = self.right.clone().filled(txn);
            let (left, right) = try_join!(left, right)?;
            Ok(self.filled_inner(left, right))
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            let count = self
                .filled(txn)
                .await?
                .fold(0u64, |count, _| future::ready(count + 1))
                .await;

            Ok(count)
        })
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let left = self.left.clone().filled_in(txn.clone(), bounds.clone());
            let right = self.right.clone().filled_in(txn, bounds);
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
            Ok(combinator(left, right))
        })
    }

    fn write_value<'a>(
        &'a self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
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
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let filled = self
                .source
                .clone()
                .filled(txn)
                .await?
                .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        self.source.clone().filled_count(txn)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let filled_in = self
                .source
                .clone()
                .filled_in(txn, bounds)
                .await?
                .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

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

struct SparseReduce {
    source: Arc<dyn SparseAccessor>,
    shape: Shape,
    axis: usize,
    reductor: fn(&SparseTensor, Arc<Txn>) -> TCBoxTryFuture<Number>,
}

impl TensorView for SparseReduce {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim() - 1
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.shape.size()
    }
}

impl SparseAccessor for SparseReduce {
    fn filled<'a>(self: Arc<Self>, _txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(future::ready(Err(error::not_implemented())))
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        if axes.is_empty() {
            let filled_at: TCTryStream<Vec<u64>> = Box::pin(stream::empty());
            return Box::pin(future::ready(Ok(filled_at)));
        } else if axes.iter().cloned().fold(axes[0], Ord::max) < self.axis {
            return self.source.clone().filled_at(txn, axes);
        }

        Box::pin(async move {
            let source_axes: Vec<usize> = axes
                .iter()
                .cloned()
                .map(|x| if x < self.axis { x } else { x + 1 })
                .chain(iter::once(self.axis))
                .collect();

            let left = self
                .source
                .clone()
                .filled_at(txn.clone(), source_axes.to_vec())
                .await?;
            let mut right = self.source.clone().filled_at(txn, source_axes).await?;

            if right.next().await.is_none() {
                let filled_at: TCTryStream<Vec<u64>> = Box::pin(stream::empty());
                return Ok(filled_at);
            }

            let filled_at = left
                .zip(right)
                .map(|(lr, rr)| Ok((lr?, rr?)))
                .map_ok(|(mut l, mut r)| {
                    l.pop();
                    r.pop();
                    (l, r)
                })
                .try_filter_map(|(l, r)| {
                    let row = if l != r { Some(l) } else { None };

                    future::ready(Ok(row))
                });

            let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, _txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        Box::pin(future::ready(Err(error::not_implemented())))
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        _txn: Arc<Txn>,
        _bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(future::ready(Err(error::not_implemented())))
    }

    fn read_value<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        _coord: &'a [u64],
    ) -> TCBoxTryFuture<'a, Number> {
        Box::pin(future::ready(Err(error::not_implemented())))
    }

    fn write_value<'a>(
        &'a self,
        _txn_id: TxnId,
        _coord: Vec<u64>,
        _value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(future::ready(Err(error::not_implemented())))
    }
}

struct SparseReshape {
    source: Arc<dyn SparseAccessor>,
    rebase: transform::Reshape,
}

impl TensorView for SparseReshape {
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
        self.source.size()
    }
}

#[async_trait]
impl SparseAccessor for SparseReshape {
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let filled = self
                .source
                .clone()
                .filled(txn)
                .await?
                .map_ok(move |(coord, value)| (rebase.map_coord(coord), value));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        self.source.clone().filled_count(txn)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            if self.source.ndim() == 1 {
                let (start, end) = self.rebase.offsets(&bounds);

                let rebase = transform::Slice::new(
                    self.source.shape().clone(),
                    vec![AxisBounds::from(start..end)].into(),
                )?;

                let slice = Arc::new(SparseSlice {
                    source: self.source.clone(),
                    rebase,
                });

                let rebase = self.rebase.clone();
                let filled = slice
                    .filled(txn)
                    .await?
                    .map_ok(move |(coord, value)| (rebase.map_coord(coord), value))
                    .try_filter(move |(coord, _)| future::ready(bounds.contains_coord(coord)));

                let filled: SparseStream = Box::pin(filled);
                Ok(filled)
            } else {
                let rebase = transform::Reshape::new(
                    self.source.shape().clone(),
                    vec![self.source.size()].into(),
                )?;
                let flat = Arc::new(SparseReshape {
                    source: self.source.clone(),
                    rebase,
                });

                let rebase = transform::Reshape::new(flat.shape().clone(), self.shape().clone())?;
                let unflat = Arc::new(SparseReshape {
                    source: flat,
                    rebase,
                });
                unflat.filled_in(txn, bounds).await
            }
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
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let filled = self
                .source
                .clone()
                .filled_in(txn, rebase.bounds().clone())
                .await?
                .map_ok(move |(coord, value)| (rebase.map_coord(coord), value));

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        group_axes(self, txn, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move {
            let count = self
                .filled(txn)
                .await?
                .fold(0u64, |count, _| future::ready(count + 1))
                .await;

            Ok(count)
        })
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let bounds = self.rebase.invert_bounds(bounds);
            let filled_in = self
                .source
                .clone()
                .filled_in(txn, bounds)
                .await?
                .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

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
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let ndim = self.ndim();
            let txn_id = txn.id().clone();
            let filled = self
                .clone()
                .filled_at(txn, (0..ndim).collect())
                .await?
                .and_then(move |coord| {
                    self.clone()
                        .read_value_owned(txn_id.clone(), coord.to_vec())
                        .map_ok(|value| (coord, value))
                });

            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        // can't use group_axes here because it would lead to a circular dependency in self.filled
        Box::pin(async move {
            let rebase = self.rebase.clone();
            let filled_at = self
                .source
                .clone()
                .filled_at(txn, rebase.invert_axes(&axes))
                .await?
                .map_ok(move |coord| rebase.map_coord_axes(coord, &axes));

            let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        self.source.clone().filled_count(txn)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let filled_in = self
                .source
                .clone()
                .filled_in(txn, self.rebase.invert_bounds(bounds))
                .await?
                .map_ok(move |(coord, value)| (self.rebase.map_coord(coord), value));

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
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let rows = self.table.clone().stream(txn.id().clone()).await?;
            let filled = rows.map(unwrap_row);
            let filled: SparseStream = Box::pin(filled);
            Ok(filled)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        Box::pin(async move {
            let columns: Vec<ValueId> = axes.iter().map(|x| (*x).into()).collect();
            let filled_at = self
                .table
                .group_by(txn.id().clone(), columns)
                .await?
                .stream(txn.id().clone())
                .await?
                .map(|coord| unwrap_coord(&coord));

            let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
            Ok(filled_at)
        })
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        Box::pin(async move { self.table.count(txn.id().clone()).await })
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let source = slice_table(self.table.clone().into(), txn.id(), &bounds).await?;
            let filled_in = source.stream(txn.id().clone()).await?.map(unwrap_row);
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

struct SparseUnary {
    source: Arc<dyn SparseAccessor>,
    transform: fn(Number) -> Number,
    dtype: NumberType,
}

impl TensorView for SparseUnary {
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

impl SparseAccessor for SparseUnary {
    fn filled<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let transform = self.transform;
            let filled = self.source.clone().filled(txn).await?;
            let cast = filled.map_ok(move |(coord, value)| (coord, transform(value)));
            let cast: SparseStream = Box::pin(cast);
            Ok(cast)
        })
    }

    fn filled_at<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        axes: Vec<usize>,
    ) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
        self.source.clone().filled_at(txn, axes)
    }

    fn filled_count<'a>(self: Arc<Self>, txn: Arc<Txn>) -> TCBoxTryFuture<'a, u64> {
        self.source.clone().filled_count(txn)
    }

    fn filled_in<'a>(
        self: Arc<Self>,
        txn: Arc<Txn>,
        bounds: Bounds,
    ) -> TCBoxTryFuture<'a, SparseStream> {
        Box::pin(async move {
            let transform = self.transform;
            let source = self.source.clone().filled_in(txn, bounds).await?;
            let filled_in = source.map_ok(move |(coord, value)| (coord, transform(value)));
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

#[derive(Clone)]
pub struct SparseTensor {
    accessor: Arc<dyn SparseAccessor>,
}

impl SparseTensor {
    pub fn filled(&'_ self, txn: Arc<Txn>) -> TCBoxTryFuture<'_, SparseStream> {
        self.accessor.clone().filled(txn)
    }

    pub fn from_dense(source: DenseTensor) -> SparseTensor {
        let accessor = Arc::new(DenseAccessor { source });
        SparseTensor { accessor }
    }

    fn combine(
        &self,
        other: &Self,
        combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<Self> {
        let (this, that) = broadcast(self, other)?;

        let accessor = SparseCombinator::new(
            this.accessor.clone(),
            that.accessor.clone(),
            combinator,
            dtype,
        )
        .map(Arc::new)?;

        Ok(SparseTensor { accessor })
    }

    async fn condense(
        &self,
        other: &Self,
        txn: Arc<Txn>,
        default: Number,
        condensor: fn(Number, Number) -> Number,
    ) -> TCResult<DenseTensor> {
        let (this, that) = broadcast(self, other)?;

        let accessor = SparseCombinator::new(
            this.accessor.clone(),
            that.accessor.clone(),
            condensor,
            default.class(),
        )
        .map(Arc::new)?;

        let condensed = DenseTensor::constant(txn.clone(), this.shape().clone(), default).await?;

        let txn_id = txn.id().clone();
        accessor
            .filled(txn)
            .await?
            .map_ok(|(coord, value)| condensed.write_value_at(txn_id.clone(), coord, value))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(condensed)
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
    async fn all(&self, txn: Arc<Txn>) -> TCResult<bool> {
        let filled_count = self.accessor.clone().filled_count(txn).await?;
        Ok(filled_count == self.size())
    }

    fn any(&'_ self, txn: Arc<Txn>) -> TCBoxTryFuture<'_, bool> {
        Box::pin(async move {
            let mut filled = self.accessor.clone().filled(txn).await?;
            Ok(filled.next().await.is_some())
        })
    }

    fn and(&self, other: &Self) -> TCResult<Self> {
        self.combine(other, Number::and, NumberType::Bool)
    }

    fn not(&self) -> TCResult<Self> {
        Err(error::unsupported(ERR_NOT_SPARSE))
    }

    fn or(&self, other: &Self) -> TCResult<Self> {
        self.combine(other, Number::or, NumberType::Bool)
    }

    fn xor(&self, _other: &Self) -> TCResult<Self> {
        Err(error::unsupported(ERR_NOT_SPARSE))
    }
}

#[async_trait]
impl TensorCompare for SparseTensor {
    async fn eq(&self, other: &Self, txn: Arc<Txn>) -> TCResult<DenseTensor> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::eq)
            .await
    }

    fn gt(&self, other: &Self) -> TCResult<Self> {
        self.combine(other, <Number as NumberInstance>::gt, NumberType::Bool)
    }

    async fn gte(&self, other: &Self, txn: Arc<Txn>) -> TCResult<DenseTensor> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::gte)
            .await
    }

    fn lt(&self, other: &Self) -> TCResult<Self> {
        self.combine(other, <Number as NumberInstance>::lt, NumberType::Bool)
    }

    async fn lte(&self, other: &Self, txn: Arc<Txn>) -> TCResult<DenseTensor> {
        self.condense(other, txn, true.into(), <Number as NumberInstance>::lte)
            .await
    }

    fn ne(&self, other: &Self) -> TCResult<Self> {
        self.combine(other, <Number as NumberInstance>::ne, NumberType::Bool)
    }
}

impl TensorIO for SparseTensor {
    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        self.accessor.read_value(txn_id, coord)
    }

    fn write<'a>(&'a self, txn: Arc<Txn>, bounds: Bounds, other: Self) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let slice = self.slice(bounds)?;
            let other = other
                .broadcast(slice.shape().clone())?
                .as_type(self.dtype())?;

            let txn_id = txn.id().clone();
            other
                .filled(txn)
                .await?
                .map_ok(|(coord, value)| slice.write_value_at(txn_id.clone(), coord, value))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
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

impl TensorMath for SparseTensor {
    fn abs(&self) -> TCResult<Self> {
        let is_abs = match self.dtype() {
            NumberType::Bool => true,
            NumberType::UInt(_) => true,
            _ => false,
        };

        if is_abs {
            return Ok(self.clone());
        }

        let accessor = Arc::new(SparseUnary {
            source: self.accessor.clone(),
            transform: <Number as NumberInstance>::abs,
            dtype: NumberType::Bool,
        });

        Ok(SparseTensor { accessor })
    }

    fn add(&self, other: &Self) -> TCResult<Self> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, <Number as NumberInstance>::add, dtype)
    }

    fn multiply(&self, other: &Self) -> TCResult<Self> {
        let dtype = Ord::max(self.dtype(), other.dtype());
        self.combine(other, <Number as NumberInstance>::multiply, dtype)
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

        let rebase = transform::Reshape::new(self.shape().clone(), shape)?;
        let accessor = Arc::new(SparseReshape {
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

#[async_trait]
impl TensorUnary for SparseTensor {
    fn product(&self, _txn: Arc<Txn>, axis: usize) -> TCResult<Self> {
        if axis >= self.ndim() {
            return Err(error::bad_request(
                &format!("Tensor with shape {} has no such axis", self.shape()),
                axis,
            ));
        }

        Err(error::not_implemented())
    }

    async fn product_all(&self, txn: Arc<Txn>) -> TCResult<Number> {
        if !self.all(txn.clone()).await? {
            return Ok(self.dtype().zero());
        }

        self.accessor
            .clone()
            .filled(txn)
            .await?
            .map_ok(|(_, value)| value)
            .try_fold(self.dtype().one(), |product, value| {
                future::ready(Ok(product * value))
            })
            .await
    }

    fn sum(&self, _txn: Arc<Txn>, axis: usize) -> TCResult<Self> {
        if axis >= self.ndim() {
            return Err(error::bad_request(
                &format!("Tensor with shape {} has no such axis", self.shape()),
                axis,
            ));
        }

        Err(error::not_implemented())
    }

    async fn sum_all(&self, txn: Arc<Txn>) -> TCResult<Number> {
        self.accessor
            .clone()
            .filled(txn)
            .await?
            .map_ok(|(_, value)| value)
            .try_fold(self.dtype().one(), |product, value| {
                future::ready(Ok(product + value))
            })
            .await
    }
}

fn group_axes<'a>(
    accessor: Arc<dyn SparseAccessor>,
    txn: Arc<Txn>,
    axes: Vec<usize>,
) -> TCBoxTryFuture<'a, TCTryStream<Vec<u64>>> {
    Box::pin(async move {
        if axes.len() > accessor.ndim() {
            let axes: Vec<String> = axes.iter().map(|x| x.to_string()).collect();
            return Err(error::bad_request("Axis out of bounds", axes.join(", ")));
        }

        let axes_clone = axes.to_vec();
        let map = move |(coord, _): (Vec<u64>, Number)| {
            axes_clone.iter().map(|x| coord[*x]).collect::<Vec<u64>>()
        };

        let sorted_axes: Vec<usize> = itertools::sorted(axes.to_vec()).collect::<Vec<usize>>();
        let (left, mut right): (TCTryStream<Vec<u64>>, TCTryStream<Vec<u64>>) = if axes
            == sorted_axes
        {
            let left = accessor
                .clone()
                .filled(txn.clone())
                .await?
                .map_ok(map.clone());
            let right = accessor.clone().filled(txn.clone()).await?.map_ok(map);
            (Box::pin(left), Box::pin(right))
        } else {
            let schema: btree::Schema = axes
                .iter()
                .cloned()
                .map(ValueId::from)
                .map(|x| (x, ValueType::Number(NumberType::UInt(UIntType::U64)), None).into())
                .collect::<Vec<btree::Column>>()
                .into();

            let btree_file = txn
                .clone()
                .subcontext_tmp()
                .await?
                .context()
                .create_btree(txn.id().clone(), "axes".parse()?)
                .await?;
            let btree = Arc::new(btree::BTree::create(txn.id().clone(), schema, btree_file).await?);

            btree
                .try_insert_from(
                    txn.id(),
                    accessor
                        .clone()
                        .filled(txn.clone())
                        .await?
                        .map_ok(map)
                        .map_ok(|mut coord| {
                            coord
                                .drain(..)
                                .map(|i| Number::UInt(i.into()))
                                .map(|n| n.into())
                                .collect::<Vec<Value>>()
                        }),
                )
                .await?;

            let left = btree
                .clone()
                .slice(txn.id().clone(), btree::Selector::all())
                .await?
                .map(move |coord| unwrap_coord(&coord));

            let right = btree
                .slice(txn.id().clone(), btree::Selector::all())
                .await?
                .map(move |coord| unwrap_coord(&coord));

            (Box::pin(left), Box::pin(right))
        };

        if right.next().await.is_none() {
            let filled_at: TCTryStream<Vec<u64>> = Box::pin(stream::empty());
            return Ok(filled_at);
        }

        let filled_at = left
            .zip(right)
            .map(|(lr, rr)| Ok((lr?, rr?)))
            .try_filter_map(|(l, r)| {
                if l == r {
                    future::ready(Ok(Some(l)))
                } else {
                    future::ready(Ok(None))
                }
            });

        let filled_at: TCTryStream<Vec<u64>> = Box::pin(filled_at);
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
