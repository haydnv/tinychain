use std::convert::TryInto;

use afarray::Array;
use futures::{future, Future, StreamExt, TryFutureExt, TryStreamExt};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_btree::Node;
use tc_error::*;
use tc_tensor::*;
use tc_transact::fs::{CopyFrom, Dir};
use tc_transact::Transaction;
use tc_value::{Bound, Number, NumberClass, Range, TCString, Value, ValueType};
use tcgeneric::{label, PathSegment, TCBoxTryFuture, Tuple};

use crate::collection::{Collection, DenseTensor, DenseTensorFile, SparseTensor, Tensor};
use crate::fs;
use crate::route::{AttributeHandler, GetHandler, PostHandler, PutHandler};
use crate::state::State;
use crate::stream::TCStream;
use crate::txn::Txn;

use super::{Handler, Route};

struct CastHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for CastHandler<T>
where
    T: TensorTransform + Send + Sync + 'a,
    Tensor: From<T::Cast>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let dtype =
                    ValueType::try_cast_from(key, |v| TCError::bad_request("not a NumberType", v))?;

                let dtype = dtype.try_into()?;
                self.tensor
                    .cast_into(dtype)
                    .map(Tensor::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for CastHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct ConstantHandler;

impl<'a> Handler<'a> for ConstantHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let (shape, value): (Vec<u64>, Number) =
                    key.try_cast_into(|v| TCError::bad_request("invalid Tensor schema", v))?;

                constant(&txn, shape.into(), value)
                    .map_ok(Tensor::from)
                    .map_ok(Collection::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

struct CopyHandler;

impl<'a> Handler<'a> for CopyHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let source = params.require(&label("tensor").into())?;
                params.expect_empty()?;

                let copy = match source {
                    Tensor::Dense(source) => {
                        let file = txn
                            .context()
                            .create_file_tmp(*txn.id(), TensorType::Dense)
                            .await?;

                        let blocks =
                            BlockListFile::copy_from(source.into_inner(), file, txn).await?;

                        DenseTensor::from(blocks.accessor()).into()
                    }
                    Tensor::Sparse(source) => {
                        let dir = txn.context().create_dir_tmp(*txn.id()).await?;
                        let table = SparseTable::copy_from(source, dir, txn).await?;
                        SparseTensor::from(table.accessor()).into()
                    }
                };

                Ok(State::Collection(Collection::Tensor(copy)))
            })
        }))
    }
}

struct CopyDenseHandler;

impl<'a> Handler<'a> for CopyDenseHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let Schema { dtype, shape } =
                    schema.try_cast_into(|v| TCError::bad_request("invalid Tensor schema", v))?;

                let source: TCStream = params.require(&label("source").into())?;
                params.expect_empty()?;

                let elements = source.into_stream(txn.clone()).await?;
                let elements = elements.map(|r| {
                    r.and_then(|n| {
                        Number::try_cast_from(n, |n| {
                            TCError::bad_request("invalid Tensor element", n)
                        })
                    })
                });

                let txn_id = *txn.id();
                let file = create_file(txn).await?;
                DenseTensorFile::from_values(file, txn_id, shape, dtype, elements)
                    .map_ok(DenseTensor::from)
                    .map_ok(Collection::from)
                    .map_ok(State::Collection)
                    .await
            })
        }))
    }
}

struct CopySparseHandler;

impl<'a> Handler<'a> for CopySparseHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let schema: Schema =
                    schema.try_cast_into(|v| TCError::bad_request("invalid Tensor schema", v))?;

                let source: TCStream = params.require(&label("source").into())?;
                params.expect_empty()?;

                let elements = source.into_stream(txn.clone()).await?;

                let txn_id = *txn.id();
                let dir = txn.context().create_dir_tmp(txn_id).await?;
                let tensor = SparseTensor::create(&dir, schema, txn_id).await?;

                let elements = elements
                    .map(|r| {
                        r.and_then(|state| {
                            Value::try_cast_from(state, |s| {
                                TCError::bad_request("invalid sparse Tensor element", s)
                            })
                        })
                    })
                    .map(|r| {
                        r.and_then(|row| {
                            row.try_cast_into(|v| {
                                TCError::bad_request(
                                    "sparse Tensor expected a (Coord, Number) tuple, found",
                                    v,
                                )
                            })
                        })
                    });

                elements
                    .map_ok(|(coord, value)| tensor.write_value_at(txn_id, coord, value))
                    .try_buffer_unordered(num_cpus::get())
                    .try_fold((), |(), ()| future::ready(Ok(())))
                    .await?;

                Ok(Collection::Tensor(tensor.into()).into())
            })
        }))
    }
}

struct CreateHandler {
    class: TensorType,
}

impl<'a> Handler<'a> for CreateHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let schema: Schema =
                    key.try_cast_into(|v| TCError::bad_request("invalid Tensor schema", v))?;

                match self.class {
                    TensorType::Dense => {
                        constant(&txn, schema.shape, schema.dtype.zero())
                            .map_ok(Tensor::from)
                            .map_ok(Collection::Tensor)
                            .map_ok(State::Collection)
                            .await
                    }
                    TensorType::Sparse => {
                        let txn_id = *txn.id();
                        let dir = txn.context().create_dir_tmp(txn_id).await?;

                        SparseTensor::create(&dir, schema, txn_id)
                            .map_ok(Tensor::from)
                            .map_ok(Collection::Tensor)
                            .map_ok(State::Collection)
                            .await
                    }
                }
            })
        }))
    }
}

struct EinsumHandler;

impl<'a> Handler<'a> for EinsumHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let format: TCString = params.require(&label("format").into())?;
                let tensors: Vec<Tensor> = params.require(&label("tensors").into())?;
                einsum(&format, tensors)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

struct ElementsHandler<T> {
    tensor: T,
}

impl<T> ElementsHandler<T> {
    fn new(tensor: T) -> Self {
        Self { tensor }
    }
}

impl<'a, T> Handler<'a> for ElementsHandler<T>
where
    T: TensorAccess + TensorTransform + Send + Sync + 'a,
    Tensor: From<T> + From<T::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let tensor = if key.is_none() {
                    Tensor::from(self.tensor)
                } else {
                    let bounds = cast_bounds(self.tensor.shape(), key)?;
                    let slice = self.tensor.slice(bounds)?;
                    Tensor::from(slice)
                };

                Ok(TCStream::from(Collection::Tensor(tensor)).into())
            })
        }))
    }
}

struct ExpandHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for ExpandHandler<T>
where
    T: TensorTransform + Send + 'a,
    Tensor: From<T::Expand>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let axis = key.try_cast_into(|v| TCError::bad_request("invalid tensor axis", v))?;

                self.tensor
                    .expand_dims(axis)
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for ExpandHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct RangeHandler;

impl<'a> Handler<'a> for RangeHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<(Vec<u64>, Number, Number)>() {
                    let (shape, start, stop): (Vec<u64>, Number, Number) =
                        key.opt_cast_into().unwrap();

                    let file = create_file(&txn).await?;

                    DenseTensor::range(file, *txn.id(), shape, start, stop)
                        .map_ok(Tensor::from)
                        .map_ok(Collection::from)
                        .map_ok(State::from)
                        .await
                } else {
                    Err(TCError::bad_request("invalid schema for range tensor", key))
                }
            })
        }))
    }
}

struct TransposeHandler<T> {
    tensor: T,
}

impl<'a, T> Handler<'a> for TransposeHandler<T>
where
    T: TensorTransform + Send + 'a,
    Tensor: From<T::Transpose>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let transpose = if key.is_none() {
                    self.tensor.transpose(None)
                } else {
                    let permutation = key.try_cast_into(|v| {
                        TCError::bad_request("invalid permutation for transpose", v)
                    })?;

                    self.tensor.transpose(Some(permutation))
                };

                transpose
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for TransposeHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

impl Route for TensorType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(CreateHandler { class: *self }));
        } else if path.len() != 1 {
            return None;
        }

        match self {
            Self::Dense => match path[0].as_str() {
                "copy_from" => Some(Box::new(CopyDenseHandler)),
                "constant" => Some(Box::new(ConstantHandler)),
                "range" => Some(Box::new(RangeHandler)),
                _ => None,
            },
            Self::Sparse => match path[0].as_str() {
                "copy_from" => Some(Box::new(CopySparseHandler)),
                _ => None,
            },
        }
    }
}

struct DualHandler {
    tensor: Tensor,
    op: fn(Tensor, Tensor) -> TCResult<Tensor>,
}

impl DualHandler {
    fn new<T>(tensor: T, op: fn(Tensor, Tensor) -> TCResult<Tensor>) -> Self
    where
        Tensor: From<T>,
    {
        Self {
            tensor: tensor.into(),
            op,
        }
    }
}

impl<'a> Handler<'a> for DualHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let l = self.tensor;
                let r: Tensor = params.require(&label("r").into())?;
                params.expect_empty()?;

                if l.shape() == r.shape() {
                    debug!("tensor dual op with shapes {} {}", l.shape(), r.shape());
                    (self.op)(l, r).map(Collection::from).map(State::from)
                } else {
                    let (l, r) = broadcast(l, r)?;
                    debug!("tensor dual op with shapes {} {}", l.shape(), r.shape());
                    (self.op)(l, r).map(Collection::from).map(State::from)
                }
            })
        }))
    }
}

struct ReduceHandler<'a, T: TensorReduce<fs::Dir>> {
    tensor: &'a T,
    reduce: fn(T, usize) -> TCResult<<T as TensorReduce<fs::Dir>>::Reduce>,
    reduce_all: fn(&'a T, Txn) -> TCBoxTryFuture<'a, Number>,
}

impl<'a, T: TensorReduce<fs::Dir>> ReduceHandler<'a, T> {
    fn new(
        tensor: &'a T,
        reduce: fn(T, usize) -> TCResult<<T as TensorReduce<fs::Dir>>::Reduce>,
        reduce_all: fn(&'a T, Txn) -> TCBoxTryFuture<'a, Number>,
    ) -> Self {
        Self {
            tensor,
            reduce,
            reduce_all,
        }
    }
}

impl<'a, T: TensorReduce<fs::Dir> + Clone + Sync> Handler<'a> for ReduceHandler<'a, T>
where
    Tensor: From<<T as TensorReduce<fs::Dir>>::Reduce>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    (self.reduce_all)(self.tensor, txn.clone())
                        .map_ok(Value::from)
                        .map_ok(State::from)
                        .await
                } else {
                    let axis = key.try_cast_into(|v| TCError::bad_request("invalid axis", v))?;
                    (self.reduce)(self.tensor.clone(), axis)
                        .map(Tensor::from)
                        .map(Collection::from)
                        .map(State::from)
                }
            })
        }))
    }
}

struct TensorHandler<T> {
    tensor: T,
}

impl<'a, T: 'a> Handler<'a> for TensorHandler<T>
where
    T: TensorAccess
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorTransform
        + Clone
        + Send
        + Sync,
    <T as TensorTransform>::Slice: TensorAccess + Send,
    Tensor: From<<T as TensorTransform>::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                debug!("GET Tensor: {}", key);
                let bounds = cast_bounds(self.tensor.shape(), key)?;
                self.tensor
                    .slice(bounds)
                    .map(Tensor::from)
                    .map(Collection::from)
                    .map(State::from)
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            debug!("PUT Tensor: {} <- {}", key, value);
            Box::pin(write(self.tensor, txn, key, value))
        }))
    }
}

impl<T> From<T> for TensorHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct UnaryHandler {
    tensor: Tensor,
    op: fn(&Tensor) -> TCResult<Tensor>,
}

impl UnaryHandler {
    fn new(tensor: Tensor, op: fn(&Tensor) -> TCResult<Tensor>) -> Self {
        Self { tensor, op }
    }
}

impl<'a> Handler<'a> for UnaryHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let tensor = if key.is_none() {
                    self.tensor
                } else {
                    let bounds = cast_bounds(self.tensor.shape(), key.into())?;
                    self.tensor.slice(bounds)?
                };

                (self.op)(&tensor).map(Collection::from).map(State::from)
            })
        }))
    }
}

struct UnaryHandlerAsync<F: Send> {
    tensor: Tensor,
    op: fn(Tensor, Txn) -> F,
}

impl<'a, F: Send> UnaryHandlerAsync<F> {
    fn new(tensor: Tensor, op: fn(Tensor, Txn) -> F) -> Self {
        Self { tensor, op }
    }
}

impl<'a, F> Handler<'a> for UnaryHandlerAsync<F>
where
    F: Future<Output = TCResult<bool>> + Send + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let txn = txn.clone();

                if key.is_none() {
                    (self.op)(self.tensor, txn).map_ok(State::from).await
                } else {
                    let bounds = cast_bounds(self.tensor.shape(), key.into())?;
                    let slice = self.tensor.slice(bounds)?;
                    (self.op)(slice, txn).map_ok(State::from).await
                }
            })
        }))
    }
}

impl<B: DenseWrite<fs::File<Array>, fs::File<Node>, fs::Dir, Txn>> Route for DenseTensor<B> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

impl<A: SparseWrite<fs::File<Array>, fs::File<Node>, fs::Dir, Txn>> Route for SparseTensor<A> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

impl Route for Tensor {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

fn route<'a, T>(tensor: &'a T, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>
where
    T: TensorAccess
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorCompare<Tensor, Compare = Tensor, Dense = Tensor>
        + TensorBoolean<Tensor, Combine = Tensor>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorMath<fs::Dir, Tensor, Combine = Tensor>
        + TensorReduce<fs::Dir, Txn = Txn>
        + TensorTransform
        + TensorUnary<fs::Dir, Txn = Txn>
        + Clone
        + Send
        + Sync,
    Collection: From<T>,
    Tensor: From<T>,
    <T as TensorTransform>::Slice: TensorAccess + Send + 'a,
    Tensor: From<<T as TensorReduce<fs::Dir>>::Reduce>,
    Tensor: From<<T as TensorTransform>::Cast>,
    Tensor: From<<T as TensorTransform>::Expand>,
    Tensor: From<<T as TensorTransform>::Slice>,
    Tensor: From<<T as TensorTransform>::Transpose>,
{
    if path.is_empty() {
        Some(Box::new(TensorHandler::from(tensor.clone())))
    } else if path.len() == 1 {
        match path[0].as_str() {
            // attributes
            "shape" => {
                return Some(Box::new(AttributeHandler::from(
                    tensor
                        .shape()
                        .iter()
                        .map(|dim| Number::from(*dim))
                        .collect::<Tuple<Value>>(),
                )))
            }

            // reduce ops (which require borrowing)
            "product" => {
                return Some(Box::new(ReduceHandler::new(
                    tensor,
                    TensorReduce::product,
                    TensorReduce::product_all,
                )))
            }
            "sum" => {
                return Some(Box::new(ReduceHandler::new(
                    tensor,
                    TensorReduce::sum,
                    TensorReduce::sum_all,
                )))
            }
            _ => {}
        };

        let tensor = tensor.clone();

        match path[0].as_str() {
            // to stream
            "elements" => Some(Box::new(ElementsHandler::new(tensor))),

            // boolean ops
            "and" => Some(Box::new(DualHandler::new(tensor, TensorBoolean::and))),
            "or" => Some(Box::new(DualHandler::new(tensor, TensorBoolean::or))),
            "xor" => Some(Box::new(DualHandler::new(tensor, TensorBoolean::xor))),

            // comparison ops
            "eq" => Some(Box::new(DualHandler::new(tensor, TensorCompare::eq))),
            "gt" => Some(Box::new(DualHandler::new(tensor, TensorCompare::gt))),
            "gte" => Some(Box::new(DualHandler::new(tensor, TensorCompare::gte))),
            "lt" => Some(Box::new(DualHandler::new(tensor, TensorCompare::lt))),
            "lte" => Some(Box::new(DualHandler::new(tensor, TensorCompare::lte))),
            "ne" => Some(Box::new(DualHandler::new(tensor, TensorCompare::ne))),

            // unary ops
            "abs" => Some(Box::new(UnaryHandler::new(tensor.into(), TensorUnary::abs))),
            "all" => Some(Box::new(UnaryHandlerAsync::new(
                tensor.into(),
                TensorUnary::all,
            ))),
            "any" => Some(Box::new(UnaryHandlerAsync::new(
                tensor.into(),
                TensorUnary::any,
            ))),
            "not" => Some(Box::new(UnaryHandler::new(tensor.into(), TensorUnary::not))),

            // basic math
            "add" => Some(Box::new(DualHandler::new(tensor, TensorMath::add))),
            "div" => Some(Box::new(DualHandler::new(tensor, TensorMath::div))),
            "mul" => Some(Box::new(DualHandler::new(tensor, TensorMath::mul))),
            "sub" => Some(Box::new(DualHandler::new(tensor, TensorMath::sub))),

            // transforms
            "cast" => Some(Box::new(CastHandler::from(tensor))),
            "expand_dims" => Some(Box::new(ExpandHandler::from(tensor))),
            "transpose" => Some(Box::new(TransposeHandler::from(tensor))),

            _ => None,
        }
    } else {
        None
    }
}

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return None;
        }

        match path[0].as_str() {
            "dense" => TensorType::Dense.route(&path[1..]),
            "sparse" => TensorType::Sparse.route(&path[1..]),
            "copy_from" if path.len() == 1 => Some(Box::new(CopyHandler)),
            "einsum" if path.len() == 1 => Some(Box::new(EinsumHandler)),
            _ => None,
        }
    }
}

async fn constant(
    txn: &Txn,
    shape: Shape,
    value: Number,
) -> TCResult<DenseTensor<DenseTensorFile>> {
    let file = create_file(txn).await?;
    DenseTensor::constant(file, *txn.id(), shape, value).await
}

async fn write<T>(tensor: T, txn: &Txn, key: Value, value: State) -> TCResult<()>
where
    T: TensorAccess
        + TensorIO<fs::Dir, Txn = Txn>
        + TensorDualIO<fs::Dir, Tensor, Txn = Txn>
        + TensorTransform
        + Clone,
    <T as TensorTransform>::Slice: TensorAccess + Send,
{
    debug!("write {} to {}", value, key);
    let bounds = cast_bounds(tensor.shape(), key)?;

    match value {
        State::Collection(Collection::Tensor(value)) => {
            tensor.write(txn.clone(), bounds, value).await
        }
        State::Scalar(scalar) => {
            let value =
                scalar.try_cast_into(|v| TCError::bad_request("invalid tensor element", v))?;

            tensor.write_value(*txn.id(), bounds, value).await
        }
        other => Err(TCError::bad_request(
            "cannot write this value to tensor",
            other,
        )),
    }
}

async fn create_file(txn: &Txn) -> TCResult<fs::File<afarray::Array>> {
    txn.context()
        .create_file_tmp(*txn.id(), TensorType::Dense)
        .await
}

fn cast_bound(dim: u64, bound: Value) -> TCResult<u64> {
    let bound = i64::try_cast_from(bound, |v| TCError::bad_request("invalid bound", v))?;
    if bound.abs() as u64 > dim {
        return Err(TCError::bad_request(
            format!("Index out of bounds for dimension {}", dim),
            bound,
        ));
    }

    if bound < 0 {
        Ok(dim - bound.abs() as u64)
    } else {
        Ok(bound as u64)
    }
}

fn cast_range(dim: u64, range: Range) -> TCResult<AxisBounds> {
    debug!("cast range from {} with dimension {}", range, dim);

    let start = match range.start {
        Bound::Un => 0,
        Bound::In(start) => cast_bound(dim, start)?,
        Bound::Ex(start) => cast_bound(dim, start)? + 1,
    };

    let end = match range.end {
        Bound::Un => dim,
        Bound::In(end) => cast_bound(dim, end)? + 1,
        Bound::Ex(end) => cast_bound(dim, end)?,
    };

    if end > start {
        Ok(AxisBounds::In(start..end))
    } else {
        Err(TCError::bad_request(
            "invalid range",
            Tuple::from(vec![start, end]),
        ))
    }
}

pub fn cast_bounds(shape: &Shape, value: Value) -> TCResult<Bounds> {
    debug!("tensor bounds from {} (shape is {})", value, shape);

    match value {
        Value::None => Ok(Bounds::all(shape)),
        Value::Number(i) => {
            let bound = cast_bound(shape[0], i.into())?;
            Ok(Bounds::from(vec![bound]))
        }
        Value::Tuple(range) if range.matches::<(Bound, Bound)>() => {
            if shape.is_empty() {
                return Err(TCError::bad_request(
                    "empty Tensor has no valid bounds, but found",
                    range,
                ));
            }

            let range = range.opt_cast_into().unwrap();
            Ok(Bounds::from(vec![cast_range(shape[0], range)?]))
        }
        Value::Tuple(bounds) => {
            if bounds.len() > shape.len() {
                return Err(TCError::unsupported(format!(
                    "tensor of shape {} does not support bounds with {} axes",
                    shape,
                    bounds.len()
                )));
            }

            let mut axes = Vec::with_capacity(shape.len());

            for (axis, bound) in bounds.into_inner().into_iter().enumerate() {
                debug!(
                    "bound for axis {} with dimension {} is {}",
                    axis, shape[axis], bound
                );

                let bound = if bound.is_none() {
                    AxisBounds::all(shape[axis])
                } else if bound.matches::<Range>() {
                    let range = Range::opt_cast_from(bound).unwrap();
                    cast_range(shape[axis], range)?
                } else if bound.matches::<Vec<u64>>() {
                    bound.opt_cast_into().map(AxisBounds::Of).unwrap()
                } else if let Value::Number(value) = bound {
                    cast_bound(shape[axis], value.into()).map(AxisBounds::At)?
                } else {
                    return Err(TCError::bad_request(
                        format!("invalid bound for axis {}", axis),
                        bound,
                    ));
                };

                axes.push(bound);
            }

            Ok(Bounds::from(axes))
        }
        other => Err(TCError::bad_request("invalid tensor bounds", other)),
    }
}
