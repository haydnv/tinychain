use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::{future, TryFutureExt};

use crate::class::{Class, Instance, NativeClass, State, TCBoxTryFuture, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionType};
use crate::error;
use crate::handler::Public;
use crate::request::Request;
use crate::scalar::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::*;
use super::dense::{
    dense_constant, from_sparse, BlockList, BlockListDyn, BlockListFile, DenseTensor,
};
use super::sparse::{self, from_dense, SparseAccess, SparseAccessorDyn, SparseTable, SparseTensor};
use super::{
    IntoView, TensorAccessor, TensorBoolean, TensorCompare, TensorDualIO, TensorIO, TensorMath,
    TensorReduce, TensorTransform, TensorUnary,
};

pub trait TensorInstance:
    Clone + Instance + IntoView + TensorIO + TensorTransform + TensorUnary + Send + Sync
{
}

#[derive(Clone, Eq, PartialEq)]
pub enum TensorType {
    Sparse,
    Dense,
}

impl TensorType {
    async fn constant(&self, txn: &Txn, shape: Shape, number: Number) -> TCResult<Tensor> {
        match self {
            Self::Dense => {
                dense_constant(txn, shape, number)
                    .map_ok(DenseTensor::into_dyn)
                    .map_ok(Tensor::Dense)
                    .await
            }
            Self::Sparse => {
                let tensor = self.zeros(txn, number.class(), shape).await?;
                tensor
                    .write_value(*txn.id(), Bounds::all(tensor.shape()), number)
                    .await?;
                Ok(tensor)
            }
        }
    }

    async fn zeros(&self, txn: &Txn, dtype: NumberType, shape: Shape) -> TCResult<Tensor> {
        match self {
            Self::Dense => {
                dense_constant(txn, shape, dtype.zero())
                    .map_ok(DenseTensor::into_dyn)
                    .map_ok(Tensor::Dense)
                    .await
            }
            Self::Sparse => {
                sparse::create(txn, shape, dtype)
                    .map_ok(SparseTensor::into_dyn)
                    .map_ok(Tensor::Sparse)
                    .await
            }
        }
    }
}

impl Class for TensorType {
    type Instance = Tensor;
}

impl NativeClass for TensorType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "dense" => Ok(TensorType::Dense),
                "sparse" => Ok(TensorType::Sparse),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        CollectionType::prefix().append(label("tensor"))
    }
}

#[async_trait]
impl CollectionClass for TensorType {
    type Instance = Tensor;

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<Tensor> {
        if schema.matches::<(NumberType, Shape)>() {
            let (dtype, shape) = schema.opt_cast_into().unwrap();
            self.zeros(txn, dtype, shape).await
        } else if schema.matches::<(NumberType, Shape, Number)>() {
            let (dtype, shape, number): (NumberType, Shape, Number) =
                schema.opt_cast_into().unwrap();
            let number = dtype.try_cast(number).unwrap();
            self.constant(txn, shape, number).await
        } else if schema.matches::<(NumberType, Shape, Vec<(Vec<u64>, Number)>)>() {
            let (dtype, shape, values): (NumberType, Shape, Vec<(Vec<u64>, Number)>) =
                schema.opt_cast_into().unwrap();
            let tensor = self.zeros(txn, dtype, shape).await?;

            let view = tensor.clone().into_view();
            let zero = dtype.zero();
            stream::iter(values.into_iter().filter(|(_, value)| value != &zero))
                .map(|(coord, value)| Ok(view.write_value_at(txn.id().clone(), coord, value)))
                .try_buffer_unordered(2usize)
                .try_fold((), |(), ()| future::ready(Ok(())))
                .await?;

            Ok(tensor)
        } else if schema.matches::<(NumberType, Shape, Value)>() {
            let (dtype, shape, values): (NumberType, Shape, Value) =
                schema.opt_cast_into().unwrap();

            let values = flatten_ndarray(values)?;
            if shape.size() != values.len() as u64 {
                return Err(error::bad_request(
                    format!(
                        "Tensor with shape {} has {} values but found",
                        shape,
                        shape.size()
                    ),
                    values.len(),
                ));
            }

            let tensor = match self {
                Self::Dense => {
                    let file =
                        BlockListFile::from_values(txn, shape, dtype, stream::iter(values)).await?;

                    Tensor::Dense(DenseTensor::from(file).into_dyn())
                }
                Self::Sparse => {
                    let table =
                        SparseTable::from_values(txn, shape, dtype, stream::iter(values)).await?;

                    Tensor::Sparse(SparseTensor::from(table).into_dyn())
                }
            };

            Ok(tensor)
        } else {
            Err(error::bad_request(
                "Tensor schema is (NumberType, Shape), not",
                schema,
            ))
        }
    }
}

impl From<TensorType> for Link {
    fn from(tt: TensorType) -> Link {
        let prefix = TensorType::prefix();

        use TensorType::*;
        match tt {
            Dense => prefix.append(label("dense")).into(),
            Sparse => prefix.append(label("sparse")).into(),
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dense => write!(f, "type: DenseTensor"),
            Self::Sparse => write!(f, "type: SparseTensor"),
        }
    }
}

#[derive(Clone)]
pub enum Tensor {
    Dense(DenseTensor<BlockListDyn>),
    Sparse(SparseTensor<SparseAccessorDyn>),
}

impl Instance for Tensor {
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Dense(_) => Self::Class::Dense,
            Self::Sparse(_) => Self::Class::Sparse,
        }
    }
}

#[async_trait]
impl CollectionInstance for Tensor {
    type Item = Number;

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.any(txn.clone()).map_ok(|any| !any).await
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        match self {
            // TODO: Forward errors, don't panic!
            Self::Dense(dense) => {
                let result_stream = dense.value_stream(txn).await?;
                let values: TCStream<Scalar> = Box::pin(
                    result_stream.map(|r| r.map(Value::Number).map(Scalar::Value).unwrap()),
                );
                Ok(values)
            }
            Self::Sparse(sparse) => {
                let result_stream = sparse.filled(txn).await?;
                let values: TCStream<Scalar> = Box::pin(
                    result_stream
                        .map(|r| r.unwrap())
                        .map(Value::from)
                        .map(Scalar::Value),
                );
                Ok(values)
            }
        }
    }
}

impl TensorAccessor for Tensor {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ndim(),
            Self::Sparse(sparse) => sparse.ndim(),
        }
    }

    fn shape(&'_ self) -> &'_ Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Dense(dense) => dense.size(),
            Self::Sparse(sparse) => sparse.size(),
        }
    }
}

#[async_trait]
impl Transact for Tensor {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.commit(txn_id).await,
            Self::Sparse(sparse) => sparse.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.rollback(txn_id).await,
            Self::Sparse(sparse) => sparse.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.finalize(txn_id).await,
            Self::Sparse(sparse) => sparse.finalize(txn_id).await,
        }
    }
}

impl IntoView for Tensor {
    fn into_view(self) -> Tensor {
        self
    }
}

impl TensorInstance for Tensor {}

#[async_trait]
impl TensorBoolean<Tensor> for Tensor {
    type Combine = Tensor;

    fn and(&self, other: &Self) -> TCResult<Self> {
        use Tensor::*;
        match (self, other) {
            (Dense(left), Dense(right)) => left.and(right).map(Self::from),
            (Sparse(left), Sparse(right)) => left.and(right).map(Self::from),
            (Sparse(left), Dense(right)) => left.and(&from_dense(right.clone())).map(Self::from),

            _ => other.and(self),
        }
    }

    fn or(&self, other: &Self) -> TCResult<Self> {
        use Tensor::*;
        match (self, other) {
            (Dense(left), Dense(right)) => left.or(right).map(Self::from),
            (Sparse(left), Sparse(right)) => left.or(right).map(Self::from),
            (Dense(left), Sparse(right)) => left.or(&from_sparse(right.clone())).map(Self::from),

            _ => other.or(self),
        }
    }

    fn xor(&self, other: &Self) -> TCResult<Self> {
        use Tensor::*;
        match (self, other) {
            (Dense(left), Dense(right)) => left.xor(right).map(Self::from),
            (Sparse(left), _) => from_sparse(left.clone()).into_view().xor(other),
            (left, right) => right.xor(left),
        }
    }
}

#[async_trait]
impl TensorUnary for Tensor {
    type Unary = Tensor;

    fn abs(&self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.abs().map(Self::from),
            Self::Sparse(sparse) => sparse.abs().map(Self::from),
        }
    }

    async fn all(&self, txn: Txn) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.all(txn).await,
            Self::Sparse(sparse) => sparse.all(txn).await,
        }
    }

    async fn any(&self, txn: Txn) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.any(txn).await,
            Self::Sparse(sparse) => sparse.any(txn).await,
        }
    }

    fn not(&self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.not().map(Self::from),
            Self::Sparse(sparse) => sparse.not().map(Self::from),
        }
    }
}

#[async_trait]
impl TensorCompare<Tensor> for Tensor {
    type Compare = Self;
    type Dense = Self;

    async fn eq(&self, other: &Self, txn: Txn) -> TCResult<Self> {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => {
                left.eq(right, txn).map_ok(IntoView::into_view).await
            }
            (Self::Sparse(left), Self::Sparse(right)) => {
                left.eq(right, txn).map_ok(IntoView::into_view).await
            }
            (Self::Sparse(left), right) => {
                from_sparse(left.clone())
                    .into_view()
                    .eq(right, txn)
                    .map_ok(IntoView::into_view)
                    .await
            }
            (left, Self::Sparse(right)) => {
                left.eq(&from_sparse(right.clone()).into_view(), txn)
                    .map_ok(IntoView::into_view)
                    .await
            }
        }
    }

    fn gt(&self, other: &Self) -> TCResult<Self> {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => left.gt(right).map(Self::from),
            (Self::Sparse(left), Self::Sparse(right)) => left.gt(right).map(Self::from),
            (Self::Sparse(left), right) => from_sparse(left.clone()).into_view().gt(right),
            (left, Self::Sparse(right)) => left
                .gt(&from_sparse(right.clone()).into_view())
                .map(Self::from),
        }
    }

    async fn gte(&self, other: &Self, txn: Txn) -> TCResult<Self> {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => {
                left.gte(right, txn).map_ok(IntoView::into_view).await
            }
            (Self::Sparse(left), Self::Sparse(right)) => {
                left.gte(right, txn).map_ok(IntoView::into_view).await
            }
            (Self::Sparse(left), right) => {
                from_sparse(left.clone())
                    .into_view()
                    .gte(right, txn)
                    .map_ok(IntoView::into_view)
                    .await
            }
            (left, Self::Sparse(right)) => {
                left.gte(&from_sparse(right.clone()).into_view(), txn)
                    .map_ok(IntoView::into_view)
                    .await
            }
        }
    }

    fn lt(&self, other: &Self) -> TCResult<Self> {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => left.lt(right).map(Self::from),
            (Self::Sparse(left), Self::Sparse(right)) => left.lt(right).map(Self::from),
            (Self::Sparse(left), right) => from_sparse(left.clone()).into_view().lt(right),
            (left, Self::Sparse(right)) => left.lt(&from_sparse(right.clone()).into_view()),
        }
    }

    async fn lte(&self, other: &Self, txn: Txn) -> TCResult<Self> {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => {
                left.lte(right, txn).map_ok(IntoView::into_view).await
            }
            (Self::Sparse(left), Self::Sparse(right)) => {
                left.lte(right, txn).map_ok(IntoView::into_view).await
            }
            (Self::Sparse(left), right) => {
                from_sparse(left.clone())
                    .into_view()
                    .lte(right, txn)
                    .map_ok(IntoView::into_view)
                    .await
            }
            (left, Self::Sparse(right)) => {
                left.lte(&from_sparse(right.clone()).into_view(), txn)
                    .map_ok(IntoView::into_view)
                    .await
            }
        }
    }

    fn ne(&self, other: &Self) -> TCResult<Self> {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => left.ne(right).map(Self::from),
            (Self::Sparse(left), Self::Sparse(right)) => left.ne(right).map(Self::from),
            (Self::Sparse(left), right) => from_sparse(left.clone())
                .into_view()
                .ne(right)
                .map(Self::from),
            (left, Self::Sparse(right)) => left
                .ne(&from_sparse(right.clone()).into_view())
                .map(Self::from),
        }
    }
}

#[async_trait]
impl TensorIO for Tensor {
    async fn read_value(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        match self {
            Self::Dense(dense) => dense.read_value(txn, coord).await,
            Self::Sparse(sparse) => sparse.read_value(txn, coord).await,
        }
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        match self {
            Self::Dense(dense) => dense.write_value(txn_id, bounds, value).await,
            Self::Sparse(sparse) => sparse.write_value(txn_id, bounds, value).await,
        }
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCResult<()> {
        match self {
            Self::Dense(dense) => dense.write_value_at(txn_id, coord, value).await,
            Self::Sparse(sparse) => sparse.write_value_at(txn_id, coord, value).await,
        }
    }
}

#[async_trait]
impl TensorDualIO<Tensor> for Tensor {
    async fn mask(&self, txn: &Txn, other: Self) -> TCResult<()> {
        match (self, &other) {
            (Self::Dense(l), Self::Dense(r)) => l.mask(txn, r.clone()).await,
            (Self::Sparse(l), Self::Sparse(r)) => l.mask(txn, r.clone()).await,
            (Self::Sparse(l), Self::Dense(r)) => l.mask(txn, from_dense(r.clone())).await,
            (l, Self::Sparse(r)) => l.mask(txn, from_sparse(r.clone()).into_view()).await,
        }
    }

    async fn write(&self, txn: Txn, bounds: Bounds, value: Self) -> TCResult<()> {
        match self {
            Self::Dense(this) => match value {
                Self::Dense(that) => this.write(txn, bounds, that).await,
                Self::Sparse(that) => this.write(txn, bounds, from_sparse(that)).await,
            },
            Self::Sparse(this) => match value {
                Self::Dense(that) => this.write(txn, bounds, from_dense(that)).await,
                Self::Sparse(that) => this.write(txn, bounds, that).await,
            },
        }
    }
}

impl TensorMath<Tensor> for Tensor {
    type Combine = Self;

    fn add(&self, other: &Self) -> TCResult<Self> {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => left.add(right).map(Self::from),
            (Self::Sparse(left), Self::Sparse(right)) => left.add(right).map(Self::from),
            (Self::Dense(left), Self::Sparse(right)) => {
                left.add(&from_sparse(right.clone())).map(Self::from)
            }
            (Self::Sparse(left), Self::Dense(right)) => {
                from_sparse(left.clone()).add(right).map(Self::from)
            }
        }
    }

    fn multiply(&self, other: &Self) -> TCResult<Self> {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => left.multiply(right).map(Self::from),
            (Self::Sparse(left), Self::Sparse(right)) => left.multiply(right).map(Self::from),
            (Self::Dense(left), Self::Sparse(right)) => {
                left.multiply(&from_sparse(right.clone())).map(Self::from)
            }
            (Self::Sparse(left), Self::Dense(right)) => {
                from_sparse(left.clone()).multiply(right).map(Self::from)
            }
        }
    }
}

impl TensorReduce for Tensor {
    type Reduce = Self;

    fn product(&self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.product(axis).map(Self::from),
            Self::Sparse(sparse) => sparse.product(axis).map(Self::from),
        }
    }

    fn product_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        match self {
            Self::Dense(dense) => dense.product_all(txn),
            Self::Sparse(sparse) => sparse.product_all(txn),
        }
    }

    fn sum(&self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.product(axis).map(Self::from),
            Self::Sparse(sparse) => sparse.product(axis).map(Self::from),
        }
    }

    fn sum_all(&self, txn: Txn) -> TCBoxTryFuture<Number> {
        match self {
            Self::Dense(dense) => dense.product_all(txn),
            Self::Sparse(sparse) => sparse.product_all(txn),
        }
    }
}

impl TensorTransform for Tensor {
    type Cast = Self;
    type Broadcast = Self;
    type Expand = Self;
    type Slice = Self;
    type Reshape = Self;
    type Transpose = Self;

    fn as_type(&self, dtype: NumberType) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.as_type(dtype).map(Self::from),
            Self::Sparse(sparse) => sparse.as_type(dtype).map(Self::from),
        }
    }

    fn broadcast(&self, shape: Shape) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.broadcast(shape).map(Self::from),
            Self::Sparse(sparse) => sparse.broadcast(shape).map(Self::from),
        }
    }

    fn expand_dims(&self, axis: usize) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.expand_dims(axis).map(Self::from),
            Self::Sparse(sparse) => sparse.expand_dims(axis).map(Self::from),
        }
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.slice(bounds).map(Self::from),
            Self::Sparse(sparse) => sparse.slice(bounds).map(Self::from),
        }
    }

    fn reshape(&self, shape: Shape) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.reshape(shape).map(Self::from),
            Self::Sparse(sparse) => sparse.reshape(shape).map(Self::from),
        }
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.transpose(permutation).map(Self::from),
            Self::Sparse(sparse) => sparse.transpose(permutation).map(Self::from),
        }
    }
}

#[async_trait]
impl Public for Tensor {
    async fn get(
        &self,
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State> {
        if path.is_empty() {
            let bounds = if selector.is_none() {
                Bounds::all(self.shape())
            } else {
                selector
                    .try_cast_into(|s| error::bad_request("Expected Tensor bounds but found", s))?
            };

            if bounds.is_coord() {
                let coord: Vec<u64> = bounds.try_into()?;
                let value = self.read_value(&txn, &coord).await?;
                Ok(State::Scalar(Scalar::Value(Value::Number(value))))
            } else {
                let slice = self.slice(bounds)?;
                Ok(State::Collection(slice.into()))
            }
        } else if path.len() == 1 {
            match path[0].as_str() {
                "all" => self
                    .all(txn.clone())
                    .await
                    .map(Value::from)
                    .map(State::from),
                "any" => self
                    .any(txn.clone())
                    .await
                    .map(Value::from)
                    .map(State::from),
                "as_type" => {
                    let dtype: NumberType =
                        selector.try_cast_into(|v| error::bad_request("Invalid NumberType", v))?;

                    self.as_type(dtype)
                        .map(Collection::from)
                        .map(State::Collection)
                }
                "broadcast" => {
                    let shape =
                        selector.try_cast_into(|v| error::bad_request("Invalid shape", v))?;

                    self.broadcast(shape)
                        .map(Collection::from)
                        .map(State::Collection)
                }
                "expand_dims" => {
                    let axis = selector.try_cast_into(|v| error::bad_request("Invalid axis", v))?;

                    self.expand_dims(axis)
                        .map(Collection::from)
                        .map(State::Collection)
                }
                "not" => self.not().map(Collection::from).map(State::Collection),
                "reshape" => {
                    let shape =
                        selector.try_cast_into(|v| error::bad_request("Invalid shape", v))?;

                    self.reshape(shape)
                        .map(Collection::from)
                        .map(State::Collection)
                }
                "transpose" => {
                    let permutation = if selector.is_none() {
                        None
                    } else {
                        let permutation = selector.try_cast_into(|v| {
                            error::bad_request("Permutation should be a tuple of axes, not", v)
                        })?;
                        Some(permutation)
                    };

                    self.transpose(permutation)
                        .map(Collection::from)
                        .map(State::Collection)
                }
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn put(
        &self,
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let bounds = if selector.is_none() {
            Bounds::all(self.shape())
        } else {
            selector.try_cast_into(|s| error::bad_request("Expected Tensor bounds but found", s))?
        };

        match value {
            State::Scalar(Scalar::Value(Value::Number(value))) => {
                self.write_value(txn.id().clone(), bounds, value).await
            }
            State::Collection(Collection::Tensor(tensor)) => {
                self.write(txn.clone(), bounds, tensor).await
            }
            other => Err(error::bad_request(
                "Not a valid Tensor value or slice",
                other,
            )),
        }
    }

    async fn post(
        &self,
        _request: &Request,
        _txn: &Txn,
        path: &[PathSegment],
        mut params: Object,
    ) -> TCResult<State> {
        if path.is_empty() {
            Err(error::method_not_allowed("Tensor::POST /"))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "slice" => {
                    let bounds = params
                        .remove(&label("bounds").into())
                        .ok_or(error::bad_request("Missing parameter", "bounds"))?;
                    let bounds = Bounds::from_scalar(self.shape(), bounds)?;

                    if params.is_empty() {
                        self.slice(bounds)
                            .map(Collection::from)
                            .map(State::Collection)
                    } else {
                        Err(error::bad_request(
                            "Unrecognized parameters",
                            Scalar::from_iter(params),
                        ))
                    }
                }
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn delete(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _selector: Value,
    ) -> TCResult<()> {
        Err(error::not_implemented("Tensor::delete"))
    }
}

impl<T: Clone + BlockList> From<DenseTensor<T>> for Tensor {
    fn from(dense: DenseTensor<T>) -> Tensor {
        let blocks = Arc::new(BlockListDyn::new(dense.clone_into()));
        Self::Dense(DenseTensor::from(blocks))
    }
}

impl<T: Clone + SparseAccess> From<SparseTensor<T>> for Tensor {
    fn from(sparse: SparseTensor<T>) -> Tensor {
        let accessor = Arc::new(SparseAccessorDyn::new(sparse.clone_into()));
        Self::Sparse(SparseTensor::from(accessor))
    }
}

impl TryFrom<Collection> for Tensor {
    type Error = error::TCError;

    fn try_from(collection: Collection) -> TCResult<Tensor> {
        match collection {
            Collection::Tensor(tensor) => Ok(tensor),
            other => Err(error::bad_request("Expected Tensor but found", other)),
        }
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dense(_) => write!(f, "(DenseTensor)"),
            Self::Sparse(_) => write!(f, "(SparseTensor)"),
        }
    }
}

fn flatten_ndarray(values: Value) -> TCResult<Vec<Number>> {
    if values.matches::<Vec<Number>>() {
        values.try_cast_into(|v| error::bad_request("Invalid ndarray", v))
    } else if values.matches::<Vec<Value>>() {
        let values: Vec<Value> = values.opt_cast_into().unwrap();
        let mut ndarray = vec![];
        for value in values.into_iter() {
            ndarray.extend(flatten_ndarray(value)?);
        }
        Ok(ndarray)
    } else {
        Err(error::bad_request(
            "Expected an n-dimensional array of numbers but found",
            values,
        ))
    }
}
