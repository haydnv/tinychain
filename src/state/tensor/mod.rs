use std::sync::Arc;

use crate::transaction::{Txn, TxnId};
use crate::value::class::NumberType;
use crate::value::{Number, TCBoxTryFuture, TCResult};

mod bounds;
mod dense;
mod sparse;
mod transform;

pub type Array = dense::array::Array;

use dense::DenseTensor;
use sparse::SparseTensor;

trait TensorView: Send + Sync {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ bounds::Shape;

    fn size(&self) -> u64;
}

trait TensorBoolean: Sized + TensorView {
    fn all(&self, txn_id: TxnId) -> TCResult<bool>;

    fn any(&self, txn_id: TxnId) -> TCResult<bool>;

    fn and(&self, other: &Self) -> TCResult<Self>;

    fn not(&self) -> TCResult<Self>;

    fn or(&self, other: &Self) -> TCResult<Self>;

    fn xor(&self, other: &Self) -> TCResult<Self>;
}

trait TensorCompare: Sized + TensorView {
    fn eq(&self, other: &Self) -> TCResult<DenseTensor>;

    fn gt(&self, other: &Self) -> TCResult<Self>;

    fn gte(&self, other: &Self) -> TCResult<DenseTensor>;

    fn lt(&self, other: &Self) -> TCResult<Self>;

    fn lte(&self, other: &Self) -> TCResult<DenseTensor>;

    fn ne(&self, other: &Self) -> TCResult<Self>;
}

trait TensorIO: Sized + TensorView {
    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number>;

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        bounds: bounds::Bounds,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()>;

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()>;
}

trait TensorMath: Sized + TensorView {
    fn abs(&self) -> TCResult<Self>;

    fn add(&self, other: &Self) -> TCResult<Self>;

    fn multiply(&self, other: &Self) -> TCResult<Self>;
}

trait TensorTransform: Sized + TensorView {
    fn as_type(&self, dtype: NumberType) -> TCResult<Self>;

    fn broadcast(&self, shape: bounds::Shape) -> TCResult<Self>;

    fn expand_dims(&self, axis: usize) -> TCResult<Self>;

    fn slice(&self, bounds: bounds::Bounds) -> TCResult<Self>;

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self>;
}

trait TensorUnary: Sized + TensorView {
    fn product(&self, txn: Arc<Txn>, axis: usize) -> TCResult<Self>;

    fn product_all(&self, txn_id: TxnId) -> TCResult<Number>;

    fn sum(&self, txn: Arc<Txn>, axis: usize) -> TCResult<Self>;

    fn sum_all(&self, txn_id: TxnId) -> TCResult<Number>;
}

enum Tensor {
    Dense(DenseTensor),
    Sparse(SparseTensor),
}

impl TensorView for Tensor {
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

    fn shape(&'_ self) -> &'_ bounds::Shape {
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

impl TensorBoolean for Tensor {
    fn all(&self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.all(txn_id),
            Self::Sparse(sparse) => sparse.all(txn_id),
        }
    }

    fn any(&self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.any(txn_id),
            Self::Sparse(sparse) => sparse.any(txn_id),
        }
    }

    fn and(&self, other: &Self) -> TCResult<Self> {
        use Tensor::*;
        match (self, other) {
            (Dense(left), Dense(right)) => left.and(right).map(Self::from),
            (Sparse(left), Sparse(right)) => left.and(right).map(Self::from),
            (Dense(left), Sparse(right)) => left
                .and(&DenseTensor::from_sparse(right.clone()))
                .map(Self::from),
            _ => other.and(self),
        }
    }

    fn not(&self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.not().map(Self::from),
            Self::Sparse(sparse) => sparse.not().map(Self::from),
        }
    }

    fn or(&self, other: &Self) -> TCResult<Self> {
        use Tensor::*;
        match (self, other) {
            (Dense(left), Dense(right)) => left.or(right).map(Self::from),
            (Sparse(left), Sparse(right)) => left.or(right).map(Self::from),
            (Dense(left), Sparse(right)) => left
                .or(&DenseTensor::from_sparse(right.clone()))
                .map(Self::from),
            _ => other.and(self),
        }
    }

    fn xor(&self, other: &Self) -> TCResult<Self> {
        use Tensor::*;
        match (self, other) {
            (Dense(left), Dense(right)) => left.xor(right).map(Self::from),
            (Sparse(left), _) => Dense(DenseTensor::from_sparse(left.clone())).xor(other),
            (left, right) => right.xor(left),
        }
    }
}

impl TensorIO for Tensor {
    fn read_value<'a>(&'a self, txn_id: &'a TxnId, coord: &'a [u64]) -> TCBoxTryFuture<'a, Number> {
        match self {
            Self::Dense(dense) => dense.read_value(txn_id, coord),
            Self::Sparse(sparse) => sparse.read_value(txn_id, coord),
        }
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        bounds: bounds::Bounds,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Dense(dense) => dense.write_value(txn_id, bounds, value),
            Self::Sparse(sparse) => sparse.write_value(txn_id, bounds, value),
        }
    }

    fn write_value_at<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> TCBoxTryFuture<'a, ()> {
        match self {
            Self::Dense(dense) => dense.write_value_at(txn_id, coord, value),
            Self::Sparse(sparse) => sparse.write_value_at(txn_id, coord, value),
        }
    }
}

impl TensorTransform for Tensor {
    fn as_type(&self, dtype: NumberType) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.as_type(dtype).map(Self::from),
            Self::Sparse(sparse) => sparse.as_type(dtype).map(Self::from),
        }
    }

    fn broadcast(&self, shape: bounds::Shape) -> TCResult<Self> {
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

    fn slice(&self, bounds: bounds::Bounds) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.slice(bounds).map(Self::from),
            Self::Sparse(sparse) => sparse.slice(bounds).map(Self::from),
        }
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.transpose(permutation).map(Self::from),
            Self::Sparse(sparse) => sparse.transpose(permutation).map(Self::from),
        }
    }
}

impl From<DenseTensor> for Tensor {
    fn from(dense: DenseTensor) -> Tensor {
        Self::Dense(dense)
    }
}

impl From<SparseTensor> for Tensor {
    fn from(sparse: SparseTensor) -> Tensor {
        Self::Sparse(sparse)
    }
}
