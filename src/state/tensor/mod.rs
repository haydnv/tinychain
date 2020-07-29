use std::sync::Arc;

use crate::transaction::{Txn, TxnId};
use crate::value::class::NumberType;
use crate::value::{Number, TCResult};

mod array;
mod bounds;
mod dense;
mod sparse;
mod stream;

pub type Array = array::Array;

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

impl TensorTransform for Tensor {
    fn as_type(&self, dtype: NumberType) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.as_type(dtype).map(Tensor::from),
            Self::Sparse(sparse) => sparse.as_type(dtype).map(Tensor::from),
        }
    }

    fn broadcast(&self, shape: bounds::Shape) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.broadcast(shape).map(Tensor::from),
            Self::Sparse(sparse) => sparse.broadcast(shape).map(Tensor::from),
        }
    }

    fn expand_dims(&self, axis: usize) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.expand_dims(axis).map(Tensor::from),
            Self::Sparse(sparse) => sparse.expand_dims(axis).map(Tensor::from),
        }
    }

    fn slice(&self, bounds: bounds::Bounds) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.slice(bounds).map(Tensor::from),
            Self::Sparse(sparse) => sparse.slice(bounds).map(Tensor::from),
        }
    }

    fn transpose(&self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.transpose(permutation).map(Tensor::from),
            Self::Sparse(sparse) => sparse.transpose(permutation).map(Tensor::from),
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
