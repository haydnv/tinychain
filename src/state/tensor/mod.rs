use std::sync::Arc;

use crate::transaction::{Txn, TxnId};
use crate::value::class::NumberType;
use crate::value::{Number, TCResult};

mod array;
mod bounds;
mod dense;
mod sparse;

use dense::DenseTensor;
use sparse::SparseTensor;

trait TensorView {
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
    fn into_type(self, dtype: NumberType) -> TCResult<Self>;

    fn broadcast(self, shape: bounds::Shape) -> TCResult<Self>;

    fn expand_dims(self, axis: usize) -> TCResult<Self>;

    fn slice(self, bounds: bounds::Bounds) -> TCResult<Self>;

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self>;
}

trait TensorUnary: Sized + TensorView {
    fn product(&self, txn: Arc<Txn>, axis: usize) -> TCResult<Self>;

    fn product_all(&self, txn_id: TxnId) -> TCResult<Number>;

    fn sum(&self, txn: Arc<Txn>, axis: usize) -> TCResult<Self>;

    fn sum_all(&self, txn_id: TxnId) -> TCResult<Number>;
}

enum Tensor {
    Sparse(SparseTensor),
    Dense(DenseTensor),
}

impl TensorView for Tensor {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Sparse(sparse) => sparse.dtype(),
            Self::Dense(dense) => dense.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Sparse(sparse) => sparse.ndim(),
            Self::Dense(dense) => dense.ndim(),
        }
    }

    fn shape(&'_ self) -> &'_ bounds::Shape {
        match self {
            Self::Sparse(sparse) => sparse.shape(),
            Self::Dense(dense) => dense.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Sparse(sparse) => sparse.size(),
            Self::Dense(dense) => dense.size(),
        }
    }
}
