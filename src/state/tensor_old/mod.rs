use crate::value::class::NumberType;
use crate::value::TCResult;

mod array;
mod base;
mod bounds;
mod dense;
mod einsum;
mod sparse;
mod stream;

use base::*;
use dense::BlockTensor;
use sparse::TableTensor;

pub trait TensorView: Clone + Sized + Send + Sync {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ bounds::Shape;

    fn size(&self) -> u64;
}

pub type Array = array::Array;

#[derive(Clone)]
pub enum Tensor {
    Dense(BlockTensor),
    DenseTranspose(Permutation<BlockTensor>),
    Sparse(TableTensor),
    SparseTranspose(Permutation<TableTensor>),
}

impl Transpose for Tensor {
    type Permutation = Self;

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Tensor> {
        if permutation == Some((0..self.ndim()).collect::<Vec<usize>>()) {
            return Ok(self);
        }

        match self {
            Self::Dense(dense) => dense.transpose(permutation).map(|t| t.into()),
            Self::DenseTranspose(dt) => dt.transpose(permutation).map(|t| t.into()),
            Self::Sparse(sparse) => sparse.transpose(permutation).map(|t| t.into()),
            Self::SparseTranspose(st) => st.transpose(permutation).map(|t| t.into()),
        }
    }
}

impl TensorView for Tensor {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
            Self::DenseTranspose(dt) => dt.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
            Self::SparseTranspose(st) => st.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ndim(),
            Self::DenseTranspose(dt) => dt.ndim(),
            Self::Sparse(sparse) => sparse.ndim(),
            Self::SparseTranspose(st) => st.ndim(),
        }
    }

    fn shape(&'_ self) -> &'_ bounds::Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::DenseTranspose(dt) => dt.shape(),
            Self::Sparse(sparse) => sparse.shape(),
            Self::SparseTranspose(st) => st.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Dense(dense) => dense.size(),
            Self::DenseTranspose(dt) => dt.size(),
            Self::Sparse(sparse) => sparse.size(),
            Self::SparseTranspose(st) => st.size(),
        }
    }
}

impl From<BlockTensor> for Tensor {
    fn from(dense: BlockTensor) -> Self {
        Self::Dense(dense)
    }
}

impl From<Permutation<BlockTensor>> for Tensor {
    fn from(transpose: Permutation<BlockTensor>) -> Self {
        Self::DenseTranspose(transpose)
    }
}

impl From<TableTensor> for Tensor {
    fn from(sparse: TableTensor) -> Self {
        Self::Sparse(sparse)
    }
}

impl From<Permutation<TableTensor>> for Tensor {
    fn from(sparse: Permutation<TableTensor>) -> Self {
        Self::SparseTranspose(sparse)
    }
}
