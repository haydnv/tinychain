use async_trait::async_trait;

use crate::transaction::{Transact, TxnId};
use crate::value::class::NumberType;

use super::bounds::Shape;
use super::dense::BlockListFile;
use super::sparse::SparseTable;
use super::{DenseTensor, SparseTensor};

pub trait TensorInstance: Send + Sync {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;
}

#[derive(Clone)]
pub enum TensorBase {
    Dense(BlockListFile),
    Sparse(SparseTable),
}

impl TensorInstance for TensorBase {
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
impl Transact for TensorBase {
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
}

impl From<TensorBase> for TensorView {
    fn from(base: TensorBase) -> TensorView {
        match base {
            TensorBase::Dense(blocks) => Self::Dense(blocks.into()),
            TensorBase::Sparse(table) => Self::Sparse(table.into()),
        }
    }
}

#[derive(Clone)]
pub enum TensorView {
    Dense(DenseTensor),
    Sparse(SparseTensor),
}

impl TensorInstance for TensorView {
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
impl Transact for TensorView {
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
}

impl From<DenseTensor> for TensorView {
    fn from(dense: DenseTensor) -> TensorView {
        Self::Dense(dense)
    }
}

impl From<SparseTensor> for TensorView {
    fn from(sparse: SparseTensor) -> TensorView {
        Self::Sparse(sparse)
    }
}

#[derive(Clone)]
pub enum Tensor {
    Base(TensorBase),
    View(TensorView),
}

impl TensorInstance for Tensor {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Base(base) => base.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Base(base) => base.ndim(),
            Self::View(view) => view.ndim(),
        }
    }

    fn shape(&'_ self) -> &'_ Shape {
        match self {
            Self::Base(base) => base.shape(),
            Self::View(view) => view.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Base(base) => base.size(),
            Self::View(view) => view.size(),
        }
    }
}

#[async_trait]
impl Transact for Tensor {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.commit(txn_id).await,
            Self::View(view) => view.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.rollback(txn_id).await,
            Self::View(view) => view.rollback(txn_id).await,
        }
    }
}
