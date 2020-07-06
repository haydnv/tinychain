use arrayfire::HasAfEnum;
use async_trait::async_trait;

use crate::transaction::TxnId;
use crate::value::{TCResult, TCType};

#[async_trait]
pub trait TensorView {
    type DType: HasAfEnum;

    fn shape(&'_ self) -> &'_ [u64];

    fn size(&self) -> u64;

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Self::DType>;
}

#[async_trait]
pub trait BlockTensorView: TensorView {
    async fn sum(&self, txn_id: &TxnId, axis: Option<usize>) -> TCResult<BlockTensor>;

    async fn product(&self, txn_id: &TxnId, axis: Option<usize>) -> TCResult<BlockTensor>;

    async fn add<T: TensorView>(&self, txn_id: &TxnId, other: T) -> TCResult<BlockTensor>;

    async fn multiply<T: BlockTensorView>(&self, txn_id: &TxnId, other: T)
        -> TCResult<BlockTensor>;

    async fn subtract<T: TensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<BlockTensor>;

    async fn equals<T: BlockTensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<BlockTensor>;

    async fn and<T: BlockTensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<BlockTensor>;

    async fn or<T: TensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<BlockTensor>;

    async fn xor<T: TensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<BlockTensor>;

    async fn not(&self, txn_id: &TxnId) -> TCResult<BlockTensor>;
}

#[async_trait]
pub trait SparseTensorView: TensorView {
    async fn sum(&self, txn_id: &TxnId, axis: Option<usize>) -> TCResult<SparseTensor>;

    async fn product(&self, txn_id: &TxnId, axis: Option<usize>) -> TCResult<SparseTensor>;

    async fn add<T: SparseTensorView>(&self, txn_id: &TxnId, other: T) -> TCResult<SparseTensor>;

    async fn multiply<T: TensorView>(&self, txn_id: &TxnId, other: T) -> TCResult<SparseTensor>;

    async fn subtract<T: SparseTensorView>(
        &self,
        txn_id: &TxnId,
        other: &T,
    ) -> TCResult<SparseTensor>;

    async fn equals<T: TensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<BlockTensor>;

    async fn and<T: TensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<SparseTensor>;

    async fn or<T: SparseTensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<SparseTensor>;

    async fn xor<T: TensorView>(&self, txn_id: &TxnId, other: &T) -> TCResult<BlockTensor>;

    async fn not(&self, txn_id: &TxnId) -> TCResult<BlockTensor>;
}

pub struct BlockTensor {
    dtype: TCType,
    shape: Vec<u64>,
    size: u64,
}

pub struct SparseTensor {
    dtype: TCType,
    shape: Vec<u64>,
    size: u64,
}

pub enum Tensor {
    Dense(BlockTensor),
    Sparse(SparseTensor),
}
