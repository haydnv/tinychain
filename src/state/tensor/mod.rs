use std::sync::Arc;

use arrayfire::{Array, HasAfEnum};
use async_trait::async_trait;

use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, TCType};

#[async_trait]
pub trait TensorView {
    type DType: HasAfEnum;

    fn shape(&'_ self) -> &'_ [u64];

    fn size(&self) -> u64;

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Self::DType>;
}

#[async_trait]
pub trait BlockTensorView: TensorView {
    async fn sum(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<BlockTensor>;

    async fn product(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<BlockTensor>;

    async fn add<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: T) -> TCResult<BlockTensor>;

    async fn multiply<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: T)
        -> TCResult<BlockTensor>;

    async fn subtract<T: BlockTensorView>(
        &self,
        txn: &Arc<Txn>,
        other: &T,
    ) -> TCResult<BlockTensor>;

    async fn equals<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn and<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn or<T: TensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn xor<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn not(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

    async fn ones(&self, txn: &Arc<Txn>, shape: Vec<u64>) -> BlockTensor;

    async fn blocks(&self, txn_id: &Arc<TxnId>) -> TCStream<Array<<Self as TensorView>::DType>>;
}

#[async_trait]
pub trait SparseTensorView: TensorView {
    async fn sum(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<SparseTensor>;

    async fn product(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<SparseTensor>;

    async fn add<T: TensorView>(&self, txn: &Arc<Txn>, other: T) -> TCResult<SparseTensor>;

    async fn multiply<T: TensorView>(&self, txn: &Arc<Txn>, other: T) -> TCResult<SparseTensor>;

    async fn subtract<T: TensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<SparseTensor>;

    async fn equals<T: TensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn and<T: TensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<SparseTensor>;

    async fn or<T: SparseTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<SparseTensor>;

    async fn xor<T: TensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn not(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

    async fn filled(&self, txn_id: &TxnId) -> TCStream<(Vec<u64>, <Self as TensorView>::DType)>;

    async fn filled_at(&self, txn_id: &TxnId, axes: &[usize]) -> TCStream<Vec<u64>>;

    async fn filled_count(&self, txn_id: &TxnId) -> TCResult<u64>;

    async fn to_dense(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;
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
