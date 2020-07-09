use std::sync::Arc;

use async_trait::async_trait;

use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, TCType, Value};

use super::base::TensorView;
use super::dense::BlockTensor;

#[async_trait]
pub trait SparseTensorView: TensorView {
    async fn as_dtype(&self, txn: &Arc<Txn>, dtype: TCType) -> TCResult<SparseTensor>;

    async fn copy(&self, txn: &Arc<Txn>) -> TCResult<SparseTensor>;

    async fn abs(&self, txn: &Arc<Txn>) -> TCResult<SparseTensor>;

    async fn sum(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<SparseTensor>;

    async fn product(&self, txn: &Arc<Txn>, axis: Option<usize>) -> TCResult<SparseTensor>;

    async fn multiply<T: SparseTensorView>(
        &self,
        txn: &Arc<Txn>,
        other: T,
    ) -> TCResult<SparseTensor>;

    async fn subtract<T: SparseTensorView>(
        &self,
        txn: &Arc<Txn>,
        other: &T,
    ) -> TCResult<SparseTensor>;

    async fn equals<T: SparseTensorView>(&self, txn: &Arc<Txn>, other: &T)
        -> TCResult<BlockTensor>;

    async fn and<T: SparseTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<SparseTensor>;

    async fn or<T: SparseTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<SparseTensor>;

    async fn xor<T: SparseTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn not(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

    async fn filled(&self, txn_id: &TxnId) -> TCStream<(Vec<u64>, Value)>;

    async fn filled_at(&self, txn_id: &TxnId, axes: &[usize]) -> TCStream<Vec<u64>>;

    async fn filled_count(&self, txn_id: &TxnId) -> TCResult<u64>;

    async fn to_dense(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;
}

pub struct SparseTensor {
    dtype: TCType,
    shape: Vec<u64>,
    size: u64,
}
