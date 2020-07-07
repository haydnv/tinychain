use std::sync::Arc;

use arrayfire::Array;
use async_trait::async_trait;

use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, TCType};

use super::base::TensorView;

#[async_trait]
pub trait BlockTensorView: TensorView {
    async fn as_dtype(&self, txn: &Arc<Txn>, dtype: TCType) -> TCResult<BlockTensor>;

    async fn copy(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

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

    async fn or<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn xor<T: BlockTensorView>(&self, txn: &Arc<Txn>, other: &T) -> TCResult<BlockTensor>;

    async fn not(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;

    async fn blocks(&self, txn_id: &Arc<TxnId>) -> TCStream<Array<<Self as TensorView>::DType>>;
}

pub struct BlockTensor {
    dtype: TCType,
    shape: Vec<u64>,
    size: u64,
}
