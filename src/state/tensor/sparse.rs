use std::sync::Arc;

use async_trait::async_trait;

use crate::transaction::{Txn, TxnId};
use crate::value::class::NumberType;
use crate::value::{TCResult, TCStream, Value};

use super::base::*;
use super::dense::BlockTensor;

#[async_trait]
pub trait SparseTensorView: TensorView {
    async fn filled(&self, txn_id: &TxnId) -> TCStream<(Vec<u64>, Value)>;

    async fn filled_at(&self, txn_id: &TxnId, axes: &[usize]) -> TCStream<Vec<u64>>;

    async fn filled_count(&self, txn_id: &TxnId) -> TCResult<u64>;

    async fn to_dense(&self, txn: &Arc<Txn>) -> TCResult<BlockTensor>;
}

pub struct SparseTensor {
    dtype: NumberType,
    shape: Vec<u64>,
    size: u64,
}
