use std::sync::Arc;

use async_trait::async_trait;

use crate::state::table::{Column, Schema, TableBase};
use crate::state::Dir;
use crate::transaction::{Txn, TxnId};
use crate::value::class::NumberType;
use crate::value::{TCResult, TCStream, Value};

use super::base::*;
use super::bounds::Shape;
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
    shape: Shape,
    table: TableBase,
}

impl SparseTensor {
    pub async fn create(
        txn_id: TxnId,
        dir: Arc<Dir>,
        dtype: NumberType,
        shape: Shape,
    ) -> TCResult<SparseTensor> {
        let key: Vec<Column> = (0..shape.len())
            .map(|axis| Column {
                name: axis.into(),
                dtype: dtype.into(),
                max_len: None,
            })
            .collect();

        let schema = Schema::new(key, vec![]);
        let table = TableBase::create(txn_id, dir, schema).await?;

        Ok(SparseTensor {
            dtype,
            shape,
            table,
        })
    }
}

impl TensorView for SparseTensor {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.shape.size()
    }
}
