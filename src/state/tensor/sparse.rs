use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::StreamExt;

use crate::error;
use crate::state::table::{Column, Schema, Selection, TableBase};
use crate::state::Dir;
use crate::transaction::{Txn, TxnId};
use crate::value::class::NumberType;
use crate::value::{Number, TCResult, TCStream, UInt, Value};

use super::base::*;
use super::bounds::Shape;
use super::dense::BlockTensor;

#[async_trait]
pub trait SparseTensorView: TensorView {
    async fn filled(&self, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>>;

    async fn filled_at(&self, txn_id: &TxnId, axes: &[usize]) -> TCResult<TCStream<Vec<u64>>>;

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64>;

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

#[async_trait]
impl SparseTensorView for SparseTensor {
    async fn filled(&self, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let filled = self.table.stream(txn_id).await?.map(unwrap_row);

        Ok(Box::pin(filled))
    }

    async fn filled_at(&self, _txn_id: &TxnId, _axes: &[usize]) -> TCResult<TCStream<Vec<u64>>> {
        Err(error::not_implemented())
    }

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.table.count(txn_id).await
    }

    async fn to_dense(&self, _txn: &Arc<Txn>) -> TCResult<BlockTensor> {
        Err(error::not_implemented())
    }
}

fn unwrap_row(mut row: Vec<Value>) -> (Vec<u64>, Number) {
    let coord = row[0..row.len() - 1]
        .iter()
        .map(|val| unwrap_u64(val))
        .collect();
    let value = row.pop().unwrap().try_into().unwrap();
    (coord, value)
}

fn unwrap_u64(value: &Value) -> u64 {
    if let Value::Number(Number::UInt(UInt::U64(unwrapped))) = value {
        *unwrapped
    } else {
        panic!("Expected u64 but found {}", value)
    }
}
