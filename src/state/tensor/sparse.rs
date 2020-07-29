use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use futures::stream::StreamExt;

use crate::error;
use crate::state::table::schema::*;
use crate::state::table::{Selection, TableBase};
use crate::transaction::{Txn, TxnId};
use crate::value::class::{NumberClass, NumberType, UIntType, ValueType};
use crate::value::{Number, TCResult, TCStream, UInt, Value, ValueId};

use super::bounds::{Bounds, Shape};
use super::TensorView;

const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

#[async_trait]
trait SparseAccessor: TensorView {
    async fn filled(&self, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>>;

    async fn filled_at(&self, txn: Arc<Txn>, axes: &[usize]) -> TCResult<TCStream<Vec<u64>>>;

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64>;

    fn read_value<'a>(&'a self, txn_id: TxnId, coord: Vec<u64>) -> BoxFuture<'a, TCResult<Number>>;

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        coord: Vec<u64>,
        value: Number,
    ) -> BoxFuture<'a, TCResult<()>>;
}

struct SparseTable {
    table: TableBase,
    shape: Shape,
    dtype: NumberType,
}

impl SparseTable {
    pub async fn create(txn: Arc<Txn>, shape: Shape, dtype: NumberType) -> TCResult<SparseTable> {
        let u64_type = ValueType::Number(NumberType::UInt(UIntType::U64));
        let key: Vec<Column> = (0..shape.len())
            .map(|axis| Column {
                name: axis.into(),
                dtype: u64_type,
                max_len: None,
            })
            .collect();

        let value = vec![Column {
            name: "value".parse()?,
            dtype: dtype.into(),
            max_len: None,
        }];

        let schema = Schema::new(key, value);
        let table = TableBase::create(txn, schema).await?;

        Ok(SparseTable {
            table,
            dtype,
            shape,
        })
    }
}

impl TensorView for SparseTable {
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
impl SparseAccessor for SparseTable {
    async fn filled(&self, txn_id: TxnId) -> TCResult<TCStream<(Vec<u64>, Number)>> {
        let filled = self.table.stream(txn_id).await?.map(unwrap_row);

        Ok(Box::pin(filled))
    }

    async fn filled_at(&self, txn: Arc<Txn>, axes: &[usize]) -> TCResult<TCStream<Vec<u64>>> {
        let txn_id = txn.id().clone();
        let columns: Vec<ValueId> = axes.iter().map(|x| (*x).into()).collect();
        Ok(Box::pin(
            self.table
                .group_by(txn, columns)
                .await?
                .stream(txn_id)
                .await?
                .map(|coord| unwrap_coord(&coord)),
        ))
    }

    async fn filled_count(&self, txn_id: TxnId) -> TCResult<u64> {
        self.table.count(txn_id).await
    }

    fn read_value<'a>(&'a self, txn_id: TxnId, coord: Vec<u64>) -> BoxFuture<'a, TCResult<Number>> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(error::bad_request(
                    "Coordinate out of bounds",
                    Bounds::from(coord),
                ));
            }

            let selector: HashMap<ValueId, Value> = coord
                .iter()
                .enumerate()
                .map(|(axis, at)| (axis.into(), u64_to_value(*at)))
                .collect();

            let mut slice = self
                .table
                .slice(&txn_id, selector.into())
                .await?
                .select(vec!["value".parse()?])?
                .stream(txn_id)
                .await?;

            match slice.next().await {
                Some(mut number) if number.len() == 1 => number.pop().unwrap().try_into(),
                None => Ok(self.dtype().zero()),
                _ => Err(error::internal(ERR_CORRUPT)),
            }
        })
    }

    fn write_value<'a>(
        &'a self,
        txn_id: TxnId,
        mut coord: Vec<u64>,
        value: Number,
    ) -> BoxFuture<'a, TCResult<()>> {
        Box::pin(async move {
            let mut row: HashMap<ValueId, Value> = coord
                .drain(..)
                .enumerate()
                .map(|(x, v)| (x.into(), Value::Number(Number::UInt(UInt::U64(v)))))
                .collect();

            row.insert("value".parse()?, value.into());
            self.table.upsert(&txn_id, row).await
        })
    }
}

pub struct SparseTensor {
    accessor: Box<dyn SparseAccessor>,
}

impl TensorView for SparseTensor {
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn ndim(&self) -> usize {
        self.accessor.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.accessor.shape()
    }

    fn size(&self) -> u64 {
        self.accessor.size()
    }
}

fn u64_to_value(u: u64) -> Value {
    Value::Number(Number::UInt(UInt::U64(u)))
}

fn unwrap_coord(coord: &[Value]) -> Vec<u64> {
    coord.iter().map(|val| unwrap_u64(val)).collect()
}

fn unwrap_row(mut row: Vec<Value>) -> (Vec<u64>, Number) {
    let coord = unwrap_coord(&row[0..row.len() - 1]);
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
