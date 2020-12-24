use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;

use async_trait::async_trait;
use futures::future;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};

use crate::class::Instance;
use crate::collection::schema::{Column, IndexSchema};
use crate::collection::table::{self, ColumnBound, Table, TableIndex, TableInstance};
use crate::error;
use crate::general::TCResult;
use crate::scalar::value::number::*;
use crate::scalar::{label, Bound, Id, Label, Value, ValueType};
use crate::transaction::{Transact, Txn, TxnId};

use super::super::bounds::{AxisBounds, Bounds, Shape};
use super::super::stream::*;
use super::super::{Coord, TensorAccess};

use super::access::{SparseAccess, SparseAccessor};
use super::{CoordStream, SparseStream};

const VALUE: Label = label("value");

const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

#[derive(Clone)]
pub struct SparseTable {
    table: TableIndex,
    shape: Shape,
    dtype: NumberType,
}

impl SparseTable {
    pub async fn constant(txn: &Txn, shape: Shape, value: Number) -> TCResult<SparseTable> {
        let bounds = Bounds::all(&shape);
        let table = Self::create(txn, shape, value.class()).await?;

        if value != value.class().zero() {
            stream::iter(bounds.affected())
                .map(|coord| Ok(table.write_value(txn.id().clone(), coord, value.clone())))
                .try_buffer_unordered(2usize)
                .try_fold((), |(), ()| future::ready(Ok(())))
                .await?;
        }

        Ok(table)
    }

    pub async fn create(txn: &Txn, shape: Shape, dtype: NumberType) -> TCResult<Self> {
        let table = Self::create_table(txn, shape.len(), dtype).await?;
        Ok(Self {
            table,
            dtype,
            shape,
        })
    }

    pub async fn create_table(txn: &Txn, ndim: usize, dtype: NumberType) -> TCResult<TableIndex> {
        let key: Vec<Column> = Self::key(ndim);
        let value: Vec<Column> = vec![(VALUE.into(), ValueType::Number(dtype)).into()];
        let indices = (0..ndim).map(|axis| (axis.into(), vec![axis.into()]));
        Table::create(txn, (IndexSchema::from((key, value)), indices).into()).await
    }

    pub async fn from_values<S: Stream<Item = Number>>(
        txn: &Txn,
        shape: Shape,
        dtype: NumberType,
        values: S,
    ) -> TCResult<Self> {
        let zero = dtype.zero();
        let bounds = Bounds::all(&shape);

        let table = Self::create(txn, shape, dtype).await?;

        let txn_id = *txn.id();
        stream::iter(bounds.affected())
            .zip(values)
            .filter(|(_, value)| future::ready(value != &zero))
            .map(|(coord, value)| Ok(table.write_value(txn_id, coord, value)))
            .try_buffer_unordered(2usize)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(table)
    }

    pub fn try_from_table(table: TableIndex, shape: Shape) -> TCResult<SparseTable> {
        let expected_key = Self::key(shape.len());
        let actual_key = table.key();

        for (expected, actual) in actual_key.iter().zip(expected_key.iter()) {
            if expected != actual {
                let key: Vec<String> = table.key().iter().map(|c| c.to_string()).collect();
                return Err(error::bad_request(
                    "Table has invalid key for SparseTable",
                    format!("[{}]", key.join(", ")),
                ));
            }
        }

        let actual_value = table.values();
        if actual_value.len() != 1 {
            let actual_value: Vec<String> = actual_value.iter().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Table has invalid value for SparseTable",
                format!("[{}]", actual_value.join(", ")),
            ));
        }

        let dtype = actual_value[0].dtype();
        if let ValueType::Number(dtype) = dtype {
            Ok(SparseTable {
                table,
                shape,
                dtype,
            })
        } else {
            Err(error::bad_request(
                "Table has invalid data type for SparseTable",
                dtype,
            ))
        }
    }

    fn key(ndim: usize) -> Vec<Column> {
        let u64_type = NumberType::uint64();
        (0..ndim).map(|axis| (axis, u64_type).into()).collect()
    }
}

impl TensorAccess for SparseTable {
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
impl SparseAccess for SparseTable {
    fn accessor(self) -> SparseAccessor {
        SparseAccessor::Table(self)
    }

    async fn filled<'a>(&'a self, txn: &'a Txn) -> TCResult<SparseStream<'a>> {
        let rows = self.table.stream(txn.id()).await?;
        let filled = rows.and_then(|row| future::ready(unwrap_row(row)));
        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at<'a>(
        &'a self,
        _txn: &'a Txn,
        _axes: Vec<usize>,
    ) -> TCResult<CoordStream<'a>> {
        // let columns: Vec<Id> = axes.iter().map(|x| (*x).into()).collect();
        //
        // let filled_at = self
        //     .table
        //     .group_by(columns.to_vec())?;
        //
        // let filled_at = filled_at.stream(txn.id().clone())
        //     .await?
        //     .map(|coord| unwrap_coord(&coord));
        //
        // let filled_at: TCTryStreamOld<Coord> = Box::pin(filled_at);
        // Ok(filled_at)
        unimplemented!()
    }

    async fn filled_count(&self, txn: &Txn) -> TCResult<u64> {
        self.table.count(txn.id()).await
    }

    async fn filled_in<'a>(&'a self, _txn: &'a Txn, _bounds: Bounds) -> TCResult<SparseStream<'a>> {
        // let source = slice_table(self.table.clone().into(), &bounds).await?;
        // let filled_in = source.stream(txn.id().clone()).await?.map(unwrap_row);
        // let filled_in: SparseStream = Box::pin(filled_in);
        // Ok(filled_in)
        unimplemented!()
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        let value = value.into_type(self.dtype);

        let key = coord
            .into_iter()
            .map(Number::from)
            .map(Value::Number)
            .collect();

        self.table
            .upsert(&txn_id, key, vec![Value::Number(value)])
            .await
    }
}

impl ReadValueAt for SparseTable {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(error::bad_request(
                    "Coordinate out of bounds",
                    Bounds::from(coord),
                ));
            }

            let selector: HashMap<Id, ColumnBound> = coord
                .iter()
                .enumerate()
                .map(|(axis, at)| (axis.into(), u64_to_value(*at).into()))
                .collect();

            let slice = self
                .table
                .clone()
                .slice(selector.into())?
                .select(vec![VALUE.into()])?;

            let mut slice = slice.stream(txn.id()).await?;

            let value = match slice.try_next().await? {
                Some(mut number) if number.len() == 1 => number.pop().unwrap().try_into(),
                None => Ok(self.dtype().zero()),
                _ => Err(error::internal(ERR_CORRUPT)),
            }?;

            Ok((coord, value))
        })
    }
}

#[async_trait]
impl Transact for SparseTable {
    async fn commit(&self, txn_id: &TxnId) {
        self.table.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.table.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.table.finalize(txn_id).await
    }
}

async fn slice_table(mut table: Table, bounds: &'_ Bounds) -> TCResult<Table> {
    use AxisBounds::*;

    for (axis, axis_bound) in bounds.to_vec().into_iter().enumerate() {
        let axis: Id = axis.into();
        let column_bound = match axis_bound {
            At(x) => table::ColumnBound::Is(u64_to_value(x)),
            In(range) => {
                let start = Bound::In(u64_to_value(range.start));
                let end = Bound::Ex(u64_to_value(range.end));
                (start, end).into()
            }
            _ => todo!(),
        };

        let bounds: HashMap<Id, ColumnBound> = iter::once((axis, column_bound)).collect();
        table = table.slice(bounds.into())?
    }

    Ok(table)
}

fn u64_to_value(u: u64) -> Value {
    Value::Number(Number::UInt(UInt::U64(u)))
}

fn unwrap_coord(coord: &[Value]) -> TCResult<Coord> {
    coord.iter().map(|val| unwrap_u64(val)).collect()
}

fn unwrap_row(mut row: Vec<Value>) -> TCResult<(Coord, Number)> {
    let coord = unwrap_coord(&row[0..row.len() - 1])?;
    if let Some(value) = row.pop() {
        Ok((coord, value.try_into()?))
    } else {
        Err(error::internal(ERR_CORRUPT))
    }
}

fn unwrap_u64(value: &Value) -> TCResult<u64> {
    if let Value::Number(Number::UInt(UInt::U64(unwrapped))) = value {
        Ok(*unwrapped)
    } else {
        Err(error::bad_request("Expected u64 but found", value))
    }
}
