use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};

use async_trait::async_trait;
use destream::de;
use futures::future::{self, TryFutureExt};
use futures::stream::TryStreamExt;

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_table::{Column, ColumnBound, TableIndex, TableInstance, TableSchema};
use tc_transact::fs::{Dir, File};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Number, NumberClass, NumberType, UInt, Value, ValueType};
use tcgeneric::{label, Id, Label};

use crate::{Bounds, Coord, Read, ReadValueAt, Shape, TensorAccess};

use super::{SparseAccess, SparseAccessor, SparseStream};

const VALUE: Label = label("value");
const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

#[derive(Clone)]
pub struct SparseTable<F: File<Node>, D: Dir, T: Transaction<D>> {
    table: TableIndex<F, D, T>,
    shape: Shape,
    dtype: NumberType,
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> SparseTable<F, D, T>
where
    F: TryFrom<D::File, Error = TCError>,
    D::FileClass: From<BTreeType>,
{
    pub async fn create(
        context: &D,
        txn_id: TxnId,
        shape: Shape,
        dtype: NumberType,
    ) -> TCResult<Self> {
        let table = Self::create_table(context, txn_id, shape.len(), dtype).await?;
        Ok(Self {
            table,
            dtype,
            shape,
        })
    }

    async fn create_table(
        context: &D,
        txn_id: TxnId,
        ndim: usize,
        dtype: NumberType,
    ) -> TCResult<TableIndex<F, D, T>> {
        let key: Vec<Column> = Self::key(ndim);
        let value: Vec<Column> = vec![(VALUE.into(), ValueType::Number(dtype)).into()];
        let indices = (0..ndim).map(|axis| (axis.into(), vec![axis.into()]));
        let schema = TableSchema::new((key, value).into(), indices);
        TableIndex::create(schema, context, txn_id).await
    }

    fn key(ndim: usize) -> Vec<Column> {
        let u64_type = NumberType::uint64();
        (0..ndim).map(|axis| (axis, u64_type).into()).collect()
    }
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> TensorAccess for SparseTable<F, D, T> {
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
impl<F: File<Node>, D: Dir, T: Transaction<D>> SparseAccess<F, D, T> for SparseTable<F, D, T> {
    fn accessor(self) -> SparseAccessor<F, D, T> {
        SparseAccessor::Table(self)
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rows = self.table.rows(*txn.id()).await?;
        let filled = rows.and_then(|row| future::ready(expect_row(row)));
        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_count(&self, txn: &T) -> TCResult<u64> {
        self.table.clone().count(*txn.id()).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        upsert_value(&self.table, txn_id, coord, value).await
    }
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> ReadValueAt<D> for SparseTable<F, D, T> {
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(TCError::bad_request(
                    "Coordinate out of bounds",
                    Bounds::from(coord),
                ));
            }

            read_value_at(self.table, txn, coord, self.dtype).await
        })
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> Transact for SparseTable<F, D, T>
where
    TableIndex<F, D, T>: Transact,
{
    async fn commit(&self, txn_id: &TxnId) {
        self.table.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.table.finalize(txn_id).await
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> de::FromStream for SparseTable<F, D, T>
where
    F: TryFrom<D::File, Error = TCError>,
    D::FileClass: From<BTreeType>,
{
    type Context = (Self, TxnId);

    async fn from_stream<De: de::Decoder>(
        context: (Self, TxnId),
        decoder: &mut De,
    ) -> Result<Self, De::Error> {
        let (table, txn_id) = context;

        decoder
            .decode_seq(SparseTableVisitor { table, txn_id })
            .await
    }
}

struct SparseTableVisitor<F: File<Node>, D: Dir, T: Transaction<D>>
where
    F: TryFrom<D::File>,
    D::FileClass: From<BTreeType>,
{
    table: SparseTable<F, D, T>,
    txn_id: TxnId,
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> de::Visitor for SparseTableVisitor<F, D, T>
where
    F: TryFrom<D::File>,
    D::FileClass: From<BTreeType>,
{
    type Value = SparseTable<F, D, T>;

    fn expecting() -> &'static str {
        "the contents of a SparseTensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        while let Some((coord, value)) = seq.next_element(()).await? {
            self.table
                .write_value(self.txn_id, coord, value)
                .map_err(de::Error::custom)
                .await?;
        }

        Ok(self.table)
    }
}

async fn read_value_at<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>(
    table: T,
    txn: Txn,
    coord: Coord,
    dtype: NumberType,
) -> TCResult<(Coord, Number)> {
    let selector: HashMap<Id, ColumnBound> = coord
        .iter()
        .enumerate()
        .map(|(axis, at)| (axis.into(), u64_into_value(*at).into()))
        .collect();

    let slice = table.slice(selector.into())?.select(vec![VALUE.into()])?;
    let mut slice = slice.rows(*txn.id()).await?;

    let value = match slice.try_next().await? {
        Some(mut number) => number.pop().unwrap().try_into()?,
        None => dtype.zero(),
    };

    Ok((coord, value))
}

async fn upsert_value<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>(
    table: &T,
    txn_id: TxnId,
    coord: Coord,
    value: Number,
) -> TCResult<()> {
    let key = coord
        .into_iter()
        .map(Number::from)
        .map(Value::Number)
        .collect();

    table.upsert(txn_id, key, vec![Value::Number(value)]).await
}

#[inline]
fn u64_into_value(u: u64) -> Value {
    Value::Number(Number::UInt(UInt::U64(u)))
}

#[inline]
fn expect_coord(coord: &[Value]) -> TCResult<Coord> {
    coord.iter().map(|val| expect_u64(val)).collect()
}

#[inline]
fn expect_row(mut row: Vec<Value>) -> TCResult<(Coord, Number)> {
    let coord = expect_coord(&row[0..row.len() - 1])?;
    if let Some(value) = row.pop() {
        Ok((coord, value.try_into()?))
    } else {
        Err(TCError::internal(ERR_CORRUPT))
    }
}

#[inline]
fn expect_u64(value: &Value) -> TCResult<u64> {
    if let Value::Number(Number::UInt(UInt::U64(unwrapped))) = value {
        Ok(*unwrapped)
    } else {
        Err(TCError::bad_request("Expected u64 but found", value))
    }
}
