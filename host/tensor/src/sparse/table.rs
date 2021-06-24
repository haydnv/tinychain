use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;

use afarray::Array;
use async_trait::async_trait;
use destream::de;
use futures::future::{self, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_table::{Column, ColumnBound, Merged, TableIndex, TableInstance, TableSchema};
use tc_transact::fs::{CopyFrom, Dir, File, Persist, Restore};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Bound, Number, NumberClass, NumberType, UInt, Value, ValueType};
use tcgeneric::{label, GroupStream, Id, Label};

use crate::stream::{sorted_coords, Read, ReadValueAt};
use crate::transform::{self, Rebase};
use crate::{AxisBounds, Bounds, Coord, Schema, Shape, TensorAccess, TensorType};

use super::access::SparseTranspose;
use super::{CoordStream, SparseAccess, SparseAccessor, SparseStream, SparseTensor};

const VALUE: Label = label("value");
const ERR_CORRUPT: &str = "SparseTensor corrupted! Please file a bug report.";

/// The base accessor type of [`SparseTensor`], implementing [`SparseAccess`] for a [`TableIndex`].
#[derive(Clone)]
pub struct SparseTable<FD, FS, D, T> {
    table: TableIndex<FS, D, T>,
    schema: Schema,
    dense: PhantomData<FD>,
}

impl<FD, FS, D, T> SparseTable<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<BTreeType>,
{
    /// Create a new `SparseTable` with the given [`Schema`].
    pub async fn create(context: &D, schema: Schema, txn_id: TxnId) -> TCResult<Self> {
        let table_schema = Self::table_schema(&schema);
        let table = TableIndex::create(context, table_schema, txn_id).await?;

        Ok(Self {
            table,
            schema,
            dense: PhantomData,
        })
    }

    fn table_schema(schema: &Schema) -> TableSchema {
        let ndim = schema.shape.len();
        let u64_type = NumberType::uint64();
        let key = (0..ndim).map(|axis| (axis, u64_type).into()).collect();
        let value: Vec<Column> = vec![(VALUE.into(), ValueType::Number(schema.dtype)).into()];
        let indices = (0..ndim).map(|axis| (axis.into(), vec![axis.into()]));
        TableSchema::new((key, value).into(), indices)
    }
}

impl<FD, FS, D, T> TensorAccess for SparseTable<FD, FS, D, T> {
    #[inline]
    fn dtype(&self) -> NumberType {
        self.schema.dtype
    }

    #[inline]
    fn ndim(&self) -> usize {
        self.schema.shape.len()
    }

    #[inline]
    fn shape(&'_ self) -> &'_ Shape {
        &self.schema.shape
    }

    #[inline]
    fn size(&self) -> u64 {
        self.schema.shape.size()
    }
}

#[async_trait]
impl<FD, FS, D, T> SparseAccess<FD, FS, D, T> for SparseTable<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Slice = SparseTableSlice<FD, FS, D, T>;
    type Transpose = SparseTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Table(self)
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rows = self.table.rows(*txn.id()).await?;
        let filled = rows.and_then(|row| future::ready(expect_row(row)));
        let filled: SparseStream = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        let shape = self.shape();
        let shape = axes.iter().map(|x| shape[*x]).collect();
        filled_at::<FD, FS, D, T, _>(&txn, shape, axes, self.table).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.table.count(*txn.id()).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let table = slice_table(self.table.clone(), &bounds)?;
        let rebase = transform::Slice::new(self.shape().clone(), bounds)?;
        Ok(SparseTableSlice::new(self, table, rebase))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        SparseTranspose::new(self, permutation)
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        upsert_value(&self.table, txn_id, coord, value).await
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseTable<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(TCError::bad_request(
                    "Coordinate out of bounds",
                    Bounds::from(coord),
                ));
            }

            let dtype = self.dtype();
            read_value_at(self.table, txn, coord, dtype).await
        })
    }
}

#[async_trait]
impl<FD, FS, D, T> Transact for SparseTable<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    TableIndex<FS, D, T>: Transact,
{
    async fn commit(&self, txn_id: &TxnId) {
        self.table.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.table.finalize(txn_id).await
    }
}

#[async_trait]
impl<FD, FS, D, T, A> CopyFrom<D, SparseTensor<FD, FS, D, T, A>> for SparseTable<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    async fn copy_from(
        instance: SparseTensor<FD, FS, D, T, A>,
        store: D,
        txn: T,
    ) -> TCResult<Self> {
        let shape = instance.shape().clone();
        let dtype = instance.dtype();
        let schema = Schema { shape, dtype };
        let accessor = SparseTable::create(&store, schema, *txn.id()).await?;

        let txn_id = *txn.id();
        let filled = instance.accessor.filled(txn).await?;

        filled
            .map_ok(|(coord, value)| accessor.write_value(txn_id, coord, value))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(accessor.into())
    }
}

#[async_trait]
impl<FD, FS, D, T> Persist<D> for SparseTable<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Schema = Schema;
    type Store = D;
    type Txn = T;

    fn schema(&self) -> &Self::Schema {
        &self.schema
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let table_schema = Self::table_schema(&schema);
        let table = TableIndex::load(txn, table_schema, store).await?;
        Ok(Self {
            table,
            schema,
            dense: PhantomData,
        })
    }
}

#[async_trait]
impl<FD, FS, D, T> Restore<D> for SparseTable<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        self.table.restore(&backup.table, txn_id).await
    }
}

#[async_trait]
impl<FD, FS, D, T> de::FromStream for SparseTable<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<BTreeType> + From<TensorType>,
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

#[derive(Clone)]
pub struct SparseTableSlice<FD, FS, D, T> {
    source: SparseTable<FD, FS, D, T>,
    table: Merged<FS, D, T>,
    rebase: transform::Slice,
}

impl<FD, FS, D, T> SparseTableSlice<FD, FS, D, T> {
    fn new(
        source: SparseTable<FD, FS, D, T>,
        table: Merged<FS, D, T>,
        rebase: transform::Slice,
    ) -> Self {
        Self {
            source,
            table,
            rebase,
        }
    }
}

impl<FD, FS, D, T> TensorAccess for SparseTableSlice<FD, FS, D, T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.rebase.ndim()
    }

    fn shape(&self) -> &Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.rebase.size()
    }
}

#[async_trait]
impl<FD, FS, D, T> SparseAccess<FD, FS, D, T> for SparseTableSlice<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Slice = Self;
    type Transpose = SparseTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Slice(self)
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rebase = self.rebase;
        let rows = self.table.rows(*txn.id()).await?;
        let filled = rows
            .and_then(|row| future::ready(expect_row(row)))
            .map_ok(move |(coord, value)| (rebase.map_coord(coord), value));

        let filled: SparseStream<'a> = Box::pin(filled);
        Ok(filled)
    }

    async fn filled_at<'a>(self, txn: T, axes: Vec<usize>) -> TCResult<CoordStream<'a>> {
        let shape = self.shape();
        let shape = axes.iter().map(|x| shape[*x]).collect();
        filled_at::<FD, FS, D, T, _>(&txn, shape, axes, self.table).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.table.count(*txn.id()).await
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.shape().validate_bounds(&bounds)?;
        let source_bounds = self.rebase.invert_bounds(bounds);
        self.source.slice(source_bounds)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        SparseTranspose::new(self, permutation)
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.rebase.invert_coord(&coord);
        self.source.write_value(txn_id, source_coord, value).await
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseTableSlice<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(TCError::bad_request(
                    "Coordinate out of bounds",
                    Bounds::from(coord),
                ));
            }

            let dtype = self.dtype();
            let source_coord = self.rebase.invert_coord(&coord);
            read_value_at(self.table, txn, source_coord, dtype).await
        })
    }
}

struct SparseTableVisitor<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node> + TryFrom<D::File>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<BTreeType>,
{
    table: SparseTable<FD, FS, D, T>,
    txn_id: TxnId,
}

#[async_trait]
impl<FD, FS, D, T> de::Visitor for SparseTableVisitor<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Value = SparseTable<FD, FS, D, T>;

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

async fn filled_at<'a, FD, FS, D, Txn, T>(
    txn: &Txn,
    shape: Shape,
    axes: Vec<usize>,
    table: T,
) -> TCResult<CoordStream<'a>>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    Txn: Transaction<D>,
    T: TableInstance<FS, D, Txn>,
    D::FileClass: From<TensorType>,
{
    let coords = table.select(axes.into_iter().map(Id::from).collect())?;
    let coords = coords.rows(*txn.id()).await?;
    let coords = coords.map(|r| r.and_then(expect_coord));
    let coords = sorted_coords::<FD, FS, D, Txn, _>(txn, &shape, coords).await?;
    Ok(Box::pin(GroupStream::from(coords)))
}

fn slice_table<F: File<Node>, D: Dir, Txn: Transaction<D>, T: TableInstance<F, D, Txn>>(
    table: T,
    bounds: &Bounds,
) -> TCResult<T::Slice> {
    use AxisBounds::*;

    let mut table_bounds = HashMap::new();
    for (axis, axis_bound) in bounds.to_vec().into_iter().enumerate() {
        let axis: Id = axis.into();
        let column_bound = match axis_bound {
            At(x) => ColumnBound::Is(u64_into_value(x)),
            In(range) => {
                let start = Bound::In(u64_into_value(range.start));
                let end = Bound::Ex(u64_into_value(range.end));
                (start, end).into()
            }
            _ => todo!(),
        };

        table_bounds.insert(axis, column_bound);
    }

    table.slice(table_bounds.into())
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
fn expect_coord(coord: Vec<Value>) -> TCResult<Coord> {
    coord.into_iter().map(|val| expect_u64(val)).collect()
}

#[inline]
fn expect_row(mut row: Vec<Value>) -> TCResult<(Coord, Number)> {
    if let Some(value) = row.pop() {
        let value = value.try_into()?;
        expect_coord(row).map(|coord| (coord, value))
    } else {
        Err(TCError::internal(ERR_CORRUPT))
    }
}

#[inline]
fn expect_u64(value: Value) -> TCResult<u64> {
    if let Value::Number(Number::UInt(UInt::U64(unwrapped))) = value {
        Ok(unwrapped)
    } else {
        Err(TCError::bad_request("Expected u64 but found", value))
    }
}
