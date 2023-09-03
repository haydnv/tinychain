use std::fmt;

use async_trait::async_trait;
use collate::OverlapsRange;
use futures::{future, TryFutureExt, TryStreamExt};
use safecast::AsType;

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tcgeneric::{Id, Instance, ThreadSafe};

use crate::btree::{BTreeSchema, Node};

use super::file::TableFile;
use super::schema::TableSchema;
use super::stream::Rows;
use super::{Key, Range, Row, Table, TableInstance, TableOrder, TableRead, TableStream, TableType};

/// A result set from a database table with a limited number of rows
#[derive(Clone)]
pub struct Limited<T> {
    source: T,
    limit: usize,
}

impl<T> Limited<T> {
    pub(super) fn new(source: T, limit: u64) -> TCResult<Self> {
        usize::try_from(limit)
            .map(|limit| Self { source, limit })
            .map_err(|cause| bad_request!("limit {} is too large: {}", limit, cause))
    }
}

impl<T> Instance for Limited<T>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        TableType::Limit
    }
}

impl<T: TableInstance> TableInstance for Limited<T> {
    fn schema(&self) -> &TableSchema {
        self.source.schema()
    }
}

#[async_trait]
impl<T: TableStream> TableStream for Limited<T> {
    type Limit = Self;
    type Selection = Limited<T::Selection>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        let rows = self.rows(txn_id).await?;
        rows.try_fold(0, |count, _row| future::ready(Ok(count + 1)))
            .await
    }

    fn limit(self, limit: u64) -> TCResult<Self::Limit> {
        usize::try_from(limit)
            .map(|limit| Self {
                source: self.source,
                limit,
            })
            .map_err(|cause| bad_request!("limit {} is too large: {}", limit, cause))
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        self.source.select(columns).map(|source| Limited {
            source,
            limit: self.limit,
        })
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>> {
        self.source
            .rows(txn_id)
            .map_ok(|rows| rows.limit(self.limit))
            .await
    }
}

impl<Txn, FE, T> From<Limited<T>> for Table<Txn, FE>
where
    T: Into<Table<Txn, FE>>,
{
    fn from(limited: Limited<T>) -> Self {
        Table::Limited(Box::new(Limited {
            source: limited.source.into(),
            limit: limited.limit,
        }))
    }
}

impl<T> fmt::Debug for Limited<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "the first {} rows of {:?}", self.limit, self.source)
    }
}

/// A result set from a database table with a limited set of columns
#[derive(Clone)]
pub struct Selection<T> {
    source: T,
    schema: TableSchema,
}

impl<T: TableInstance> Selection<T> {
    pub(super) fn new(source: T, columns: Vec<Id>) -> TCResult<Self> {
        let mut column_schema = Vec::with_capacity(columns.len());
        for name in columns.into_iter() {
            let mut present = false;
            let schema = b_table::Schema::primary(source.schema());
            for column in schema {
                if column.name == name {
                    column_schema.push(column.clone());
                    present = true;
                    break;
                }
            }

            if !present {
                return Err(bad_request!(
                    "cannot select column {} from {:?}",
                    name,
                    schema
                ));
            }
        }

        let schema = BTreeSchema::new(column_schema)?;

        Ok(Self {
            source,
            schema: TableSchema::from_index(schema),
        })
    }
}

impl<T> Instance for Selection<T>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        TableType::Selection
    }
}

impl<T: TableInstance> TableInstance for Selection<T> {
    fn schema(&self) -> &TableSchema {
        &self.schema
    }
}

impl<T: TableOrder> TableOrder for Selection<T> {
    type OrderBy = Selection<T::OrderBy>;
    type Reverse = Selection<T::Reverse>;

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.source
            .order_by(columns, reverse)
            .map(|source| Selection {
                source,
                schema: self.schema,
            })
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        self.source.reverse().map(|source| Selection {
            source,
            schema: self.schema,
        })
    }
}

#[async_trait]
impl<T: TableRead> TableRead for Selection<T> {
    async fn read(&self, txn_id: TxnId, key: Key) -> TCResult<Option<Row>> {
        let schema = b_table::Schema::primary(self.schema());
        let source_schema = b_table::Schema::primary(self.source.schema());

        if let Some(row) = self.source.read(txn_id, key).await? {
            let row = b_table::IndexSchema::extract_key(schema, &row, source_schema);
            Ok(Some(row))
        } else {
            Ok(None)
        }
    }
}

impl<T: super::TableSlice> super::TableSlice for Selection<T> {
    type Slice = Selection<<T as super::TableSlice>::Slice>;

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        self.source.slice(range).map(|source| Selection {
            source,
            schema: self.schema,
        })
    }
}

#[async_trait]
impl<T: TableStream + fmt::Debug> TableStream for Selection<T> {
    type Limit = Limited<Self>;
    type Selection = Self;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.source.count(txn_id).await
    }

    fn limit(self, limit: u64) -> TCResult<Self::Limit> {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        for name in &columns {
            let selected = b_table::IndexSchema::columns(b_table::Schema::primary(self.schema()));
            if !selected.contains(name) {
                return Err(bad_request!(
                    "cannot select column {} from {:?}",
                    name,
                    self
                ));
            }
        }

        Selection::new(self.source, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>> {
        let source_schema = self.source.schema().clone();

        self.source
            .rows(txn_id)
            .map_ok(move |rows| rows.select(source_schema, self.schema))
            .await
    }
}

impl<Txn, FE, T> From<Selection<T>> for Table<Txn, FE>
where
    T: Into<Table<Txn, FE>>,
{
    fn from(selection: Selection<T>) -> Self {
        Table::Selection(Box::new(Selection {
            source: selection.source.into(),
            schema: selection.schema,
        }))
    }
}

impl<T> fmt::Debug for Selection<T>
where
    T: TableInstance + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let columns = b_table::IndexSchema::columns(b_table::Schema::primary(&self.schema));
        write!(
            f,
            "a selection of columns {:?} from {:?}",
            columns, self.source
        )
    }
}

/// A slice of a relational database table
pub struct TableSlice<Txn, FE> {
    table: TableFile<Txn, FE>,
    range: Range,
    order: Vec<Id>,
    reverse: bool,
}

impl<Txn, FE> TableSlice<Txn, FE>
where
    TableFile<Txn, FE>: TableInstance,
{
    pub(super) fn new(
        table: TableFile<Txn, FE>,
        range: Range,
        order: Vec<Id>,
        reverse: bool,
    ) -> TCResult<Self> {
        // verify that the requested range is supported
        if !b_table::IndexSchema::supports(b_table::Schema::primary(table.schema()), &range) {
            let mut supported = false;
            for (_, index) in b_table::Schema::auxiliary(table.schema()) {
                supported = supported || b_table::IndexSchema::supports(index, &range);
            }

            if !supported {
                return Err(bad_request!(
                    "{:?} has no index which supports {:?}",
                    table,
                    range
                ));
            }
        }

        // verify that the requested order is supported
        if !b_table::IndexSchema::columns(b_table::Schema::primary(table.schema()))
            .starts_with(&order)
        {
            let mut supported = false;
            for (_, index) in b_table::Schema::auxiliary(table.schema()) {
                supported = supported || b_table::IndexSchema::columns(index).starts_with(&order);
            }
        }

        Ok(Self {
            table,
            range,
            order,
            reverse,
        })
    }
}

impl<Txn, FE> Clone for TableSlice<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            range: self.range.clone(),
            order: self.order.clone(),
            reverse: self.reverse,
        }
    }
}

impl<Txn, FE> Instance for TableSlice<Txn, FE>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> Self::Class {
        TableType::Slice
    }
}

impl<Txn, FE> TableInstance for TableSlice<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    fn schema(&self) -> &TableSchema {
        self.table.schema()
    }
}

impl<Txn, FE> TableOrder for TableSlice<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type OrderBy = Self;
    type Reverse = Self;

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        if self.order.starts_with(&columns) {
            Ok(Self {
                table: self.table,
                order: self.order,
                range: self.range,
                reverse: self.reverse ^ reverse,
            })
        } else {
            Err(bad_request!(
                "cannot reorder a table slice--consider re-slicing the table instead"
            ))
        }
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        Ok(Self {
            table: self.table,
            order: self.order,
            range: self.range,
            reverse: !self.reverse,
        })
    }
}

#[async_trait]
impl<Txn, FE> TableRead for TableSlice<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn read(&self, txn_id: TxnId, key: Key) -> TCResult<Option<Row>> {
        let range = self.schema().range_from_key(key.clone())?;

        if self.range.contains(&range, self.table.collator().inner()) {
            self.table.read(txn_id, key).await
        } else {
            Err(bad_request!("key {:?} does not lie within {:?}", key, self))
        }
    }
}

impl<Txn, FE> super::TableSlice for TableSlice<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Slice = Self;

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        if self.range.contains(&range, self.table.collator().inner()) {
            Ok(Self {
                table: self.table,
                range,
                order: self.order,
                reverse: self.reverse,
            })
        } else {
            Err(bad_request!(
                "slice range {:?} does not lie within {:?}",
                range,
                self
            ))
        }
    }
}

#[async_trait]
impl<Txn, FE> TableStream for TableSlice<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Limit = Limited<Self>;
    type Selection = Selection<Self>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        let rows = self.rows(txn_id).await?;
        rows.try_fold(0, |count, _row| future::ready(Ok(count + 1)))
            .await
    }

    fn limit(self, limit: u64) -> TCResult<Self::Limit> {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>> {
        self.table
            .into_stream(txn_id, self.range, self.order, self.reverse)
            .await
    }
}

impl<Txn, FE> fmt::Debug for TableSlice<Txn, FE>
where
    TableFile<Txn, FE>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a slice of {:?} with order {:?} and range {:?}",
            self.table, self.order, self.range
        )
    }
}
