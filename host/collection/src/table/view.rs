use std::sync::Arc;

use async_trait::async_trait;
use ds_ext::Id;
use safecast::AsType;

use tc_error::TCResult;
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{Instance, ThreadSafe};

use crate::Node;

use super::file::TableFile;
use super::stream::Rows;
use super::{
    Key, Range, Schema, Table, TableInstance, TableOrder, TableRead, TableStream, TableType,
};

/// A result set from a database table with a limited number of rows
#[derive(Clone)]
pub struct Limited<T> {
    source: T,
    limit: u64,
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
    fn schema(&self) -> &Schema {
        self.source.schema()
    }
}

#[async_trait]
impl<T: TableStream> TableStream for Limited<T> {
    type Limit = Self;
    type Selection = Limited<Selection<T>>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        todo!()
    }

    fn limit(self, limit: u64) -> Self::Limit {
        todo!()
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        todo!()
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>> {
        todo!()
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

/// A result set from a database table with a limited set of columns
#[derive(Clone)]
pub struct Selection<T> {
    source: T,
    schema: Schema,
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
    fn schema(&self) -> &Schema {
        &self.schema
    }
}

impl<T: TableOrder> TableOrder for Selection<T> {
    type OrderBy = Selection<T::OrderBy>;
    type Reverse = Selection<T::Reverse>;

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        todo!()
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        todo!()
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        todo!()
    }
}

impl<T: super::TableSlice> super::TableSlice for Selection<T> {
    type Slice = Self;

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        todo!()
    }

    fn validate_range(&self, range: &Range) -> TCResult<()> {
        todo!()
    }
}

#[async_trait]
impl<T: TableStream> TableStream for Selection<T> {
    type Limit = Limited<Self>;
    type Selection = Self;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        todo!()
    }

    fn limit(self, limit: u64) -> Self::Limit {
        todo!()
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        todo!()
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>> {
        todo!()
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

/// A slice of a relational database table
pub struct TableSlice<Txn, FE> {
    table: TableFile<Txn, FE>,
    range: Arc<Range>,
    reverse: bool,
}

impl<Txn, FE> Clone for TableSlice<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            range: self.range.clone(),
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
    fn schema(&self) -> &Schema {
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
        todo!()
    }

    fn reverse(self) -> TCResult<Self::Reverse> {
        todo!()
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        todo!()
    }
}

#[async_trait]
impl<Txn, FE> TableRead for TableSlice<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    async fn read(&self, txn_id: &TxnId, key: &Key) -> TCResult<Option<Vec<Value>>> {
        todo!()
    }
}

impl<Txn, FE> super::TableSlice for TableSlice<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Slice = Self;

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        todo!()
    }

    fn validate_range(&self, range: &Range) -> TCResult<()> {
        todo!()
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
        todo!()
    }

    fn limit(self, limit: u64) -> Self::Limit {
        todo!()
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        todo!()
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<Rows<'a>> {
        todo!()
    }
}
