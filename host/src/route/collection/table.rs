use futures::{future, StreamExt, TryFutureExt, TryStreamExt};
use log::debug;
use safecast::*;

use tc_error::*;
use tc_table::{
    Bounds, ColumnBound, Key, TableInstance, TableOrder, TableRead, TableSlice, TableStream,
    TableType, TableWrite,
};
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tc_value::{Bound, Value};
use tcgeneric::{label, Id, Map, PathSegment};

use crate::collection::{Collection, Table, TableIndex};
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::Scalar;
use crate::state::State;
use crate::stream::{Source, TCStream};

impl Route for TableType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if self == &Self::default() {
            Static.route(path)
        } else {
            None
        }
    }
}

struct CopyHandler;

impl<'a> Handler<'a> for CopyHandler {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let schema = tc_table::TableSchema::try_cast_from(schema, |v| {
                    TCError::bad_request("invalid Table schema", v)
                })?;

                let source: TCStream = params.require(&label("source").into())?;
                params.expect_empty()?;

                let txn_id = *txn.id();

                let dir = txn.context().create_dir_unique(*txn.id()).await?;
                let table = TableIndex::create(&dir, schema, *txn.id()).await?;

                let rows = source.into_stream(txn.clone()).await?;
                rows.map(|r| {
                    r.and_then(|state| {
                        Value::try_cast_from(state, |s| {
                            TCError::bad_request("invalid Table row", s)
                        })
                    })
                })
                .map(|r| {
                    r.and_then(|value| {
                        value.try_cast_into(|v| TCError::bad_request("invalid Table row", v))
                    })
                })
                .map(|r| r.and_then(|row| table.schema().primary().key_values_from_tuple(row)))
                .map_ok(|(key, values)| table.upsert(txn_id, key, values))
                .try_buffer_unordered(num_cpus::get())
                .try_fold((), |(), ()| future::ready(Ok(())))
                .await?;

                Ok(State::Collection(table.into()))
            })
        }))
    }
}

struct CreateHandler;

impl<'a> Handler<'a> for CreateHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, value| {
            Box::pin(async move {
                let schema = tc_table::TableSchema::try_cast_from(value, |v| {
                    TCError::bad_request("invalid Table schema", v)
                })?;

                let dir = txn.context().create_dir_unique(*txn.id()).await?;
                TableIndex::create(&dir, schema, *txn.id())
                    .map_ok(Collection::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

struct ContainsHandler<'a, T> {
    table: &'a T,
}

impl<'a, T: TableRead + 'a> Handler<'a> for ContainsHandler<'a, T> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let key = primary_key(key, self.table)?;
                let row = self.table.read(txn.id(), &key).await?;
                Ok(Value::from(row.is_some()).into())
            })
        }))
    }
}

impl<'a, T> From<&'a T> for ContainsHandler<'a, T> {
    fn from(table: &'a T) -> Self {
        Self { table }
    }
}

struct CountHandler<T> {
    table: T,
}

impl<'a, T: TableSlice + TableStream + 'a> Handler<'a> for CountHandler<T>
where
    T::Slice: TableStream,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    self.table.count(*txn.id()).map_ok(State::from).await
                } else {
                    let bounds = cast_into_bounds(Scalar::Value(key))?;
                    let slice = self.table.slice(bounds)?;
                    slice.count(*txn.id()).map_ok(State::from).await
                }
            })
        }))
    }
}

impl<T> From<T> for CountHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

struct LimitHandler<T> {
    table: T,
}

impl<'a, T: TableStream + 'a> Handler<'a> for LimitHandler<T>
where
    Table: From<T::Limit>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let limit = key.try_cast_into(|v| {
                    TCError::bad_request("limit must be a positive integer, not", v)
                })?;

                Ok(Collection::Table(self.table.limit(limit).into()).into())
            })
        }))
    }
}

impl<T> From<T> for LimitHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

struct LoadHandler;

impl<'a> Handler<'a> for LoadHandler {
    // TODO
}

struct OrderHandler<T> {
    table: T,
}

impl<'a, T: TableOrder + 'a> Handler<'a> for OrderHandler<T>
where
    Table: From<T::OrderBy>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let ordered = if key.matches::<(Vec<Id>, bool)>() {
                    let (order, reverse) = key.opt_cast_into().unwrap();
                    self.table.order_by(order, reverse)?
                } else if key.matches::<Vec<Id>>() {
                    let order = key.opt_cast_into().unwrap();
                    self.table.order_by(order, false)?
                } else {
                    return Err(TCError::bad_request("invalid column list to order by", key));
                };

                Ok(Collection::Table(ordered.into()).into())
            })
        }))
    }
}

impl<T> From<T> for OrderHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

struct TableHandler<'a, T> {
    table: &'a T,
}

impl<'a, T> Handler<'a> for TableHandler<'a, T>
where
    T: TableRead + TableSlice + TableWrite + Clone + 'a,
    Table: From<T>,
    Table: From<T::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    let key = primary_key(key.clone(), self.table)?;
                    self.table
                        .read(txn.id(), &key)
                        .map_ok(Value::from)
                        .map_ok(State::from)
                        .await
                } else {
                    Ok(Collection::Table(self.table.clone().into()).into())
                }
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, values| {
            Box::pin(async move {
                debug!("Table PUT {:?} <- {:?}", key, values);

                let key = primary_key(key, self.table)?;

                if values.is_map() {
                    let values =
                        values.try_into_map(|s| TCError::bad_request("invalid row values", s))?;

                    let values = values
                        .into_iter()
                        .map(|(id, state)| {
                            Value::try_cast_from(state, |s| {
                                TCError::bad_request("invalid column value", s)
                            })
                            .map(|value| (id, value))
                        })
                        .collect::<TCResult<Map<Value>>>()?;

                    self.table.update(*txn.id(), key, values).await
                } else if values.is_tuple() {
                    let values =
                        values.try_into_tuple(|s| TCError::bad_request("invalid row values", s))?;

                    let values = values
                        .into_iter()
                        .map(|state| {
                            state.try_cast_into(|s| TCError::bad_request("invalid column value", s))
                        })
                        .collect::<TCResult<Vec<Value>>>()?;

                    self.table.upsert(*txn.id(), key, values).await
                } else {
                    Err(TCError::bad_request("invalid row values", values))
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let bounds: Scalar = params.require(&label("bounds").into())?;
                let bounds = cast_into_bounds(bounds)?;
                self.table
                    .clone()
                    .slice(bounds)
                    .map(|slice| Table::from(slice))
                    .map(Collection::Table)
                    .map(State::Collection)
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let row = primary_key(key, self.table)?;
                self.table.delete(*txn.id(), row).await
            })
        }))
    }
}

struct SchemaHandler<'a, T> {
    table: &'a T,
    schema: fn(&'a T) -> Value,
}

impl<'a, T> SchemaHandler<'a, T> {
    fn new(table: &'a T, schema: fn(&'a T) -> Value) -> Self {
        Self { table, schema }
    }
}

impl<'a, T: TableInstance> Handler<'a> for SchemaHandler<'a, T> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                key.expect_none()?;
                Ok(State::from((self.schema)(self.table)))
            })
        }))
    }
}

struct SelectHandler<T> {
    table: T,
}

impl<'a, T: TableStream + 'a> Handler<'a> for SelectHandler<T>
where
    Table: From<T::Selection>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let columns =
                    key.try_cast_into(|v| TCError::bad_request("invalid column list", v))?;

                Ok(Collection::Table(self.table.select(columns)?.into()).into())
            })
        }))
    }
}

impl<T> From<T> for SelectHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

struct StreamHandler<T> {
    table: T,
}

impl<'a, T: TableSlice + 'a> Handler<'a> for StreamHandler<T>
where
    Table: From<T>,
    Table: From<T::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(TCStream::from(Table::from(self.table)).into())
                } else {
                    let bounds = cast_into_bounds(Scalar::Value(key))?;
                    let slice = self.table.slice(bounds)?;
                    Ok(TCStream::from(Table::from(slice)).into())
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let bounds = Scalar::try_cast_from(State::Map(params), |s| {
                    TCError::bad_request("invalid Table bounds", s)
                })?;

                let bounds = cast_into_bounds(bounds)?;

                let slice = self.table.slice(bounds)?;
                Ok(TCStream::from(Table::from(slice)).into())
            })
        }))
    }
}

impl<T> From<T> for StreamHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

impl<'a, T> From<&'a T> for TableHandler<'a, T> {
    fn from(table: &'a T) -> Self {
        Self { table }
    }
}

impl Route for Table {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

impl Route for TableIndex {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        route(self, path)
    }
}

#[inline]
fn route<'a, T>(table: &'a T, path: &[PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>>
where
    T: TableRead + TableOrder + TableSlice + TableStream + TableWrite + Clone,
    T::Slice: TableStream,
    Table: From<T>,
    Table: From<T::Limit>,
    Table: From<T::OrderBy>,
    Table: From<T::Selection>,
    Table: From<T::Slice>,
{
    if path.is_empty() {
        Some(Box::new(TableHandler::from(table)))
    } else if path.len() == 1 {
        match path[0].as_str() {
            "columns" => Some(Box::new(SchemaHandler::new(table, column_schema))),
            "contains" => Some(Box::new(ContainsHandler::from(table))),
            "count" => Some(Box::new(CountHandler::from(table.clone()))),
            "key_columns" => Some(Box::new(SchemaHandler::new(table, key_columns))),
            "key_names" => Some(Box::new(SchemaHandler::new(table, key_names))),
            "limit" => Some(Box::new(LimitHandler::from(table.clone()))),
            "order" => Some(Box::new(OrderHandler::from(table.clone()))),
            "select" => Some(Box::new(SelectHandler::from(table.clone()))),
            "rows" => Some(Box::new(StreamHandler::from(table.clone()))),
            _ => None,
        }
    } else {
        None
    }
}

pub struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateHandler))
        } else if path == &["copy_from"] {
            Some(Box::new(CopyHandler))
        } else {
            None
        }
    }
}

#[inline]
fn cast_into_bounds(scalar: Scalar) -> TCResult<Bounds> {
    if scalar.is_none() {
        return Ok(Bounds::default());
    }

    let scalar = Map::<Value>::try_cast_from(scalar, |s| {
        TCError::bad_request("invalid selection bounds for Table", s)
    })?;

    scalar
        .into_iter()
        .map(|(col_name, bound)| {
            if bound.matches::<(Bound, Bound)>()
                || bound.matches::<(Bound, Value)>()
                || bound.matches::<(Value, Bound)>()
            {
                Ok(ColumnBound::In(bound.opt_cast_into().unwrap()))
            } else if bound.matches::<Value>() {
                Ok(ColumnBound::Is(bound.opt_cast_into().unwrap()))
            } else {
                Err(TCError::bad_request("invalid column bound", bound))
            }
            .map(|bound| (col_name, bound))
        })
        .collect()
}

#[inline]
fn primary_key<T: TableInstance>(key: Value, table: &T) -> TCResult<Key> {
    let key: Vec<Value> = key.try_cast_into(|v| TCError::bad_request("invalid Table key", v))?;
    table.schema().primary().validate_key(key)
}

fn column_schema<T: TableInstance>(table: &T) -> Value {
    let columns = table
        .schema()
        .primary()
        .columns()
        .into_iter()
        .map(Value::from)
        .collect();

    Value::Tuple(columns)
}

fn key_columns<T: TableInstance>(table: &T) -> Value {
    let key = table
        .key()
        .iter()
        .cloned()
        .map(|column| Value::from(column))
        .collect();

    Value::Tuple(key)
}

fn key_names<T: TableInstance>(table: &T) -> Value {
    let key = table
        .key()
        .iter()
        .map(|col| &col.name)
        .cloned()
        .map(Value::Id)
        .collect();

    Value::Tuple(key)
}
