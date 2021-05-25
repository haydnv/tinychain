use std::convert::TryFrom;
use std::iter::FromIterator;

use futures::{future, TryFutureExt, TryStreamExt};
use log::debug;
use safecast::*;

use tc_btree::Node;
use tc_error::*;
use tc_table::{Bounds, ColumnBound, TableInstance, TableType};
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tc_value::{Range, Value};
use tcgeneric::{Id, Map, PathSegment, Tuple};

use crate::collection::{Collection, Table, TableIndex};
use crate::fs;
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::Scalar;
use crate::state::State;
use crate::txn::Txn;

struct CreateHandler;

impl<'a> Handler<'a> for CreateHandler {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, value| {
            Box::pin(async move {
                let schema = tc_table::TableSchema::try_cast_from(value, |v| {
                    TCError::bad_request("invalid Table schema", v)
                })?;

                let dir = txn.context().create_dir_tmp(*txn.id()).await?;
                TableIndex::create(schema, &dir, *txn.id())
                    .map_ok(Collection::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

impl Route for TableType {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() && self == &Self::Table {
            Some(Box::new(CreateHandler))
        } else {
            None
        }
    }
}

struct ContainsHandler<T> {
    table: T,
}

impl<'a, T: TableInstance<fs::File<Node>, fs::Dir, Txn> + 'a> Handler<'a> for ContainsHandler<T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let key = primary_key(key, &self.table)?;
                let slice = self.table.slice(key)?;

                let mut rows = slice.rows(*txn.id()).await?;
                rows.try_next()
                    .map_ok(|row| row.is_some())
                    .map_ok(Value::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

impl<T> From<T> for ContainsHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

struct CountHandler<T> {
    table: T,
}

impl<'a, T: TableInstance<fs::File<Node>, fs::Dir, Txn> + 'a> Handler<'a> for CountHandler<T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "Table::count does not accept a key (call Table::slice first)",
                        key,
                    ));
                }

                self.table.count(*txn.id()).map_ok(State::from).await
            })
        }))
    }
}

impl<T> From<T> for CountHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

struct GroupHandler<T> {
    table: T,
}

impl<'a, T: TableInstance<fs::File<Node>, fs::Dir, Txn> + 'a> Handler<'a> for GroupHandler<T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let columns = key.try_cast_into(|v| {
                    TCError::bad_request("invalid column list to group by", v)
                })?;

                let grouped = self.table.group_by(columns)?;
                Ok(Collection::Table(grouped.into()).into())
            })
        }))
    }
}

impl<T> From<T> for GroupHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

struct LimitHandler<T> {
    table: T,
}

impl<'a, T: TableInstance<fs::File<Node>, fs::Dir, Txn> + 'a> Handler<'a> for LimitHandler<T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
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

struct OrderHandler<T> {
    table: T,
}

impl<'a, T: TableInstance<fs::File<Node>, fs::Dir, Txn> + 'a> Handler<'a> for OrderHandler<T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
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

impl<'a, T: TableInstance<fs::File<Node>, fs::Dir, Txn>> Handler<'a> for TableHandler<'a, T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    let bounds = primary_key(key.clone(), self.table)?;
                    let slice = self.table.clone().slice(bounds)?;
                    let mut rows = slice.rows(*txn.id()).await?;
                    if let Some(row) = rows.try_next().await? {
                        let schema = self.table.schema();
                        let columns = schema.primary().column_names().cloned();

                        Ok(State::Scalar(
                            Map::<Scalar>::from_iter(
                                columns.zip(row.into_iter().map(Scalar::Value)),
                            )
                            .into(),
                        ))
                    } else {
                        Err(TCError::not_found(key))
                    }
                } else {
                    Ok(Collection::Table(self.table.clone().into()).into())
                }
            })
        }))
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, values| {
            Box::pin(async move {
                if key.is_none() {
                    if let State::Collection(Collection::Table(table)) = values {
                        let txn_id = *txn.id();
                        let key_len = self.table.key().len();
                        let rows = table.rows(txn_id).await?;

                        return rows
                            .map_ok(|mut row| (row.drain(..key_len).collect(), row))
                            .map_ok(|(key, values)| self.table.upsert(txn_id, key, values))
                            .try_buffer_unordered(num_cpus::get())
                            .try_fold((), |(), ()| future::ready(Ok(())))
                            .await;
                    }
                }

                if values.matches::<Map<Value>>() {
                    let values = values.opt_cast_into().unwrap();
                    let bounds = cast_into_bounds(Scalar::Value(key))?;
                    self.table.clone().slice(bounds)?.update(&txn, values).await
                } else if values.matches::<Value>() {
                    let key = key
                        .try_cast_into(|k| TCError::bad_request("invalid key for Table row", k))?;

                    let values = Value::try_cast_from(values, |s| {
                        TCError::bad_request("invalid values for Table row", s)
                    })?;

                    let values = values.try_cast_into(|v| {
                        TCError::bad_request("invalid values for Table row", v)
                    })?;

                    self.table.upsert(*txn.id(), key, values).await
                } else {
                    Err(TCError::bad_request("invalid row value", values))
                }
            })
        }))
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let bounds = Scalar::try_cast_from(State::Map(params), |s| {
                    TCError::bad_request("invalid Table bounds", s)
                })?;

                let bounds = cast_into_bounds(bounds)?;
                debug!("slice Table with bounds {}", bounds);

                let table = self.table.clone().slice(bounds).map(|slice| slice.into())?;
                Ok(Collection::Table(table).into())
            })
        }))
    }

    fn delete(self: Box<Self>) -> Option<DeleteHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    let key = primary_key(key, self.table)?;
                    self.table.clone().slice(key)?.delete(*txn.id()).await
                } else {
                    self.table.delete(*txn.id()).await
                }
            })
        }))
    }
}

struct SelectHandler<T> {
    table: T,
}

impl<'a, T: TableInstance<fs::File<Node>, fs::Dir, Txn> + 'a> Handler<'a> for SelectHandler<T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
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
fn route<'a, T: TableInstance<fs::File<Node>, fs::Dir, Txn>>(
    table: &'a T,
    path: &[PathSegment],
) -> Option<Box<dyn Handler<'a> + 'a>> {
    if path.is_empty() {
        Some(Box::new(TableHandler::from(table)))
    } else if path.len() == 1 {
        match path[0].as_str() {
            "contains" => Some(Box::new(ContainsHandler::from(table.clone()))),
            "count" => Some(Box::new(CountHandler::from(table.clone()))),
            "limit" => Some(Box::new(LimitHandler::from(table.clone()))),
            "group" => Some(Box::new(GroupHandler::from(table.clone()))),
            "order" => Some(Box::new(OrderHandler::from(table.clone()))),
            "select" => Some(Box::new(SelectHandler::from(table.clone()))),
            _ => None,
        }
    } else {
        None
    }
}

#[inline]
fn cast_into_bounds(scalar: Scalar) -> TCResult<Bounds> {
    let scalar = Map::<Scalar>::try_from(scalar)?;
    scalar
        .into_iter()
        .map(|(col_name, bound)| {
            if bound.matches::<Range>() {
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
fn primary_key<T: TableInstance<fs::File<Node>, fs::Dir, Txn>>(key: Value, table: &T) -> TCResult<Bounds> {
    let key: Vec<Value> = key.try_cast_into(|v| TCError::bad_request("invalid Table key", v))?;

    if key.len() == table.key().len() {
        let bounds = table
            .key()
            .iter()
            .map(|col| col.name.clone())
            .zip(key)
            .collect();

        Ok(bounds)
    } else {
        Err(TCError::bad_request(
            "invalid primary key",
            Tuple::from(key),
        ))
    }
}
