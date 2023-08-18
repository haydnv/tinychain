use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::ops::Bound;

use b_table::{IndexSchema, Schema};
use futures::TryFutureExt;
use log::debug;
use safecast::*;

use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::fs::{Dir, Persist};
use tc_transact::public::{
    DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route, StateInstance,
};
use tc_transact::Transaction;
use tc_value::{Value, ValueCollator};
use tcgeneric::{label, Id, Map, PathSegment, ThreadSafe, Tuple};

use crate::btree::{BTreeSchema, Node};
use crate::table::TableUpdate;
use crate::Collection;

use super::{
    ColumnRange, Key, Range, Table, TableFile, TableInstance, TableOrder, TableRead, TableSchema,
    TableSlice, TableStream, TableType, TableWrite, Values,
};

impl<State> Route<State> for TableType
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    TableFile<State::Txn, State::FE>: Persist<State::FE, Schema = TableSchema, Txn = State::Txn>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if self == &Self::default() {
            Static.route(path)
        } else {
            None
        }
    }
}

struct CopyHandler;

impl<'a, State> Handler<'a, State> for CopyHandler
where
    State: StateInstance,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let _schema = TableSchema::try_cast_from_value(schema)?;

                // let _source = params.require(&label("source").into())?;
                // params.expect_empty()?;
                //
                // let _store = {
                //     let mut context = txn.context().write().await;
                //     let (_, dir) = context.create_dir_unique()?;
                //     Dir::load(*txn.id(), dir).await?
                // };

                Err(not_implemented!("copy a Table"))
            })
        }))
    }
}

struct CreateHandler;

impl<'a, State> Handler<'a, State> for CreateHandler
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    TableFile<State::Txn, State::FE>: Persist<State::FE, Schema = TableSchema, Txn = State::Txn>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, value| {
            Box::pin(async move {
                let schema = TableSchema::try_cast_from_value(value)?;

                let store = {
                    let mut context = txn.context().write().await;
                    let (_, dir) = context.create_dir_unique()?;
                    Dir::load(*txn.id(), dir, false).await?
                };

                TableFile::create(*txn.id(), schema, store)
                    .map_ok(Table::from)
                    .map_ok(Collection::Table)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

struct ContainsHandler<Txn, FE> {
    table: Table<Txn, FE>,
}

impl<'a, State> Handler<'a, State> for ContainsHandler<State::Txn, State::FE>
where
    State: StateInstance,
    State::FE: AsType<Node> + ThreadSafe,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let filled = match KeyOrRange::try_from_value(&self.table, key)? {
                    KeyOrRange::All => {
                        let empty = self.table.is_empty(*txn.id()).await?;
                        !empty
                    }
                    KeyOrRange::Key(key) => {
                        let row = self.table.read(*txn.id(), key).await?;
                        row.is_some()
                    }
                    KeyOrRange::Range(bounds) => {
                        let slice = self.table.slice(bounds.into())?;
                        let empty = slice.is_empty(*txn.id()).await?;
                        !empty
                    }
                };

                Ok(Value::from(filled).into())
            })
        }))
    }
}

impl<Txn, FE> From<Table<Txn, FE>> for ContainsHandler<Txn, FE> {
    fn from(table: Table<Txn, FE>) -> Self {
        Self { table }
    }
}

struct CountHandler<T> {
    table: T,
}

impl<'a, State, T> Handler<'a, State> for CountHandler<T>
where
    State: StateInstance + From<u64>,
    T: TableRead + TableSlice + TableStream + fmt::Debug + 'a,
    T::Slice: TableStream,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let count = match KeyOrRange::try_from_value(&self.table, key)? {
                    KeyOrRange::All => self.table.count(*txn.id()).await?,
                    KeyOrRange::Key(key) => match self.table.read(*txn.id(), key).await? {
                        Some(_row) => 1,
                        None => 0,
                    },
                    KeyOrRange::Range(range) => {
                        let slice = self.table.slice(range)?;
                        slice.count(*txn.id()).await?
                    }
                };

                Ok(State::from(count))
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

impl<'a, State, T> Handler<'a, State> for LimitHandler<T>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    T: TableStream + 'a,
    Table<State::Txn, State::FE>: From<T::Limit>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let limit = key.try_cast_into(|v| {
                    bad_request!("limit must be a positive integer, not {}", v)
                })?;

                let table = self.table.limit(limit)?;
                Ok(State::from(Collection::Table(table.into())))
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

impl<'a, State> Handler<'a, State> for LoadHandler
where
    State: StateInstance,
{
    // TODO
}

struct OrderHandler<T> {
    table: T,
}

impl<'a, State, T> Handler<'a, State> for OrderHandler<T>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    T: TableOrder + 'a,
    Table<State::Txn, State::FE>: From<T::OrderBy>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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
                    return Err(bad_request!("invalid column list to order by: {}", key));
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

struct TableHandler<Txn, FE> {
    table: Table<Txn, FE>,
}

impl<Txn, FE> TableHandler<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe + Clone,
{
    async fn create_tmp(
        &self,
        txn: &Txn,
    ) -> TCResult<b_tree::BTreeLock<BTreeSchema, ValueCollator, FE>> {
        let (_, tmp) = {
            let mut context = txn.context().write().await;
            context.create_dir_unique()?
        };

        b_tree::BTreeLock::create(
            self.table.schema().primary().clone(),
            ValueCollator::default(),
            tmp,
        )
        .map_err(TCError::from)
    }

    async fn truncate(self, txn: &Txn, range: Range) -> TCResult<()> {
        let tmp = self.create_tmp(txn).await?;
        self.table.truncate(*txn.id(), range, tmp).await
    }

    async fn update(self, txn: &Txn, range: Range, values: Map<Value>) -> TCResult<()> {
        let tmp = self.create_tmp(txn).await?;
        self.table.update(*txn.id(), range, values, tmp).await
    }
}

impl<'a, State> Handler<'a, State> for TableHandler<State::Txn, State::FE>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Map<State>>,
    State::FE: AsType<Node> + ThreadSafe,
    Map<Value>: TryFrom<State, Error = TCError>,
    Scalar: TryCastFrom<State>,
    Tuple<State>: TryFrom<State, Error = TCError>,
    Value: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                match KeyOrRange::try_from_value(&self.table, key)? {
                    KeyOrRange::All => Ok(State::from(Collection::Table(self.table.into()))),
                    KeyOrRange::Range(range) => {
                        let slice = self.table.slice(range)?;
                        Ok(State::from(Collection::Table(slice.into())))
                    }
                    KeyOrRange::Key(key) => {
                        self.table
                            .read(*txn.id(), key)
                            .map_ok(Value::from)
                            .map_ok(State::from)
                            .await
                    }
                }
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("Table PUT {:?} <- {:?}", key, value);

                match KeyOrRange::try_from_value(&self.table, key)? {
                    KeyOrRange::All => {
                        let values = Map::<Value>::try_from(value)?;
                        self.update(txn, Range::default(), values).await
                    }
                    KeyOrRange::Range(range) => {
                        let values = Map::<Value>::try_from(value)?;
                        self.update(txn, range, values).await
                    }
                    KeyOrRange::Key(key) => {
                        let values = if value.is_tuple() {
                            Tuple::<State>::try_from(value)?
                        } else {
                            Tuple::<State>::from(vec![value])
                        };

                        let values = values
                            .into_iter()
                            .map(|state| {
                                state.try_cast_into(|s| TCError::unexpected(s, "a column value"))
                            })
                            .collect::<TCResult<Values>>()?;

                        self.table.upsert(*txn.id(), key, values).await
                    }
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let range = State::from(params)
                    .try_cast_into(|s| bad_request!("invalid table selector: {s:?}"))?;

                let range = cast_into_range(&self.table, range)?;

                self.table
                    .slice(range)
                    .map(|slice| Table::from(slice))
                    .map(Collection::Table)
                    .map(State::from)
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, State::Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let txn_id = *txn.id();

                match KeyOrRange::try_from_value(&self.table, key)? {
                    KeyOrRange::All => self.truncate(txn, Range::default()).await,
                    KeyOrRange::Key(key) => self.table.delete(txn_id, key).await,
                    KeyOrRange::Range(range) => self.truncate(txn, range).await,
                }
            })
        }))
    }
}

impl<Txn, FE> From<Table<Txn, FE>> for TableHandler<Txn, FE> {
    fn from(table: Table<Txn, FE>) -> Self {
        Self { table }
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

impl<'a, State, T> Handler<'a, State> for SchemaHandler<'a, T>
where
    State: StateInstance,
    T: TableInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<'a, State, T> Handler<'a, State> for SelectHandler<T>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    T: TableStream + 'a,
    Table<State::Txn, State::FE>: From<T::Selection>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let columns =
                    key.try_cast_into(|v| TCError::unexpected(v, "a Tuple of column names"))?;

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

impl<State> Route<State> for Table<State::Txn, State::FE>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<u64>,
    State::FE: AsType<Node> + ThreadSafe,
    Map<Value>: TryFrom<State, Error = TCError>,
    Scalar: TryCastFrom<State>,
    Tuple<State>: TryFrom<State, Error = TCError>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(self, path)
    }
}

impl<State> Route<State> for TableFile<State::Txn, State::FE>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<u64>,
    State::FE: AsType<Node> + ThreadSafe,
    Map<Value>: TryFrom<State, Error = TCError>,
    Scalar: TryCastFrom<State>,
    Tuple<State>: TryFrom<State, Error = TCError>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(self, path)
    }
}

#[inline]
fn route<'a, State, T>(
    table: &'a T,
    path: &[PathSegment],
) -> Option<Box<dyn Handler<'a, State> + 'a>>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<u64>,
    State::FE: AsType<Node> + ThreadSafe,
    T: TableRead + TableOrder + TableSlice + TableStream + TableWrite + Clone + fmt::Debug,
    <T as TableSlice>::Slice: TableStream,
    Map<Value>: TryFrom<State, Error = TCError>,
    Scalar: TryCastFrom<State>,
    Table<State::Txn, State::FE>: From<T>
        + From<<T as TableStream>::Limit>
        + From<<T as TableOrder>::OrderBy>
        + From<<T as TableStream>::Selection>
        + From<<T as TableSlice>::Slice>,
    Tuple<State>: TryFrom<State, Error = TCError>,
    Value: TryCastFrom<State>,
{
    if path.is_empty() {
        Some(Box::new(TableHandler::from(Table::from(table.clone()))))
    } else if path.len() == 1 {
        match path[0].as_str() {
            "columns" => Some(Box::new(SchemaHandler::new(table, column_schema))),
            "contains" => Some(Box::new(ContainsHandler::from(Table::from(table.clone())))),
            "count" => Some(Box::new(CountHandler::from(table.clone()))),
            "key_columns" => Some(Box::new(SchemaHandler::new(table, key_columns))),
            "key_names" => Some(Box::new(SchemaHandler::new(table, key_names))),
            "limit" => Some(Box::new(LimitHandler::from(table.clone()))),
            "order" => Some(Box::new(OrderHandler::from(table.clone()))),
            "select" => Some(Box::new(SelectHandler::from(table.clone()))),
            _ => None,
        }
    } else {
        None
    }
}

pub struct Static;

impl<State> Route<State> for Static
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    TableFile<State::Txn, State::FE>: Persist<State::FE, Schema = TableSchema, Txn = State::Txn>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateHandler))
        } else if path == &["copy_from"] {
            Some(Box::new(CopyHandler))
        } else {
            None
        }
    }
}

enum KeyOrRange {
    All,
    Key(Key),
    Range(Range),
}

impl KeyOrRange {
    #[inline]
    fn try_from_value<T>(table: &T, value: Value) -> TCResult<Self>
    where
        T: TableInstance + fmt::Debug,
    {
        match value {
            Value::None => Ok(Self::All),
            Value::Tuple(tuple) if tuple.is_empty() => Ok(Self::All),
            Value::Tuple(tuple)
                if tuple.iter().all(|value| match value {
                    Value::Tuple(tuple) if tuple.len() == 2 => {
                        let columns = table.schema().primary().columns();
                        match &tuple[0] {
                            Value::Id(name) => columns.contains(name),
                            Value::String(name) => columns.iter().any(|column| name == column),
                            _ => false,
                        }
                    }
                    _ => false,
                }) =>
            {
                cast_into_range(table, Scalar::Value(Value::Tuple(tuple))).map(Self::Range)
            }
            Value::Tuple(key) if key.len() == table.schema().key().len() => {
                Ok(Self::Key(key.into_inner()))
            }
            value => Err(bad_request!("invalid table selector: {value:?}")),
        }
    }
}

#[inline]
fn cast_into_range<T: TableInstance + fmt::Debug>(table: &T, scalar: Scalar) -> TCResult<Range> {
    if scalar.is_none() {
        return Ok(Range::default());
    }

    let column_ranges = Map::<Value>::try_from(scalar)
        .map_err(|cause| bad_request!("invalid selection bounds for {table:?}").consume(cause))?;

    let columns = table.schema().primary().columns();

    column_ranges
        .into_iter()
        .map(|(col_name, value)| {
            if !columns.contains(&col_name) {
                Err(TCError::not_found(&col_name))
            } else if value.matches::<(Bound<Value>, Bound<Value>)>() {
                Ok((col_name, ColumnRange::In(value.opt_cast_into().unwrap())))
            } else {
                Ok((col_name, ColumnRange::Eq(value)))
            }
        })
        .collect::<TCResult<HashMap<Id, ColumnRange>>>()
        .map(Range::from)
}

#[inline]
fn column_schema<T: TableInstance>(table: &T) -> Value {
    let columns = table
        .schema()
        .primary()
        .columns()
        .into_iter()
        .cloned()
        .map(Value::from)
        .collect();

    Value::Tuple(columns)
}

#[inline]
fn key_columns<T: TableInstance>(table: &T) -> Value {
    let key = table
        .schema()
        .key()
        .iter()
        .cloned()
        .map(Value::Id)
        .collect();

    Value::Tuple(key)
}

#[inline]
fn key_names<T: TableInstance>(table: &T) -> Value {
    let key = table
        .schema()
        .key()
        .iter()
        .cloned()
        .map(Value::Id)
        .collect();

    Value::Tuple(key)
}
