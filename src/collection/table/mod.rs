use std::convert::TryFrom;
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use futures::future;
use futures::{Stream, StreamExt};
use log::debug;

use crate::class::{Class, Instance, NativeClass, State, TCResult, TCStream};
use crate::collection::class::CollectionInstance;
use crate::collection::{Collection, CollectionBase, CollectionView};
use crate::error;
use crate::request::Request;
use crate::scalar::{
    Id, Link, Object, PathSegment, Scalar, ScalarInstance, TCPathBuf, TryCastFrom, TryCastInto,
    Value,
};
use crate::transaction::{Transact, Txn, TxnId};

use super::schema::{Column, Row, TableSchema};

mod bounds;
mod index;
mod view;

const ERR_DELETE: &str = "Deletion is not supported by instance of";
const ERR_INSERT: &str = "Insertion is not supported by instance of";
const ERR_SLICE: &str = "Slicing is not supported by instance of";
const ERR_UPDATE: &str = "Update is not supported by instance of";

pub use bounds::*;

pub type TableBase = index::TableBase;
pub type TableBaseType = index::TableBaseType;
pub type TableIndex = index::TableIndex;
pub type TableView = view::TableView;
pub type TableViewType = view::TableViewType;

#[derive(Clone, Eq, PartialEq)]
pub enum TableType {
    Base(TableBaseType),
    View(TableViewType),
}

impl Class for TableType {
    type Instance = Table;
}

impl NativeClass for TableType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        TableBaseType::from_path(path).map(TableType::Base)
    }

    fn prefix() -> TCPathBuf {
        TableBaseType::prefix()
    }
}

impl From<TableBaseType> for TableType {
    fn from(tbt: TableBaseType) -> TableType {
        TableType::Base(tbt)
    }
}

impl From<TableViewType> for TableType {
    fn from(tvt: TableViewType) -> TableType {
        TableType::View(tvt)
    }
}

impl From<TableType> for Link {
    fn from(tt: TableType) -> Link {
        use TableType::*;
        match tt {
            Base(base) => base.into(),
            View(view) => view.into(),
        }
    }
}

impl fmt::Display for TableType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Base(base) => write!(f, "{}", base),
            Self::View(view) => write!(f, "{}", view),
        }
    }
}

#[async_trait]
pub trait TableInstance: Instance + Clone + Into<Table> + Sized + Send + 'static {
    type Stream: Stream<Item = Vec<Value>> + Send + Unpin;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        let count = self
            .clone()
            .stream(txn_id)
            .await?
            .fold(0, |count, _| future::ready(count + 1))
            .await;

        Ok(count)
    }

    async fn delete(self, _txn_id: TxnId) -> TCResult<()> {
        Err(error::bad_request(ERR_DELETE, self.class()))
    }

    async fn delete_row(&self, _txn_id: &TxnId, _row: Row) -> TCResult<()> {
        Err(error::bad_request(ERR_DELETE, self.class()))
    }

    fn group_by(&self, columns: Vec<Id>) -> TCResult<view::Aggregate> {
        view::Aggregate::new(self.clone().into(), columns)
    }

    async fn index(&self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly> {
        index::ReadOnly::copy_from(self.clone().into(), txn, columns).await
    }

    async fn insert(&self, _txn_id: TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(error::bad_request(ERR_INSERT, self.class()))
    }

    fn key(&'_ self) -> &'_ [Column];

    fn values(&'_ self) -> &'_ [Column];

    fn limit(&self, limit: u64) -> view::Limited {
        view::Limited::new(self.clone().into(), limit)
    }

    fn order_by(&self, columns: Vec<Id>, reverse: bool) -> TCResult<Table>;

    fn reversed(&self) -> TCResult<Table>;

    fn select(&self, columns: Vec<Id>) -> TCResult<view::Selection> {
        let selection = view::Selection::new(self.clone().into(), columns)?;
        Ok(selection)
    }

    fn slice(&self, _bounds: Bounds) -> TCResult<Table> {
        Err(error::bad_request(ERR_SLICE, self.class()))
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream>;

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()>;

    fn validate_order(&self, order: &[Id]) -> TCResult<()>;

    async fn update(self, _txn: Txn, _value: Row) -> TCResult<()> {
        Err(error::bad_request(ERR_UPDATE, self.class()))
    }

    async fn update_row(&self, _txn_id: TxnId, _row: Row, _value: Row) -> TCResult<()> {
        Err(error::bad_request(ERR_UPDATE, self.class()))
    }

    async fn upsert(&self, _txn_id: &TxnId, _key: Vec<Value>, _value: Vec<Value>) -> TCResult<()> {
        Err(error::bad_request(ERR_INSERT, self.class()))
    }
}

#[derive(Clone)]
pub struct TableImpl<T: TableInstance> {
    inner: T,
}

impl<T: TableInstance> TableImpl<T> {
    fn into_inner(self) -> T {
        self.inner
    }
}

#[async_trait]
impl<T: TableInstance + Sync> CollectionInstance for TableImpl<T> {
    type Item = Vec<Value>;
    type Slice = TableView;

    async fn get(
        &self,
        _request: &Request,
        _txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State> {
        if path.is_empty() {
            let table: Table = self.inner.clone().into();
            Ok(State::from(table))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "limit" => {
                    let limit =
                        selector.try_cast_into(|v| error::bad_request("Invalid limit", v))?;
                    Ok(State::from(Table::from(self.limit(limit))))
                }
                "select" => {
                    let columns = if selector.matches::<Vec<Id>>() {
                        selector.opt_cast_into().unwrap()
                    } else {
                        let name = selector
                            .try_cast_into(|v| error::bad_request("Invalid column name", v))?;

                        vec![name]
                    };

                    self.select(columns).map(Table::from).map(State::from)
                }
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn is_empty(&self, _txn: &Txn) -> TCResult<bool> {
        Err(error::not_implemented("TableImpl::is_empty"))
    }

    async fn post(
        &self,
        _request: &Request,
        _txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        if path.is_empty() {
            Err(error::method_not_allowed("Table: POST /"))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "where" => {
                    let bounds = Bounds::try_cast_from(params, |v| {
                        error::bad_request("Cannot cast into Table Bounds from", v)
                    })?;

                    self.slice(bounds).map(State::from)
                }
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(path))
        }
    }

    async fn put(
        &self,
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        debug!("{}::put", self.class());

        let key = match selector {
            Value::Tuple(key) => key,
            other => vec![other],
        };

        let value = Value::try_from(value)?;
        let value = match value {
            Value::Tuple(value) => value,
            other => vec![other],
        };

        self.upsert(txn.id(), key, value).await
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        let rows = self.inner.clone().stream(txn.id().clone()).await?;
        let rows = Box::pin(rows.map(Value::Tuple).map(Scalar::Value));
        Ok(rows)
    }
}

impl<T: TableInstance> Deref for TableImpl<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.inner
    }
}

impl<T: TableInstance> From<T> for TableImpl<T> {
    fn from(inner: T) -> TableImpl<T> {
        Self { inner }
    }
}

#[derive(Clone)]
pub enum Table {
    Base(TableImpl<TableBase>),
    View(TableImpl<TableView>),
}

impl Table {
    pub async fn create(txn: &Txn, schema: TableSchema) -> TCResult<TableIndex> {
        index::TableIndex::create(txn, schema).await
    }
}

impl Instance for Table {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Base(base) => base.class().into(),
            Self::View(view) => view.class().into(),
        }
    }
}

#[async_trait]
impl CollectionInstance for Table {
    type Item = Vec<Value>;
    type Slice = TableView;

    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State> {
        match self {
            Self::Base(base) => base.get(request, txn, path, selector).await,
            Self::View(view) => view.get(request, txn, path, selector).await,
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::Base(base) => base.is_empty(txn).await,
            Self::View(view) => view.is_empty(txn).await,
        }
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        match self {
            Self::Base(base) => base.post(request, txn, path, params).await,
            Self::View(view) => view.post(request, txn, path, params).await,
        }
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()> {
        match self {
            Self::Base(base) => base.put(request, txn, path, selector, value).await,
            Self::View(view) => view.put(request, txn, path, selector, value).await,
        }
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        match self {
            Self::Base(base) => base.to_stream(txn).await,
            Self::View(view) => view.to_stream(txn).await,
        }
    }
}

#[async_trait]
impl TableInstance for Table {
    type Stream = TCStream<Vec<Value>>;

    async fn count(&self, txn_id: TxnId) -> TCResult<u64> {
        match self {
            Self::Base(base) => base.count(txn_id).await,
            Self::View(view) => view.count(txn_id).await,
        }
    }

    async fn delete(self, txn_id: TxnId) -> TCResult<()> {
        match self {
            Self::Base(base) => base.into_inner().delete(txn_id).await,
            Self::View(view) => view.into_inner().delete(txn_id).await,
        }
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        match self {
            Self::Base(base) => base.delete_row(txn_id, row).await,
            Self::View(view) => view.delete_row(txn_id, row).await,
        }
    }

    fn group_by(&self, columns: Vec<Id>) -> TCResult<view::Aggregate> {
        match self {
            Self::Base(base) => base.group_by(columns),
            Self::View(view) => view.group_by(columns),
        }
    }

    async fn index(&self, txn: Txn, columns: Option<Vec<Id>>) -> TCResult<index::ReadOnly> {
        match self {
            Self::Base(base) => base.index(txn, columns).await,
            Self::View(view) => view.index(txn, columns).await,
        }
    }

    async fn insert(&self, txn_id: TxnId, key: Vec<Value>, value: Vec<Value>) -> TCResult<()> {
        match self {
            Self::Base(base) => base.insert(txn_id, key, value).await,
            Self::View(view) => view.insert(txn_id, key, value).await,
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        match self {
            Self::Base(base) => base.key(),
            Self::View(view) => view.key(),
        }
    }

    fn values(&'_ self) -> &'_ [Column] {
        match self {
            Self::Base(base) => base.values(),
            Self::View(view) => view.values(),
        }
    }

    fn limit(&self, limit: u64) -> view::Limited {
        match self {
            Self::Base(base) => base.limit(limit),
            Self::View(view) => view.limit(limit),
        }
    }

    fn order_by(&self, columns: Vec<Id>, reverse: bool) -> TCResult<Table> {
        match self {
            Self::Base(base) => base.order_by(columns, reverse),
            Self::View(view) => view.order_by(columns, reverse),
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        match self {
            Self::Base(base) => base.reversed(),
            Self::View(view) => view.reversed(),
        }
    }

    fn select(&self, columns: Vec<Id>) -> TCResult<view::Selection> {
        match self {
            Self::Base(base) => base.select(columns),
            Self::View(view) => view.select(columns),
        }
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        match self {
            Self::Base(base) => base.slice(bounds),
            Self::View(view) => view.slice(bounds),
        }
    }

    async fn stream(self, txn_id: TxnId) -> TCResult<Self::Stream> {
        match self {
            Self::Base(base) => base.into_inner().stream(txn_id).await,
            Self::View(view) => view.into_inner().stream(txn_id).await,
        }
    }

    async fn upsert(&self, txn_id: &TxnId, key: Vec<Value>, value: Vec<Value>) -> TCResult<()> {
        debug!("{}::upsert", self.class());

        match self {
            Self::Base(base) => base.upsert(txn_id, key, value).await,
            Self::View(view) => view.upsert(txn_id, key, value).await,
        }
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        match self {
            Self::Base(base) => base.validate_bounds(bounds),
            Self::View(view) => view.validate_bounds(bounds),
        }
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        match self {
            Self::Base(base) => base.validate_order(order),
            Self::View(view) => view.validate_order(order),
        }
    }

    async fn update(self, txn: Txn, value: Row) -> TCResult<()> {
        match self {
            Self::Base(base) => base.into_inner().update(txn, value).await,
            Self::View(view) => view.into_inner().update(txn, value).await,
        }
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, value: Row) -> TCResult<()> {
        match self {
            Self::Base(base) => base.update_row(txn_id, row, value).await,
            Self::View(view) => view.update_row(txn_id, row, value).await,
        }
    }
}

#[async_trait]
impl Transact for Table {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.commit(txn_id).await,
            Self::View(view) => view.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.rollback(txn_id).await,
            Self::View(view) => view.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.finalize(txn_id).await,
            Self::View(view) => view.finalize(txn_id).await,
        }
    }
}

impl From<TableBase> for Table {
    fn from(base: TableBase) -> Table {
        Table::Base(base.into())
    }
}

impl From<TableView> for Table {
    fn from(view: TableView) -> Table {
        Table::View(view.into())
    }
}

impl From<Table> for Collection {
    fn from(table: Table) -> Collection {
        match table {
            Table::Base(base) => Collection::Base(CollectionBase::Table(base)),
            view => Collection::View(CollectionView::Table(view)),
        }
    }
}

impl From<Table> for State {
    fn from(table: Table) -> State {
        State::Collection(table.into())
    }
}
