use std::collections::{BTreeMap, HashMap, HashSet};
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, join_all, try_join_all, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};

use crate::block::dir::Dir;
use crate::class::{TCBoxTryFuture, TCResult, TCStream};
use crate::collection::btree::{self, BTree};
use crate::collection::schema::{Column, IndexSchema, Row, TableSchema};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{Value, ValueId};

use super::bounds::{self, Bounds, ColumnBound};
use super::view::{IndexSlice, MergeSource, Merged, TableSlice};
use super::{Selection, Table};

const PRIMARY_INDEX: &str = "primary";

#[derive(Clone)]
pub struct Index {
    btree: Arc<BTree>,
    schema: IndexSchema,
}

impl Index {
    pub async fn create(txn: Arc<Txn>, name: ValueId, schema: IndexSchema) -> TCResult<Index> {
        let btree = Arc::new(
            BTree::create(
                txn.id().clone(),
                schema.clone().into(),
                txn.context().create_btree(txn.id().clone(), name).await?,
            )
            .await?,
        );
        Ok(Index { btree, schema })
    }

    pub fn get(&self, txn_id: TxnId, key: Vec<Value>) -> TCBoxTryFuture<Option<Vec<Value>>> {
        Box::pin(async move {
            self.schema.validate_key(&key)?;

            let mut rows = self.btree.clone().slice(txn_id, key.into()).await?;
            if let Some(row) = rows.next().await {
                Ok(Some(row))
            } else {
                Ok(None)
            }
        })
    }

    pub fn len(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        self.btree.clone().len(txn_id, btree::Selector::all())
    }

    pub fn index_slice(&self, bounds: Bounds) -> TCResult<IndexSlice> {
        bounds::validate(&bounds, &self.schema().columns())?;
        IndexSlice::new(self.btree.clone(), self.schema().clone(), bounds)
    }

    fn insert<'a>(
        &'a self,
        txn_id: &'a TxnId,
        row: Row,
        reject_extra_columns: bool,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let key = self.schema().row_into_values(row, reject_extra_columns)?;
            self.btree.insert(txn_id, key).await
        })
    }

    pub fn schema(&'_ self) -> &'_ IndexSchema {
        &self.schema
    }

    pub fn validate_slice_bounds(&self, outer: Bounds, inner: Bounds) -> TCResult<()> {
        bounds::validate(&outer, &self.schema().columns())?;
        bounds::validate(&inner, &self.schema().columns())?;

        let outer = bounds::btree_range(&outer, &self.schema().columns())?;
        let inner = bounds::btree_range(&inner, &self.schema().columns())?;

        let dtypes = self.schema.data_types();
        if outer.contains(&inner, dtypes)? {
            Ok(())
        } else {
            Err(error::bad_request(
                "Slice does not contain requested bounds",
                "",
            ))
        }
    }
}

impl Selection for Index {
    type Stream = TCStream<Vec<Value>>;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        self.len(txn_id)
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move { self.btree.delete(&txn_id, btree::Selector::all()).await })
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let key = self.schema.row_into_values(row, false)?;
            self.btree.delete(txn_id, btree::Selector::Key(key)).await
        })
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.schema.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.schema.values()
    }

    fn order_by(&self, order: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        if self.schema.starts_with(&order) {
            if reverse {
                self.reversed()
            } else {
                Ok(self.clone().into())
            }
        } else {
            let order: Vec<String> = order.iter().map(String::from).collect();
            Err(error::bad_request(
                &format!("Index with schema {} does not support order", self.schema),
                order.join(", "),
            ))
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(IndexSlice::all(self.btree.clone(), self.schema.clone(), true).into())
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        self.index_slice(bounds).map(|is| is.into())
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            self.btree
                .clone()
                .slice(txn_id, btree::Selector::all())
                .await
        })
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        bounds::validate(bounds, &self.schema().columns())?;

        for (column, (bound_column, bound_range)) in self.schema.columns()[0..bounds.len()]
            .iter()
            .zip(bounds.iter())
        {
            if column.name() != bound_column {
                return Err(error::bad_request(
                    &format!(
                        "Expected column {} in index range selector but found",
                        column.name()
                    ),
                    bound_column,
                ));
            }

            bound_range.expect(*column.dtype(), &format!("for column {}", column.name()))?;
        }

        Ok(())
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        if !self.schema.starts_with(&order) {
            let order: Vec<String> = order.iter().map(|c| c.to_string()).collect();
            Err(error::bad_request(
                &format!("Cannot order index with schema {} by", self.schema),
                order.join(", "),
            ))
        } else {
            Ok(())
        }
    }

    fn update<'a>(self, txn: Arc<Txn>, row: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let key: btree::Key = self.schema().row_into_values(row, false)?;
            self.btree
                .update(txn.id(), &btree::Selector::all(), &key)
                .await
        })
    }
}

#[async_trait]
impl Transact for Index {
    async fn commit(&self, txn_id: &TxnId) {
        self.btree.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.btree.rollback(txn_id).await
    }
}

#[derive(Clone)]
pub struct ReadOnly {
    index: IndexSlice,
}

impl ReadOnly {
    pub fn copy_from<'a>(
        source: Table,
        txn: Arc<Txn>,
        key_columns: Option<Vec<ValueId>>,
    ) -> TCBoxTryFuture<'a, ReadOnly> {
        Box::pin(async move {
            let btree_file = txn
                .clone()
                .subcontext_tmp()
                .await?
                .context()
                .create_btree(txn.id().clone(), "index".parse()?)
                .await?;

            let source_schema: IndexSchema =
                (source.key().to_vec(), source.values().to_vec()).into();
            let (schema, btree) = if let Some(columns) = key_columns {
                let column_names: HashSet<&ValueId> = columns.iter().collect();
                let schema = source_schema.subset(column_names)?;
                let btree =
                    BTree::create(txn.id().clone(), schema.clone().into(), btree_file).await?;

                let rows = source.select(columns)?.stream(txn.id().clone()).await?;
                btree.insert_from(txn.id(), rows).await?;
                (schema, btree)
            } else {
                let btree =
                    BTree::create(txn.id().clone(), source_schema.clone().into(), btree_file)
                        .await?;
                let rows = source.stream(txn.id().clone()).await?;
                btree.insert_from(txn.id(), rows).await?;
                (source_schema, btree)
            };

            let index = Index {
                schema,
                btree: Arc::new(btree),
            };

            index
                .index_slice(bounds::all())
                .map(|index| ReadOnly { index })
        })
    }

    pub fn into_reversed(self) -> ReadOnly {
        ReadOnly {
            index: self.index.into_reversed(),
        }
    }
}

impl Selection for ReadOnly {
    type Stream = <Index as Selection>::Stream;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        Box::pin(async move { self.index.clone().count(txn_id).await })
    }

    fn order_by(&self, order: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        self.index.validate_order(&order)?;

        if reverse {
            self.reversed()
        } else {
            Ok(self.clone().into())
        }
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(self.clone().into_reversed().into())
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.index.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.index.values()
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        self.validate_bounds(&bounds)?;
        self.index
            .slice_index(bounds)
            .map(|index| ReadOnly { index }.into())
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move { self.index.clone().stream(txn_id).await })
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        self.index.validate_bounds(bounds)
    }

    fn validate_order(&self, order: &[ValueId]) -> TCResult<()> {
        self.index.validate_order(order)
    }
}

#[derive(Clone)]
pub struct TableBase {
    dir: Arc<Dir>,
    primary: Index,
    auxiliary: BTreeMap<ValueId, Index>,
}

impl TableBase {
    pub async fn create(txn: Arc<Txn>, schema: TableSchema) -> TCResult<TableBase> {
        let primary = Index::create(
            txn.clone(),
            PRIMARY_INDEX.parse()?,
            schema.primary().clone(),
        )
        .await?;

        let auxiliary: BTreeMap<ValueId, Index> =
            try_join_all(schema.indices().iter().map(|(name, column_names)| {
                Self::create_index(&txn, schema.primary(), name.clone(), column_names.to_vec())
                    .map_ok(move |index| (name.clone(), index))
            }))
            .await?
            .into_iter()
            .collect();

        Ok(TableBase {
            dir: txn.context(),
            primary,
            auxiliary,
        })
    }

    async fn create_index(
        txn: &Arc<Txn>,
        primary: &IndexSchema,
        name: ValueId,
        key: Vec<ValueId>,
    ) -> TCResult<Index> {
        if name.as_str() == PRIMARY_INDEX {
            return Err(error::bad_request(
                "This index name is reserved",
                PRIMARY_INDEX,
            ));
        }

        let index_key_set: HashSet<&ValueId> = key.iter().collect();
        if index_key_set.len() != key.len() {
            return Err(error::bad_request(
                &format!("Duplicate column in index {}", name),
                key.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        let mut columns: HashMap<ValueId, Column> = primary
            .columns()
            .iter()
            .cloned()
            .map(|c| (c.name().clone(), c))
            .collect();
        let key: Vec<Column> = key
            .iter()
            .map(|c| columns.remove(&c).ok_or_else(|| error::not_found(c)))
            .collect::<TCResult<Vec<Column>>>()?;

        let values: Vec<Column> = primary
            .key()
            .iter()
            .filter(|c| !index_key_set.contains(c.name()))
            .cloned()
            .collect();
        let schema: IndexSchema = (key, values).into();

        let btree_file = txn
            .context()
            .create_btree(txn.id().clone(), name.clone())
            .await?;
        let btree = btree::BTree::create(txn.id().clone(), schema.clone().into(), btree_file)
            .map_ok(Arc::new)
            .await?;

        Ok(Index { btree, schema })
    }

    pub fn primary(&'_ self) -> &'_ Index {
        &self.primary
    }

    pub fn supporting_index(&self, bounds: &Bounds) -> TCResult<Index> {
        if self.primary.validate_bounds(bounds).is_ok() {
            return Ok(self.primary.clone());
        }

        for index in self.auxiliary.values() {
            if index.validate_bounds(bounds).is_ok() {
                return Ok(index.clone());
            }
        }

        Err(error::bad_request(
            "This table has no index which supports bounds",
            super::bounds::format(bounds),
        ))
    }

    pub fn get<'a>(
        &'a self,
        txn_id: TxnId,
        key: Vec<Value>,
    ) -> TCBoxTryFuture<'a, Option<Vec<Value>>> {
        self.primary.get(txn_id, key)
    }

    pub fn get_owned<'a>(
        self,
        txn_id: TxnId,
        key: Vec<Value>,
    ) -> TCBoxTryFuture<'a, Option<Vec<Value>>> {
        Box::pin(async move { self.get(txn_id, key).await })
    }

    pub fn insert<'a>(
        &'a self,
        txn_id: TxnId,
        key: Vec<Value>,
        value: Vec<Value>,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            if self.get(txn_id.clone(), key.to_vec()).await?.is_some() {
                let key: Vec<String> = key.iter().map(|v| v.to_string()).collect();
                Err(error::bad_request(
                    "Tried to insert but this key already exists",
                    format!("[{}]", key.join(", ")),
                ))
            } else {
                let mut values = key;
                values.extend(value);
                let row = self.primary.schema().values_into_row(values)?;
                self.upsert(&txn_id, row).await
            }
        })
    }

    pub fn upsert<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            self.delete_row(txn_id, row.clone()).await?;

            let mut inserts = Vec::with_capacity(self.auxiliary.len() + 1);
            inserts.push(self.primary.insert(txn_id, row.clone(), true));
            for index in self.auxiliary.values() {
                inserts.push(index.insert(txn_id, row.clone(), false));
            }

            try_join_all(inserts).await?;
            Ok(())
        })
    }
}

impl Selection for TableBase {
    type Stream = <Index as Selection>::Stream;

    fn count(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        self.primary.count(txn_id)
    }

    fn delete<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let mut deletes = Vec::with_capacity(self.auxiliary.len() + 1);
            deletes.push(self.primary.delete(txn_id.clone()));
            for index in self.auxiliary.values() {
                deletes.push(index.clone().delete(txn_id.clone()));
            }

            try_join_all(deletes).await?;
            Ok(())
        })
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            self.primary.schema().validate_row(&row)?;

            let mut deletes = Vec::with_capacity(self.auxiliary.len() + 1);
            for index in self.auxiliary.values() {
                deletes.push(index.delete_row(txn_id, row.clone()));
            }
            deletes.push(self.primary.delete_row(txn_id, row));
            try_join_all(deletes).await?;

            Ok(())
        })
    }

    fn order_by(&self, columns: Vec<ValueId>, reverse: bool) -> TCResult<Table> {
        self.validate_order(&columns)?;

        if self.primary.validate_order(&columns).is_ok() {
            let ordered = TableSlice::new(self.clone(), bounds::all())?;
            if reverse {
                return ordered.reversed();
            } else {
                return Ok(ordered.into());
            }
        }

        let selection = TableSlice::new(self.clone(), bounds::all())?;
        let mut merge_source = MergeSource::Table(selection);

        let mut columns = &columns[..];
        loop {
            let initial = columns.to_vec();
            for i in (1..columns.len() + 1).rev() {
                let subset = &columns[..i];

                for index in iter::once(&self.primary).chain(self.auxiliary.values()) {
                    if index.validate_order(subset).is_ok() {
                        columns = &columns[i..];

                        let index_slice = self.primary.index_slice(bounds::all())?;
                        let merged = Merged::new(merge_source, index_slice);

                        if columns.is_empty() {
                            if reverse {
                                return merged.reversed();
                            } else {
                                return Ok(merged.into());
                            }
                        }

                        merge_source = MergeSource::Merge(Arc::new(merged));
                        break;
                    }
                }
            }

            if columns == &initial[..] {
                let order: Vec<String> = columns.iter().map(String::from).collect();
                return Err(error::bad_request(
                    "This table has no index to support the order",
                    order.join(", "),
                ));
            }
        }
    }

    fn key(&'_ self) -> &'_ [Column] {
        self.primary.key()
    }

    fn values(&'_ self) -> &'_ [Column] {
        self.primary.key()
    }

    fn reversed(&self) -> TCResult<Table> {
        Err(error::unsupported(
            "Cannot reverse a Table itself, consider reversing a slice of the table instead",
        ))
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Table> {
        if self.primary.validate_bounds(&bounds).is_ok() {
            return TableSlice::new(self.clone(), bounds).map(|t| t.into());
        }

        let mut columns: Vec<ValueId> = self
            .primary
            .schema()
            .columns()
            .iter()
            .map(|c| c.name())
            .cloned()
            .collect();
        let bounds: Vec<(ValueId, ColumnBound)> = columns
            .drain(..)
            .filter_map(|name| bounds.get(&name).map(|bound| (name, bound.clone())))
            .collect();

        let selection = TableSlice::new(self.clone(), bounds::all())?;
        let mut merge_source = MergeSource::Table(selection);

        let mut bounds = &bounds[..];
        loop {
            let initial = bounds.len();
            for i in (1..bounds.len() + 1).rev() {
                let subset: Bounds = bounds[..i].iter().cloned().collect();

                for index in iter::once(&self.primary).chain(self.auxiliary.values()) {
                    if index.validate_bounds(&subset).is_ok() {
                        bounds = &bounds[i..];

                        let index_slice = self.primary.index_slice(subset)?;
                        let merged = Merged::new(merge_source, index_slice);

                        if bounds.is_empty() {
                            return Ok(merged.into());
                        }

                        merge_source = MergeSource::Merge(Arc::new(merged));
                        break;
                    }
                }
            }

            if bounds.len() == initial {
                let order: Vec<String> = bounds.iter().map(|(name, _)| name.to_string()).collect();
                return Err(error::bad_request(
                    "This table has no index to support selection bounds on",
                    order.join(", "),
                ));
            }
        }
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        self.primary.stream(txn_id)
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let bounds: Vec<(ValueId, ColumnBound)> = self
            .primary
            .schema()
            .columns()
            .iter()
            .filter_map(|c| {
                bounds
                    .get(c.name())
                    .map(|bound| (c.name().clone(), bound.clone()))
            })
            .collect();

        let mut bounds = &bounds[..];
        while !bounds.is_empty() {
            let initial = bounds.len();
            for i in (1..bounds.len() + 1).rev() {
                let subset: Bounds = bounds[..i].iter().cloned().collect();

                for index in iter::once(&self.primary).chain(self.auxiliary.values()) {
                    if index.validate_bounds(&subset).is_ok() {
                        bounds = &bounds[i..];
                        break;
                    }
                }
            }

            if bounds.len() == initial {
                let order: Vec<String> = bounds.iter().map(|(name, _)| name.to_string()).collect();
                return Err(error::bad_request(
                    "This table has no index to support selection bounds on",
                    order.join(", "),
                ));
            }
        }

        Ok(())
    }

    fn validate_order(&self, mut order: &[ValueId]) -> TCResult<()> {
        while !order.is_empty() {
            let initial = order.to_vec();
            for i in (1..order.len() + 1).rev() {
                let subset = &order[..i];

                for index in iter::once(&self.primary).chain(self.auxiliary.values()) {
                    if index.validate_order(subset).is_ok() {
                        order = &order[i..];
                        break;
                    }
                }
            }

            if order == &initial[..] {
                let order: Vec<String> = order.iter().map(String::from).collect();
                return Err(error::bad_request(
                    "This table has no index to support the order",
                    order.join(", "),
                ));
            }
        }

        Ok(())
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema = self.primary.schema();
            schema.validate_row_partial(&value)?;

            let index = self.clone().index(txn.clone(), None).await?;

            let txn_id = txn.id();
            index
                .stream(txn_id.clone())
                .await?
                .map(|row| schema.values_into_row(row))
                .map_ok(|row| self.upsert(txn_id, row))
                .try_buffer_unordered(2)
                .try_fold((), |_, _| future::ready(Ok(())))
                .await
        })
    }
}

#[async_trait]
impl Transact for TableBase {
    async fn commit(&self, txn_id: &TxnId) {
        let mut commits = Vec::with_capacity(self.auxiliary.len() + 1);
        commits.push(self.primary.commit(txn_id));
        for index in self.auxiliary.values() {
            commits.push(index.commit(txn_id));
        }
        join_all(commits).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let mut rollbacks = Vec::with_capacity(self.auxiliary.len() + 1);
        rollbacks.push(self.primary.rollback(txn_id));
        for index in self.auxiliary.values() {
            rollbacks.push(index.commit(txn_id));
        }
        join_all(rollbacks).await;
    }
}
