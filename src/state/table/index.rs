use std::collections::{BTreeMap, HashMap, HashSet};
use std::iter;
use std::sync::Arc;

use futures::future::{self, try_join_all};
use futures::stream::{StreamExt, TryStreamExt};

use crate::error;
use crate::state::btree::{self, BTree};
use crate::state::dir::Dir;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Txn, TxnId};
use crate::value::{TCBoxTryFuture, TCResult, TCStream, Value, ValueId};

use super::schema::{Bounds, Column, ColumnBound, Row, Schema};
use super::view::{IndexSlice, MergeSource, Merged, TableSlice};
use super::{Selection, Table};

const PRIMARY_INDEX: &str = "primary";

#[derive(Clone)]
pub struct Index {
    btree: Arc<BTree>,
    schema: Schema,
}

impl Index {
    pub async fn create(txn: Arc<Txn>, name: ValueId, schema: Schema) -> TCResult<Index> {
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

    pub fn len(&self, txn_id: TxnId) -> TCBoxTryFuture<u64> {
        self.btree.clone().len(txn_id, btree::Selector::all())
    }

    pub fn index_slice(&self, bounds: Bounds) -> TCResult<IndexSlice> {
        self.schema.validate_bounds(&bounds)?;
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

    pub fn validate_schema_bounds(&self, outer: Bounds, inner: Bounds) -> TCResult<()> {
        self.schema.validate_bounds(&outer)?;
        self.schema.validate_bounds(&inner)?;

        let outer = outer.try_into_btree_range(self.schema())?;
        let inner = inner.try_into_btree_range(self.schema())?;
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

    fn order_by<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        order: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<Table> {
        let table = if self.schema.starts_with(&order) {
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
        };

        Box::pin(future::ready(table))
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(IndexSlice::all(self.btree.clone(), self.schema.clone(), true).into())
    }

    fn slice(&self, _txn_id: &TxnId, bounds: Bounds) -> TCBoxTryFuture<Table> {
        Box::pin(async move { self.index_slice(bounds).map(|is| is.into()) })
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move {
            self.btree
                .clone()
                .slice(txn_id, btree::Selector::all())
                .await
        })
    }

    fn validate_bounds<'a>(&'a self, _txn_id: &'a TxnId, bounds: &'a Bounds) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            self.schema.validate_bounds(bounds)?;

            for (column, (bound_column, bound_range)) in self.schema.columns()[0..bounds.len()]
                .iter()
                .zip(bounds.iter())
            {
                if &column.name != bound_column {
                    return Err(error::bad_request(
                        &format!(
                            "Expected column {} in index range selector but found",
                            column.name
                        ),
                        bound_column,
                    ));
                }

                bound_range.expect(column.dtype, &format!("for column {}", column.name))?;
            }

            Ok(())
        })
    }

    fn validate_order<'a>(
        &'a self,
        _txn_id: &'a TxnId,
        order: &'a [ValueId],
    ) -> TCBoxTryFuture<'a, ()> {
        let result = if !self.schema.starts_with(&order) {
            let order: Vec<String> = order.iter().map(|c| c.to_string()).collect();
            Err(error::bad_request(
                &format!("Cannot order index with schema {} by", self.schema),
                order.join(", "),
            ))
        } else {
            Ok(())
        };

        Box::pin(future::ready(result))
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

            let (schema, btree) = if let Some(columns) = key_columns {
                let column_names: HashSet<&ValueId> = columns.iter().collect();
                let schema = source.schema().subset(column_names)?;
                let btree =
                    BTree::create(txn.id().clone(), schema.clone().into(), btree_file).await?;

                let rows = source.select(columns)?.stream(txn.id().clone()).await?;
                btree.insert_from(txn.id(), rows).await?;
                (schema, btree)
            } else {
                let schema = source.schema().clone();
                let btree =
                    BTree::create(txn.id().clone(), schema.clone().into(), btree_file).await?;
                let rows = source.stream(txn.id().clone()).await?;
                btree.insert_from(txn.id(), rows).await?;
                (schema, btree)
            };

            let index = Index {
                schema,
                btree: Arc::new(btree),
            };

            index
                .index_slice(Bounds::all())
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

    fn order_by<'a>(
        &'a self,
        txn_id: &'a TxnId,
        order: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            self.index.validate_order(txn_id, &order).await?;

            if reverse {
                self.reversed()
            } else {
                Ok(self.clone().into())
            }
        })
    }

    fn reversed(&self) -> TCResult<Table> {
        Ok(self.clone().into_reversed().into())
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.index.schema()
    }

    fn slice<'a>(&'a self, txn_id: &'a TxnId, bounds: Bounds) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            self.validate_bounds(txn_id, &bounds).await?;
            self.index
                .slice_index(bounds)
                .map(|index| ReadOnly { index }.into())
        })
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        Box::pin(async move { self.index.clone().stream(txn_id).await })
    }

    fn validate_bounds<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Bounds,
    ) -> TCBoxTryFuture<'a, ()> {
        self.index.validate_bounds(txn_id, bounds)
    }

    fn validate_order<'a>(&'a self, txn_id: &'a TxnId, order: &'a [ValueId]) -> TCBoxTryFuture<()> {
        self.index.validate_order(txn_id, order)
    }
}

#[derive(Clone)]
pub struct TableBase {
    dir: Arc<Dir>,
    primary: Index,
    auxiliary: TxnLock<Mutable<BTreeMap<ValueId, Index>>>,
    schema: Schema,
}

impl TableBase {
    pub async fn create(txn: Arc<Txn>, schema: Schema) -> TCResult<TableBase> {
        let primary = Index::create(txn.clone(), PRIMARY_INDEX.parse()?, schema.clone()).await?;
        let auxiliary = TxnLock::new(txn.id().clone(), BTreeMap::new().into());

        Ok(TableBase {
            dir: txn.context(),
            primary,
            auxiliary,
            schema,
        })
    }

    pub fn primary(&'_ self) -> &'_ Index {
        &self.primary
    }

    pub async fn supporting_index<'a>(
        &'a self,
        txn_id: &'a TxnId,
        bounds: &'a Bounds,
    ) -> TCResult<Index> {
        if self.primary.validate_bounds(txn_id, bounds).await.is_ok() {
            return Ok(self.primary.clone());
        }

        for index in self.auxiliary.read(txn_id).await?.values() {
            if index.validate_bounds(txn_id, bounds).await.is_ok() {
                return Ok(index.clone());
            }
        }

        Err(error::bad_request(
            "This table has no index which supports bounds",
            bounds,
        ))
    }

    pub async fn add_index(&self, txn: Arc<Txn>, name: ValueId, key: Vec<ValueId>) -> TCResult<()> {
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

        let mut auxiliary = self.auxiliary.write(txn.id().clone()).await?;

        let columns: HashMap<ValueId, Column> = self.schema().clone().into();
        let key: Vec<Column> = key
            .iter()
            .map(|c| columns.get(&c).cloned().ok_or_else(|| error::not_found(c)))
            .collect::<TCResult<Vec<Column>>>()?;
        let values: Vec<Column> = self
            .schema()
            .key_columns()
            .iter()
            .filter(|c| !index_key_set.contains(&c.name))
            .cloned()
            .collect();
        let schema: Schema = (key, values).into();

        let btree_file = self
            .dir
            .create_btree(txn.id().clone(), name.clone())
            .await?;
        let btree = Arc::new(
            btree::BTree::create(txn.id().clone(), schema.clone().into(), btree_file).await?,
        );
        btree
            .insert_from(
                txn.id(),
                self.clone()
                    .select(schema.clone().into())?
                    .stream(txn.id().clone())
                    .await?,
            )
            .await?;

        let index = Index { btree, schema };
        if auxiliary.contains_key(&name) {
            self.dir.delete_file(txn.id().clone(), &name).await?;
            Err(error::bad_request(
                "This table already has an index named",
                name,
            ))
        } else {
            auxiliary.insert(name, index);
            Ok(())
        }
    }

    pub async fn remove_index(&self, txn_id: TxnId, name: &ValueId) -> TCResult<()> {
        let mut auxiliary = self.auxiliary.write(txn_id.clone()).await?;
        match auxiliary.remove(name) {
            Some(_index) => self.dir.clone().delete_file(txn_id, name).await,
            None => Err(error::not_found(name)),
        }
    }

    pub fn upsert<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            self.delete_row(txn_id, row.clone()).await?;

            let auxiliary = self.auxiliary.read(txn_id).await?;

            let mut inserts = Vec::with_capacity(auxiliary.len() + 1);
            for index in auxiliary.values() {
                inserts.push(index.insert(txn_id, row.clone(), false));
            }
            inserts.push(self.primary.insert(txn_id, row, true));

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
            let auxiliary = self.auxiliary.read(&txn_id).await?;
            let mut deletes = Vec::with_capacity(auxiliary.len() + 1);
            for index in auxiliary.values() {
                deletes.push(index.clone().delete(txn_id.clone()));
            }
            deletes.push(self.primary.delete(txn_id));

            try_join_all(deletes).await?;
            Ok(())
        })
    }

    fn delete_row<'a>(&'a self, txn_id: &'a TxnId, row: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            self.schema().validate_row(&row)?;

            let auxiliary = self.auxiliary.read(txn_id).await?;
            let mut deletes = Vec::with_capacity(auxiliary.len() + 1);
            for index in auxiliary.values() {
                deletes.push(index.delete_row(txn_id, row.clone()));
            }
            deletes.push(self.primary.delete_row(txn_id, row));
            try_join_all(deletes).await?;

            Ok(())
        })
    }

    fn order_by<'a>(
        &'a self,
        txn_id: &'a TxnId,
        columns: Vec<ValueId>,
        reverse: bool,
    ) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            self.validate_order(txn_id, &columns).await?;

            if self.primary.validate_order(txn_id, &columns).await.is_ok() {
                let ordered = TableSlice::new(self.clone(), txn_id, Bounds::all()).await?;
                if reverse {
                    return ordered.reversed();
                } else {
                    return Ok(ordered.into());
                }
            }

            let auxiliary = self.auxiliary.read(txn_id).await?;

            let selection = TableSlice::new(self.clone(), txn_id, Bounds::all()).await?;
            let mut merge_source = MergeSource::Table(selection);

            let mut columns = &columns[..];
            loop {
                let initial = columns.to_vec();
                for i in (1..columns.len() + 1).rev() {
                    let subset = &columns[..i];

                    for index in iter::once(&self.primary).chain(auxiliary.values()) {
                        if index.validate_order(txn_id, subset).await.is_ok() {
                            columns = &columns[i..];

                            let index_slice = self.primary.index_slice(Bounds::all())?;
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
        })
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    fn reversed(&self) -> TCResult<Table> {
        Err(error::unsupported(
            "Cannot reverse a Table itself, consider reversing a slice of the table instead",
        ))
    }

    fn slice<'a>(&'a self, txn_id: &'a TxnId, bounds: Bounds) -> TCBoxTryFuture<'a, Table> {
        Box::pin(async move {
            if self.primary.validate_bounds(txn_id, &bounds).await.is_ok() {
                return TableSlice::new(self.clone(), txn_id, bounds)
                    .await
                    .map(|t| t.into());
            }

            let mut columns: Vec<ValueId> = self.schema().column_names();
            let bounds: Vec<(ValueId, ColumnBound)> = columns
                .drain(..)
                .filter_map(|name| bounds.get(&name).map(|bound| (name, bound.clone())))
                .collect();

            let auxiliary = self.auxiliary.read(txn_id).await?;

            let selection = TableSlice::new(self.clone(), txn_id, Bounds::all()).await?;
            let mut merge_source = MergeSource::Table(selection);

            let mut bounds = &bounds[..];
            loop {
                let initial = bounds.len();
                for i in (1..bounds.len() + 1).rev() {
                    let subset: Bounds = bounds[..i].iter().cloned().collect();

                    for index in iter::once(&self.primary).chain(auxiliary.values()) {
                        if index.validate_bounds(txn_id, &subset).await.is_ok() {
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
                    let order: Vec<String> =
                        bounds.iter().map(|(name, _)| name.to_string()).collect();
                    return Err(error::bad_request(
                        "This table has no index to support selection bounds on",
                        order.join(", "),
                    ));
                }
            }
        })
    }

    fn stream<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, Self::Stream> {
        self.primary.stream(txn_id)
    }

    fn validate_bounds<'a>(&'a self, txn_id: &'a TxnId, bounds: &'a Bounds) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            let mut columns: Vec<ValueId> = self.schema().column_names();
            let bounds: Vec<(ValueId, ColumnBound)> = columns
                .drain(..)
                .filter_map(|name| bounds.get(&name).map(|bound| (name, bound.clone())))
                .collect();

            let auxiliary = self.auxiliary.read(txn_id).await?;

            let mut bounds = &bounds[..];
            while !bounds.is_empty() {
                let initial = bounds.len();
                for i in (1..bounds.len() + 1).rev() {
                    let subset: Bounds = bounds[..i].iter().cloned().collect();

                    for index in iter::once(&self.primary).chain(auxiliary.values()) {
                        if index.validate_bounds(txn_id, &subset).await.is_ok() {
                            bounds = &bounds[i..];
                            break;
                        }
                    }
                }

                if bounds.len() == initial {
                    let order: Vec<String> =
                        bounds.iter().map(|(name, _)| name.to_string()).collect();
                    return Err(error::bad_request(
                        "This table has no index to support selection bounds on",
                        order.join(", "),
                    ));
                }
            }

            Ok(())
        })
    }

    fn validate_order<'a>(
        &'a self,
        txn_id: &'a TxnId,
        mut order: &'a [ValueId],
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let auxiliary = self.auxiliary.read(txn_id).await?;

            while !order.is_empty() {
                let initial = order.to_vec();
                for i in (1..order.len() + 1).rev() {
                    let subset = &order[..i];

                    for index in iter::once(&self.primary).chain(auxiliary.values()) {
                        if index.validate_order(txn_id, subset).await.is_ok() {
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
        })
    }

    fn update<'a>(self, txn: Arc<Txn>, value: Row) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            let schema = self.schema().clone();
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
