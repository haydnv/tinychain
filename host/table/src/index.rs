use std::collections::{BTreeMap, HashMap};
use std::convert::TryFrom;
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, join_all, try_join_all, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt, TryStreamExt};
use log::debug;

use tc_btree::{BTreeFile, BTreeInstance, BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, File, Persist, Restore};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, Id, Instance, Label, TCTryStream, Tuple};

use super::view::{MergeSource, Merged, TableSlice};
use super::{
    Bounds, Column, ColumnBound, IndexSchema, IndexSlice, Row, Table, TableInstance, TableSchema,
    TableType,
};

const PRIMARY_INDEX: Label = label("primary");

#[derive(Clone)]
pub struct Index<F, D, Txn> {
    btree: BTreeFile<F, D, Txn>,
    schema: IndexSchema,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Index<F, D, Txn> {
    pub async fn create(file: F, schema: IndexSchema, txn_id: TxnId) -> TCResult<Self> {
        BTreeFile::create(file, schema.clone().into(), txn_id)
            .map_ok(|btree| Index { btree, schema })
            .await
    }

    pub fn btree(&'_ self) -> &'_ BTreeFile<F, D, Txn> {
        &self.btree
    }

    pub async fn get(&self, txn_id: TxnId, key: Vec<Value>) -> TCResult<Option<Vec<Value>>> {
        let key = self.schema.validate_key(key)?;
        let range = tc_btree::Range::with_prefix(key);
        let mut rows = self.btree.clone().slice(range, false)?.keys(txn_id).await?;
        rows.try_next().await
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.btree.is_empty(*txn.id()).await
    }

    pub fn index_slice(self, bounds: Bounds) -> TCResult<IndexSlice<F, D, Txn>> {
        debug!("Index::index_slice");
        let bounds = bounds.validate(&self.schema.columns())?;
        IndexSlice::new(self.btree, self.schema, bounds)
    }

    async fn insert(&self, txn_id: TxnId, row: Row, reject_extra_columns: bool) -> TCResult<()> {
        let key = self.schema().values_from_row(row, reject_extra_columns)?;
        self.btree.insert(txn_id, key).await
    }

    pub fn schema(&'_ self) -> &'_ IndexSchema {
        &self.schema
    }

    pub fn validate_slice_bounds(&self, outer: Bounds, inner: Bounds) -> TCResult<()> {
        let columns = &self.schema.columns();
        let outer = outer.validate(columns)?.into_btree_range(columns)?;
        let inner = inner.validate(columns)?.into_btree_range(columns)?;

        if outer.contains(&inner, self.btree.collator()) {
            Ok(())
        } else {
            Err(TCError::unsupported(
                "slice does not contain requested bounds",
            ))
        }
    }

    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.validate_bounds(&bounds)?;
        let range = bounds.into_btree_range(&self.schema.columns())?;
        self.btree.slice(range, reverse)?.keys(txn_id).await
    }
}

impl<F, D, Txn> Instance for Index<F, D, Txn>
where
    Self: Send + Sync,
{
    type Class = TableType;

    fn class(&self) -> TableType {
        TableType::Index
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableInstance<F, D, Txn> for Index<F, D, Txn> {
    type OrderBy = IndexSlice<F, D, Txn>;
    type Reverse = IndexSlice<F, D, Txn>;
    type Slice = IndexSlice<F, D, Txn>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.btree.count(txn_id).await
    }

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        self.btree.delete(txn_id).await
    }

    async fn delete_row(&self, txn_id: TxnId, row: Row) -> TCResult<()> {
        let key = self.schema.values_from_row(row, false)?;
        let range = tc_btree::Range::with_prefix(key);
        self.btree.clone().slice(range, false)?.delete(txn_id).await
    }

    fn key(&self) -> &[Column] {
        self.schema.key()
    }

    fn values(&self) -> &[Column] {
        self.schema.values()
    }

    fn schema(&self) -> TableSchema {
        self.schema.clone().into()
    }

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        if self.schema.starts_with(&order) {
            Ok(IndexSlice::all(self.btree, self.schema, reverse))
        } else {
            Err(TCError::bad_request(
                &format!("Index with schema {} does not support order", self.schema),
                Value::from_iter(order),
            ))
        }
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        Ok(IndexSlice::all(self.btree, self.schema, true).into())
    }

    fn slice(self, bounds: Bounds) -> TCResult<IndexSlice<F, D, Txn>> {
        self.index_slice(bounds).map(|is| is.into())
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        debug!("Index::rows");
        self.btree.keys(txn_id).await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        if bounds.is_empty() {
            return Ok(());
        }

        let columns = self.schema.columns();
        let mut bounds = bounds.clone();
        let mut ordered_bounds = Vec::with_capacity(columns.len());
        for column in columns {
            let bound = bounds.remove(column.name()).unwrap_or_default();
            ordered_bounds.push(bound);

            if bounds.is_empty() {
                break;
            }
        }

        if !bounds.is_empty() {
            return Err(TCError::bad_request(
                "Index has no such columns: {}",
                Value::from_iter(bounds.keys().cloned()),
            ));
        }

        debug!(
            "ordered bounds: {}",
            Tuple::<&ColumnBound>::from_iter(&ordered_bounds)
        );
        if ordered_bounds[..ordered_bounds.len() - 1]
            .iter()
            .any(ColumnBound::is_range)
        {
            return Err(TCError::unsupported(
                "Index bounds must include a maximum of one range, only on the rightmost column",
            ));
        }

        Ok(())
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        if !self.schema.starts_with(&order) {
            let order: Vec<String> = order.iter().map(|c| c.to_string()).collect();
            Err(TCError::bad_request(
                &format!("cannot order index with schema {} by", self.schema),
                order.join(", "),
            ))
        } else {
            Ok(())
        }
    }

    async fn update(&self, _txn: &Txn, _row: Row) -> TCResult<()> {
        unimplemented!()
    }
}

#[async_trait]
impl<F: File<Node> + Transact, D: Dir, Txn: Transaction<D>> Transact for Index<F, D, Txn> {
    async fn commit(&self, txn_id: &TxnId) {
        self.btree.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.btree.finalize(txn_id).await
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Persist<D> for Index<F, D, Txn>
where
    F: TryFrom<D::File>,
{
    type Schema = IndexSchema;
    type Store = F;
    type Txn = Txn;

    fn schema(&self) -> &IndexSchema {
        &self.schema
    }

    async fn load(txn: &Txn, schema: IndexSchema, file: F) -> TCResult<Self> {
        BTreeFile::load(txn, schema.clone().into(), file)
            .map_ok(|btree| Self { schema, btree })
            .await
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Restore<D> for Index<F, D, Txn>
where
    F: TryFrom<D::File>,
{
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        self.btree.restore(&backup.btree, txn_id).await
    }
}

impl<F, D, Txn> From<Index<F, D, Txn>> for Table<F, D, Txn> {
    fn from(index: Index<F, D, Txn>) -> Self {
        Table::Index(index)
    }
}

#[derive(Clone)]
pub struct ReadOnly<F, D, Txn> {
    index: IndexSlice<F, D, Txn>,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> ReadOnly<F, D, Txn> {
    pub async fn copy_from<T: TableInstance<F, D, Txn>>(
        source: T,
        txn: Txn,
        key_columns: Option<Vec<Id>>,
    ) -> TCResult<Self>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        let file = txn
            .context()
            .create_file_tmp(*txn.id(), BTreeType::default())
            .await?;

        let source_schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        let (schema, btree) = if let Some(columns) = key_columns {
            let schema = source_schema.auxiliary(&columns)?;
            let btree = BTreeFile::create(file, schema.clone().into(), *txn.id()).await?;

            let source = source.select(columns)?;
            let rows = source.rows(*txn.id()).await?;
            btree.try_insert_from(*txn.id(), rows).await?;
            (schema, btree)
        } else {
            let btree = BTreeFile::create(file, source_schema.clone().into(), *txn.id()).await?;

            let rows = source.rows(*txn.id()).await?;
            btree.try_insert_from(*txn.id(), rows).await?;
            (source_schema, btree)
        };

        let index = Index { schema, btree };

        index
            .index_slice(Bounds::default())
            .map(|index| ReadOnly { index })
    }

    pub fn into_reversed(self) -> Self {
        ReadOnly {
            index: self.index.into_reversed(),
        }
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.index.is_empty(txn).await
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Instance for ReadOnly<F, D, Txn> {
    type Class = TableType;

    fn class(&self) -> Self::Class {
        Self::Class::ReadOnly
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableInstance<F, D, Txn> for ReadOnly<F, D, Txn> {
    type OrderBy = Self;
    type Reverse = Self;
    type Slice = Self;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.index.count(txn_id).await
    }

    fn key(&self) -> &[Column] {
        self.index.key()
    }

    fn values(&self) -> &[Column] {
        self.index.values()
    }

    fn schema(&self) -> TableSchema {
        TableInstance::schema(&self.index)
    }

    fn order_by(self, order: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.index.validate_order(&order)?;

        if reverse {
            Ok(self.into_reversed())
        } else {
            Ok(self)
        }
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        Ok(self.into_reversed())
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.validate_bounds(&bounds)?;

        self.index
            .slice_index(bounds)
            .map(|index| ReadOnly { index })
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.index.rows(txn_id).await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        self.index.validate_bounds(bounds)
    }

    fn validate_order(&self, order: &[Id]) -> TCResult<()> {
        self.index.validate_order(order)
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> From<ReadOnly<F, D, Txn>> for Table<F, D, Txn> {
    fn from(index: ReadOnly<F, D, Txn>) -> Table<F, D, Txn> {
        Self::ROIndex(index)
    }
}

struct Inner<F, D, Txn> {
    schema: TableSchema,
    primary: Index<F, D, Txn>,
    auxiliary: Vec<(Id, Index<F, D, Txn>)>,
}

/// The base type of a [`Table`].
#[derive(Clone)]
pub struct TableIndex<F, D, Txn> {
    inner: Arc<Inner<F, D, Txn>>,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableIndex<F, D, Txn> {
    /// Create a new `TableIndex` with the given [`TableSchema`].
    pub async fn create(
        context: &D,
        schema: TableSchema,
        txn_id: TxnId,
    ) -> TCResult<TableIndex<F, D, Txn>>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        let primary_file = context
            .create_file(txn_id, PRIMARY_INDEX.into(), BTreeType::default())
            .await?;

        let primary = Index::create(primary_file, schema.primary().clone(), txn_id).await?;

        let primary_schema = schema.primary();
        let auxiliary = try_join_all(
            schema
                .indices()
                .iter()
                .map(|(name, column_names)| (name.clone(), column_names.to_vec()))
                .map(|(name, column_names)| async {
                    if name == PRIMARY_INDEX {
                        return Err(TCError::bad_request(
                            "cannot create an auxiliary index with reserved name",
                            PRIMARY_INDEX,
                        ));
                    }

                    let file = context
                        .create_file(txn_id, name.clone(), BTreeType::default())
                        .await?;

                    Self::create_index(file, primary_schema, column_names, txn_id)
                        .map_ok(move |index| (name, index))
                        .await
                }),
        )
        .await?
        .into_iter()
        .collect();

        Ok(TableIndex {
            inner: Arc::new(Inner {
                schema,
                primary,
                auxiliary,
            }),
        })
    }

    async fn create_index(
        file: F,
        primary: &IndexSchema,
        key: Vec<Id>,
        txn_id: TxnId,
    ) -> TCResult<Index<F, D, Txn>> {
        let schema = primary.auxiliary(&key)?;
        let btree = BTreeFile::create(file, schema.clone().into(), txn_id).await?;
        Ok(Index { btree, schema })
    }

    /// Return `true` if this table has zero rows.
    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.inner.primary.is_empty(txn).await
    }

    /// Merge the given list of `Bounds` into a single `Bounds` instance.
    ///
    /// Returns an error in the case that later [`Bounds`] are larger than earlier [`Bounds`].
    pub fn merge_bounds(&self, all_bounds: Vec<Bounds>) -> TCResult<Bounds> {
        let collator = self.inner.primary.btree().collator();

        let mut merged = Bounds::default();
        for bounds in all_bounds {
            merged.merge(bounds, collator)?;
        }

        Ok(merged)
    }

    /// Borrow the primary `Index` of this `TableIndex`.
    pub fn primary(&self) -> &Index<F, D, Txn> {
        &self.inner.primary
    }

    /// Return an index which supports the given [`Bounds`], or an error if there is none.
    pub fn supporting_index(&self, bounds: &Bounds) -> TCResult<Index<F, D, Txn>> {
        if self.inner.primary.validate_bounds(bounds).is_ok() {
            return Ok(self.inner.primary.clone());
        }

        for (_, index) in &self.inner.auxiliary {
            if index.validate_bounds(bounds).is_ok() {
                return Ok(index.clone());
            }
        }

        Err(TCError::bad_request(
            "this table has no index which supports bounds",
            bounds,
        ))
    }

    /// Return a single row in this table with the given primary `key`, or `None` if there is none.
    pub async fn get(&self, txn_id: TxnId, key: Vec<Value>) -> TCResult<Option<Vec<Value>>> {
        self.inner.primary.get(txn_id, key).await
    }

    /// Insert a new row into this `TableIndex`, or update the row at the given `key` with `values`.
    pub async fn upsert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        let primary = &self.inner.primary;

        if let Some(row) = self.get(txn_id, key.to_vec()).await? {
            let row = primary.schema.row_from_values(row.to_vec())?;
            self.delete_row(txn_id, row.clone()).await?;
        }

        let row = primary.schema().row_from_key_values(key, values)?;
        let mut inserts = FuturesUnordered::new();
        inserts.push(primary.insert(txn_id, row.clone(), true));

        for (_, index) in &self.inner.auxiliary {
            inserts.push(index.insert(txn_id, row.clone(), false));
        }

        while let Some(()) = inserts.try_next().await? {}

        Ok(())
    }

    /// Stream the rows within the given [`Bounds`] from the primary index of this `TableIndex`.
    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.inner
            .primary
            .clone()
            .slice_rows(txn_id, bounds, reverse)
            .await
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Instance for TableIndex<F, D, Txn> {
    type Class = TableType;

    fn class(&self) -> TableType {
        TableType::Table
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableInstance<F, D, Txn>
    for TableIndex<F, D, Txn>
{
    type OrderBy = Merged<F, D, Txn>;
    type Reverse = Merged<F, D, Txn>;
    type Slice = Merged<F, D, Txn>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.inner.primary.clone().count(txn_id).await
    }

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        let aux = &self.inner.auxiliary;

        let mut deletes = Vec::with_capacity(aux.len() + 1);
        deletes.push(self.inner.primary.delete(txn_id));
        for (_, index) in aux {
            deletes.push(index.delete(txn_id));
        }

        try_join_all(deletes).await?;
        Ok(())
    }

    async fn delete_row(&self, txn_id: TxnId, row: Row) -> TCResult<()> {
        let aux = &self.inner.auxiliary;
        let row = self.inner.primary.schema().validate_row(row)?;

        let mut deletes = Vec::with_capacity(aux.len() + 1);
        for (_, index) in aux {
            deletes.push(index.delete_row(txn_id, row.clone()));
        }
        deletes.push(self.inner.primary.delete_row(txn_id, row));
        try_join_all(deletes).await?;

        Ok(())
    }

    fn key(&self) -> &[Column] {
        self.inner.primary.key()
    }

    fn values(&self) -> &[Column] {
        self.inner.primary.values()
    }

    fn schema(&self) -> TableSchema {
        self.inner.schema.clone()
    }

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.validate_order(&columns)?;

        let selection = TableSlice::new(self.clone(), Bounds::default())?;
        let merge_source = MergeSource::Table(selection);

        if self.primary().validate_order(&columns).is_ok() {
            debug!("primary key can order by {}", Tuple::from(columns.clone()));

            let index_slice = self.primary().clone().index_slice(Bounds::default())?;
            let merged = Merged::new(merge_source, index_slice)?;
            return if reverse {
                merged.reversed()
            } else {
                Ok(merged.into())
            };
        } else {
            for (name, index) in &self.inner.auxiliary {
                if index.validate_order(&columns).is_ok() {
                    debug!(
                        "index {} can order by {}",
                        name,
                        Tuple::from(columns.clone())
                    );

                    let index_slice = index.clone().index_slice(Bounds::default())?;
                    let merged = Merged::new(merge_source, index_slice)?;
                    return if reverse {
                        merged.reversed()
                    } else {
                        Ok(merged.into())
                    };
                }
            }
        }

        Err(TCError::bad_request(
            "table has no index to order by",
            Tuple::<Id>::from_iter(columns),
        ))
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        Err(TCError::unsupported(
            "cannot reverse a Table itself, consider reversing a slice of the table instead",
        ))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Merged<F, D, Txn>> {
        debug!("TableIndex::slice {}", bounds);

        let columns: Vec<Id> = self
            .inner
            .primary
            .schema()
            .columns()
            .iter()
            .map(|c| c.name())
            .cloned()
            .collect();

        let bounds: Vec<(Id, ColumnBound)> = columns
            .into_iter()
            .filter_map(|name| bounds.get(&name).map(|bound| (name, bound.clone())))
            .collect();

        let selection = TableSlice::new(self.clone(), Bounds::default())?;
        let mut merge_source = MergeSource::Table(selection);

        let mut bounds = &bounds[..];
        loop {
            let initial = bounds.len();
            let mut i = bounds.len();
            while i > 0 {
                let subset: HashMap<Id, ColumnBound> = bounds[..i].to_vec().into_iter().collect();
                let subset = Bounds::from(subset);

                if self.inner.primary.validate_bounds(&subset).is_ok() {
                    debug!("primary key can slice {}", subset);

                    let index_slice = self.inner.primary.clone().index_slice(subset)?;
                    let merged = Merged::new(merge_source, index_slice)?;

                    bounds = &bounds[i..];
                    if bounds.is_empty() {
                        return Ok(merged);
                    }

                    merge_source = MergeSource::Merge(Box::new(merged));
                    break;
                } else {
                    let mut supported = false;
                    for (name, index) in self.inner.auxiliary.iter() {
                        debug!("checking index {} with schema {}", name, index.schema());

                        match index.validate_bounds(&subset) {
                            Ok(()) => {
                                debug!("index {} can slice {}", name, subset);
                                supported = true;

                                let index_slice = index.clone().index_slice(subset)?;
                                let merged = Merged::new(merge_source, index_slice)?;

                                bounds = &bounds[i..];
                                if bounds.is_empty() {
                                    return Ok(merged);
                                }

                                merge_source = MergeSource::Merge(Box::new(merged));
                                break;
                            }
                            Err(cause) => {
                                debug!("index {} cannot slice {}: {}", name, subset, cause);
                            }
                        }
                    }

                    if supported {
                        break;
                    }
                };

                i = i - 1;
            }

            if bounds.len() == initial {
                return Err(TCError::unsupported(
                    "this Table has no Index to support the requested selection bounds",
                ));
            }
        }
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.inner.primary.clone().rows(txn_id).await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        if self.inner.primary.validate_bounds(bounds).is_ok() {
            return Ok(());
        }

        let bounds: Vec<(Id, ColumnBound)> = self
            .inner
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

            let mut i = bounds.len();
            loop {
                let subset: HashMap<Id, ColumnBound> = bounds[..i].iter().cloned().collect();
                let subset = Bounds::from(subset);

                if self.inner.primary.validate_bounds(&subset).is_ok() {
                    bounds = &bounds[i..];
                    break;
                }

                for (_, index) in &self.inner.auxiliary {
                    if index.validate_bounds(&subset).is_ok() {
                        bounds = &bounds[i..];
                        break;
                    }
                }

                if bounds.is_empty() {
                    break;
                } else {
                    i = i - 1;
                }
            }

            if bounds.len() == initial {
                let order: Vec<String> = bounds.iter().map(|(name, _)| name.to_string()).collect();
                return Err(TCError::bad_request(
                    format!("this table has no index to support selection bounds on {}--available indices are", order.join(", ")),
                    self.inner.auxiliary.iter().map(|(id, _)| id).collect::<Tuple<&Id>>(),
                ));
            }
        }

        Ok(())
    }

    fn validate_order(&self, mut order: &[Id]) -> TCResult<()> {
        while !order.is_empty() {
            let initial = order.to_vec();
            let mut i = order.len();
            loop {
                let subset = &order[..i];

                if self.inner.primary.validate_order(subset).is_ok() {
                    order = &order[i..];
                    break;
                }

                for (_, index) in &self.inner.auxiliary {
                    if index.validate_order(subset).is_ok() {
                        order = &order[i..];
                        break;
                    }
                }

                if order.is_empty() {
                    break;
                } else {
                    i = i - 1;
                }
            }

            if order == &initial[..] {
                let order: Vec<String> = order.iter().map(|id| id.to_string()).collect();
                return Err(TCError::bad_request(
                    "This table has no index to support the order",
                    order.join(", "),
                ));
            }
        }

        Ok(())
    }

    async fn update(&self, txn: &Txn, update: Row) -> TCResult<()>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        for col in self.inner.primary.schema().key() {
            if update.contains_key(col.name()) {
                return Err(TCError::bad_request(
                    "Cannot update the value of a primary key column",
                    col.name(),
                ));
            }
        }

        let schema = self.inner.primary.schema();
        let update = schema.validate_row_partial(update)?;

        let index = self.clone().index(txn.clone(), None).await?;
        let index = index.rows(*txn.id()).await?;

        index
            .map(|values| values.and_then(|values| schema.row_from_values(values)))
            .map_ok(|row| self.update_row(*txn.id(), row, update.clone()))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(())
    }

    async fn update_row(&self, txn_id: TxnId, row: Row, update: Row) -> TCResult<()> {
        let mut updated_row = row.clone();
        updated_row.extend(update);
        let (key, values) = self.inner.primary.schema.key_values_from_row(updated_row)?;

        self.upsert(txn_id, key, values).await
    }

    async fn upsert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        TableIndex::upsert(self, txn_id, key, values).await
    }
}

#[async_trait]
impl<F: File<Node> + Transact, D: Dir, Txn: Transaction<D>> Transact for TableIndex<F, D, Txn> {
    async fn commit(&self, txn_id: &TxnId) {
        let mut commits = Vec::with_capacity(self.inner.auxiliary.len() + 1);
        commits.push(self.inner.primary.commit(txn_id));
        for (_, index) in &self.inner.auxiliary {
            commits.push(index.commit(txn_id));
        }

        join_all(commits).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let mut cleanups = Vec::with_capacity(self.inner.auxiliary.len() + 1);
        cleanups.push(self.inner.primary.finalize(txn_id));
        for (_, index) in &self.inner.auxiliary {
            cleanups.push(index.finalize(txn_id));
        }

        join_all(cleanups).await;
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Persist<D> for TableIndex<F, D, Txn>
where
    F: TryFrom<D::File, Error = TCError>,
    <D as Dir>::FileClass: From<BTreeType> + Send,
{
    type Schema = TableSchema;
    type Store = D;
    type Txn = Txn;

    fn schema(&self) -> &Self::Schema {
        &self.inner.schema
    }

    async fn load(txn: &Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let file = store
            .get_file(txn.id(), &PRIMARY_INDEX.into())
            .await?
            .ok_or_else(|| TCError::internal("cannot load Table: primary index is missing"))?;

        let primary = Index::load(txn, schema.primary().clone(), file).await?;

        let mut auxiliary = Vec::with_capacity(schema.indices().len());
        for (name, columns) in schema.indices() {
            let file = store.get_file(txn.id(), name).await?.ok_or_else(|| {
                TCError::internal(format!("cannot load Table: missing index {}", name))
            })?;

            let index_schema = schema.primary().auxiliary(columns)?;

            let index = Index::load(txn, index_schema, file).await?;
            auxiliary.push((name.clone(), index));
        }

        Ok(Self {
            inner: Arc::new(Inner {
                schema,
                primary,
                auxiliary,
            }),
        })
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Restore<D> for TableIndex<F, D, Txn>
where
    F: TryFrom<D::File, Error = TCError>,
    <D as Dir>::FileClass: From<BTreeType> + Send,
{
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        if self.inner.schema != backup.inner.schema {
            return Err(TCError::unsupported(
                "cannot restore a Table using a backup with a different schema",
            ));
        }

        let mut restores = Vec::with_capacity(self.inner.auxiliary.len() + 1);
        restores.push(self.inner.primary.restore(&backup.inner.primary, txn_id));

        let mut backup_indices = BTreeMap::from_iter(
            backup
                .inner
                .auxiliary
                .iter()
                .map(|(name, index)| (name, index)),
        );

        for (name, index) in &self.inner.auxiliary {
            restores.push(index.restore(backup_indices.remove(name).unwrap(), txn_id));
        }

        try_join_all(restores).await?;

        Ok(())
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>, I: TableInstance<F, D, Txn> + 'static>
    CopyFrom<D, I> for TableIndex<F, D, Txn>
where
    F: TryFrom<D::File, Error = TCError>,
    <D as Dir>::FileClass: From<BTreeType> + Send,
{
    async fn copy_from(source: I, dir: D, txn: Txn) -> TCResult<Self> {
        let txn_id = *txn.id();
        let schema = source.schema();
        let key_len = schema.primary().key().len();
        let table = Self::create(&dir, schema, txn_id).await?;

        let rows = source.rows(txn_id).await?;

        rows.map_ok(|mut row| (row.drain(..key_len).collect(), row))
            .map_ok(|(key, values)| table.upsert(txn_id, key, values))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), ()| future::ready(Ok(())))
            .await?;

        Ok(table)
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> From<TableIndex<F, D, Txn>> for Table<F, D, Txn> {
    fn from(table: TableIndex<F, D, Txn>) -> Self {
        Self::Table(table)
    }
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> fmt::Display for TableIndex<F, D, Txn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Table")
    }
}
