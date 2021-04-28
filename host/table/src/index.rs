use std::collections::{BTreeMap, HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use futures::future::{self, join_all, try_join_all, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};
use log::debug;

use tc_btree::{BTreeFile, BTreeInstance, BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{Dir, File, Persist};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, Id, Instance, Label, TCTryStream};

use super::view::{MergeSource, Merged, TableSlice};
use super::{
    Bounds, Column, ColumnBound, IndexSchema, IndexSlice, Row, Table, TableInstance, TableSchema,
    TableType,
};

const PRIMARY_INDEX: Label = label("primary");

#[derive(Clone)]
pub struct Index<F: File<Node>, D: Dir, Txn: Transaction<D>> {
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
                "Slice does not contain requested bounds",
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

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Instance for Index<F, D, Txn> {
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
                &format!("Cannot order index with schema {} by", self.schema),
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

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> From<Index<F, D, Txn>> for Table<F, D, Txn> {
    fn from(index: Index<F, D, Txn>) -> Self {
        Table::Index(index)
    }
}

#[derive(Clone)]
pub struct ReadOnly<F: File<Node>, D: Dir, Txn: Transaction<D>> {
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
        let file = file.try_into()?;

        let source_schema: IndexSchema = (source.key().to_vec(), source.values().to_vec()).into();

        let (schema, btree) = if let Some(columns) = key_columns {
            let column_names: HashSet<&Id> = columns.iter().collect();
            let schema = source_schema.subset(column_names)?;
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

#[derive(Clone)]
pub struct TableIndex<F: File<Node>, D: Dir, Txn: Transaction<D>> {
    schema: TableSchema,
    primary: Index<F, D, Txn>,
    auxiliary: BTreeMap<Id, Index<F, D, Txn>>,
}

impl<F: File<Node>, D: Dir, Txn: Transaction<D>> TableIndex<F, D, Txn> {
    pub async fn create(
        schema: TableSchema,
        context: &D,
        txn_id: TxnId,
    ) -> TCResult<TableIndex<F, D, Txn>>
    where
        F: TryFrom<D::File, Error = TCError>,
        D::FileClass: From<BTreeType>,
    {
        let primary_file = context
            .create_file(txn_id, PRIMARY_INDEX.into(), BTreeType::default())
            .await?;

        let primary_file = primary_file.try_into()?;
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

                    let file = file.try_into()?;

                    Self::create_index(file, primary_schema, column_names, txn_id)
                        .map_ok(move |index| (name, index))
                        .await
                }),
        )
        .await?
        .into_iter()
        .collect();

        Ok(TableIndex {
            schema,
            primary,
            auxiliary,
        })
    }

    async fn create_index(
        file: F,
        primary: &IndexSchema,
        key: Vec<Id>,
        txn_id: TxnId,
    ) -> TCResult<Index<F, D, Txn>> {
        let index_key_set: HashSet<&Id> = key.iter().collect();
        if index_key_set.len() != key.len() {
            return Err(TCError::bad_request(
                "duplicate column in index",
                key.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        let mut columns: HashMap<Id, Column> = primary
            .columns()
            .iter()
            .cloned()
            .map(|c| (c.name().clone(), c))
            .collect();

        let key: Vec<Column> = key
            .iter()
            .map(|c| columns.remove(&c).ok_or_else(|| TCError::not_found(c)))
            .collect::<TCResult<Vec<Column>>>()?;

        let values: Vec<Column> = primary
            .key()
            .iter()
            .filter(|c| !index_key_set.contains(c.name()))
            .cloned()
            .collect();

        let schema: IndexSchema = (key, values).into();

        let btree = BTreeFile::create(file, schema.clone().into(), txn_id).await?;

        Ok(Index { btree, schema })
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.primary.is_empty(txn).await
    }

    pub fn merge_bounds(&self, all_bounds: Vec<Bounds>) -> TCResult<Bounds> {
        let collator = self.primary.btree().collator();

        let mut merged = Bounds::default();
        for bounds in all_bounds {
            merged.merge(bounds, collator)?;
        }

        Ok(merged)
    }

    pub fn primary(&'_ self) -> &'_ Index<F, D, Txn> {
        &self.primary
    }

    pub fn supporting_index(&self, bounds: &Bounds) -> TCResult<Index<F, D, Txn>> {
        if self.primary.validate_bounds(bounds).is_ok() {
            return Ok(self.primary.clone());
        }

        for index in self.auxiliary.values() {
            if index.validate_bounds(bounds).is_ok() {
                return Ok(index.clone());
            }
        }

        Err(TCError::bad_request(
            "This table has no index which supports bounds",
            bounds,
        ))
    }

    pub async fn get(&self, txn_id: TxnId, key: Vec<Value>) -> TCResult<Option<Vec<Value>>> {
        self.primary.get(txn_id, key).await
    }

    pub async fn insert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        if self.get(txn_id, key.to_vec()).await?.is_some() {
            let key: Vec<String> = key.iter().map(|v| v.to_string()).collect();
            Err(TCError::bad_request(
                "Tried to insert but this key already exists",
                format!("[{}]", key.join(", ")),
            ))
        } else {
            self.upsert(txn_id, key, values).await
        }
    }

    pub async fn upsert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        if let Some(row) = self.get(txn_id, key.to_vec()).await? {
            let row = self.primary.schema.row_from_values(row.to_vec())?;
            self.delete_row(txn_id, row.clone()).await?;
        }

        let row = self.primary.schema().row_from_key_values(key, values)?;

        let mut inserts = Vec::with_capacity(self.auxiliary.len() + 1);
        inserts.push(self.primary.insert(txn_id, row.clone(), true));
        for index in self.auxiliary.values() {
            inserts.push(index.insert(txn_id, row.clone(), false));
        }

        try_join_all(inserts).await?;
        Ok(())
    }

    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.primary.slice_rows(txn_id, bounds, reverse).await
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
        self.primary.count(txn_id).await
    }

    async fn delete(&self, txn_id: TxnId) -> TCResult<()> {
        let mut deletes = Vec::with_capacity(self.auxiliary.len() + 1);
        deletes.push(self.primary.delete(txn_id));
        for index in self.auxiliary.values() {
            deletes.push(index.delete(txn_id));
        }

        try_join_all(deletes).await?;
        Ok(())
    }

    async fn delete_row(&self, txn_id: TxnId, row: Row) -> TCResult<()> {
        let row = self.primary.schema().validate_row(row)?;

        let mut deletes = Vec::with_capacity(self.auxiliary.len() + 1);
        for index in self.auxiliary.values() {
            deletes.push(index.delete_row(txn_id, row.clone()));
        }
        deletes.push(self.primary.delete_row(txn_id, row));
        try_join_all(deletes).await?;

        Ok(())
    }

    async fn insert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        TableIndex::insert(self, txn_id, key, values).await
    }

    fn key(&self) -> &[Column] {
        self.primary.key()
    }

    fn values(&self) -> &[Column] {
        self.primary.values()
    }

    fn schema(&self) -> TableSchema {
        self.schema.clone()
    }

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.validate_order(&columns)?;

        let selection = TableSlice::new(self.clone(), Bounds::default())?;
        let mut merge_source = MergeSource::Table(selection);

        let mut columns = &columns[..];
        loop {
            let initial = columns.to_vec();

            let mut i = 0;
            while i < columns.len() {
                let subset = &columns[i..];
                let mut supported = false;

                if self.primary.validate_order(subset).is_ok() {
                    supported = true;
                    columns = &columns[..i];
                    i = 0;

                    debug!(
                        "primary key supports order {}",
                        Value::from_iter(subset.to_vec())
                    );

                    let index_slice = self.primary.clone().index_slice(Bounds::default())?;
                    let merged = Merged::new(merge_source, index_slice)?;

                    if columns.is_empty() {
                        return if reverse {
                            merged.reversed()
                        } else {
                            Ok(merged.into())
                        };
                    }

                    merge_source = MergeSource::Merge(Box::new(merged));
                } else {
                    debug!(
                        "primary key does not support order {}",
                        Value::from_iter(subset.to_vec())
                    );

                    for (name, index) in self.auxiliary.iter() {
                        if index.validate_order(subset).is_ok() {
                            supported = true;
                            columns = &columns[..i];
                            i = 0;

                            debug!(
                                "index {} supports order {}",
                                name,
                                Value::from_iter(subset.to_vec())
                            );

                            let index_slice = index.clone().index_slice(Bounds::default())?;
                            let merged = Merged::new(merge_source, index_slice)?;

                            if columns.is_empty() {
                                return if reverse {
                                    merged.reversed()
                                } else {
                                    Ok(merged.into())
                                };
                            }

                            merge_source = MergeSource::Merge(Box::new(merged));
                            break;
                        } else {
                            debug!(
                                "index {} does not support order {}",
                                name,
                                Value::from_iter(subset.to_vec())
                            );
                        }
                    }
                }

                if !supported {
                    i = i + 1;
                }
            }

            if columns == &initial[..] {
                return Err(TCError::bad_request(
                    "This table has no index to support the order",
                    Value::from_iter(columns.to_vec()),
                ));
            }
        }
    }

    fn reversed(self) -> TCResult<Self::Reverse> {
        Err(TCError::unsupported(
            "Cannot reverse a Table itself, consider reversing a slice of the table instead",
        ))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Merged<F, D, Txn>> {
        let columns: Vec<Id> = self
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

                if self.primary.validate_bounds(&subset).is_ok() {
                    debug!("primary key can slice {}", subset);

                    let index_slice = self.primary.clone().index_slice(subset)?;
                    let merged = Merged::new(merge_source, index_slice)?;

                    bounds = &bounds[i..];
                    if bounds.is_empty() {
                        return Ok(merged);
                    }

                    merge_source = MergeSource::Merge(Box::new(merged));
                } else {
                    for (name, index) in self.auxiliary.iter() {
                        if index.validate_bounds(&subset).is_ok() {
                            debug!("index {} can slice {}", name, subset);
                            let index_slice = index.clone().index_slice(subset)?;
                            let merged = Merged::new(merge_source, index_slice)?;

                            bounds = &bounds[i..];
                            if bounds.is_empty() {
                                return Ok(merged);
                            }

                            merge_source = MergeSource::Merge(Box::new(merged));
                            break;
                        }
                    }
                };

                i = i - 1;
            }

            if bounds.len() == initial {
                return Err(TCError::unsupported(
                    "This table has no index to support the requested selection bounds",
                ));
            }
        }
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCTryStream<'a, Vec<Value>>> {
        self.primary.rows(txn_id).await
    }

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        if self.primary.validate_bounds(bounds).is_ok() {
            return Ok(());
        }

        let bounds: Vec<(Id, ColumnBound)> = self
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

                if self.primary.validate_bounds(&subset).is_ok() {
                    bounds = &bounds[i..];
                    break;
                }

                for index in self.auxiliary.values() {
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
                    format!("This table has no index to support selection bounds on {}--available indices are", order.join(", ")),
                    Value::from_iter(self.auxiliary.keys().cloned()),
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

                if self.primary.validate_order(subset).is_ok() {
                    order = &order[i..];
                    break;
                }

                for index in self.auxiliary.values() {
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
        for col in self.primary.schema().key() {
            if update.contains_key(col.name()) {
                return Err(TCError::bad_request(
                    "Cannot update the value of a primary key column",
                    col.name(),
                ));
            }
        }

        let schema = self.primary.schema();
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
        let (key, values) = self.primary.schema.key_values_from_row(updated_row)?;
        self.delete_row(txn_id, row)
            .and_then(|()| self.insert(txn_id, key, values))
            .await
    }

    async fn upsert(&self, txn_id: TxnId, key: Vec<Value>, values: Vec<Value>) -> TCResult<()> {
        TableIndex::upsert(self, txn_id, key, values).await
    }
}

#[async_trait]
impl<F: File<Node> + Transact, D: Dir, Txn: Transaction<D>> Transact for TableIndex<F, D, Txn> {
    async fn commit(&self, txn_id: &TxnId) {
        let mut commits = Vec::with_capacity(self.auxiliary.len() + 1);
        commits.push(self.primary.commit(txn_id));
        for index in self.auxiliary.values() {
            commits.push(index.commit(txn_id));
        }

        join_all(commits).await;
    }

    async fn finalize(&self, txn_id: &TxnId) {
        let mut cleanups = Vec::with_capacity(self.auxiliary.len() + 1);
        cleanups.push(self.primary.finalize(txn_id));
        for index in self.auxiliary.values() {
            cleanups.push(index.finalize(txn_id));
        }

        join_all(cleanups).await;
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, Txn: Transaction<D>> Persist for TableIndex<F, D, Txn> {
    type Schema = TableSchema;
    type Store = D;

    fn schema(&self) -> &Self::Schema {
        &self.schema
    }

    async fn load(_schema: Self::Schema, _store: Self::Store, _txn_id: TxnId) -> TCResult<Self> {
        todo!()
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
