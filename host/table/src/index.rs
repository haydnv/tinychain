use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, join_all, try_join_all, TryFutureExt};
use futures::stream::TryStreamExt;
use log::debug;

use tc_btree::{BTreeFile, BTreeInstance, BTreeWrite, Node, NodeId};
use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, DirReadFile, File, DirCreateFile, Persist, Restore};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{label, Id, Instance, Label, TCBoxTryStream, Tuple};

use super::view::{Limited, MergeSource, Merged, Selection, TableSlice as Slice};
use super::{
    Bounds, Column, ColumnBound, IndexSchema, IndexSlice, Key, Row, Table, TableInstance,
    TableOrder, TableRead, TableSchema, TableSlice, TableStream, TableType, TableWrite, Values,
};

const PRIMARY_INDEX: Label = label("primary");

#[derive(Clone)]
pub struct Index<F, D, Txn> {
    btree: BTreeFile<F, D, Txn>,
    schema: IndexSchema,
}

impl<F, D, Txn> Index<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    D::Write: DirCreateFile<F>,
{
    pub async fn create(file: F, schema: IndexSchema, txn_id: TxnId) -> TCResult<Self> {
        BTreeFile::create(file, schema.clone().into(), txn_id)
            .map_ok(|btree| Index { btree, schema })
            .await
    }

    pub fn btree(&'_ self) -> &'_ BTreeFile<F, D, Txn> {
        &self.btree
    }

    pub async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.btree.is_empty(*txn.id()).await
    }

    pub fn index_slice(self, bounds: Bounds) -> TCResult<IndexSlice<F, D, Txn>> {
        debug!("Index::index_slice");
        let bounds = bounds.validate(&self.schema.columns())?;
        IndexSlice::new(self.btree, self.schema, bounds)
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
    ) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        self.validate_bounds(&bounds)?;
        let range = bounds.into_btree_range(&self.schema.columns())?;
        self.btree.slice(range, reverse)?.keys(txn_id).await
    }

    async fn delete_inner(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        debug!("Index::delete {:?}", key);
        let range = tc_btree::Range::with_prefix(key.to_vec());
        self.btree.delete(txn_id, range).await
    }

    async fn delete(&self, txn_id: TxnId, mut row: Row) -> TCResult<()> {
        let key = self
            .schema
            .key()
            .iter()
            .map(|col| {
                row.remove(&col.name)
                    .ok_or_else(|| TCError::bad_request("missing value for column", &col.name))
            })
            .collect::<TCResult<Key>>()?;

        self.delete_inner(txn_id, key).await
    }

    async fn replace(&self, txn_id: TxnId, mut row: Row, mut update: Row) -> TCResult<()> {
        debug!("Index::replace {} with updated values {}", row, update);

        let old_key = self
            .schema
            .key()
            .iter()
            .map(|col| {
                row.get(&col.name)
                    .cloned()
                    .ok_or_else(|| TCError::bad_request("missing value for column", &col.name))
            })
            .collect::<TCResult<Key>>()?;

        let new_key = BTreeInstance::schema(&self.btree)
            .iter()
            .map(|col| {
                if let Some(value) = update.remove(&col.name) {
                    Ok(value)
                } else if let Some(value) = row.remove(&col.name) {
                    Ok(value)
                } else {
                    Err(TCError::bad_request("missing value for column", &col.name))
                }
            })
            .collect::<TCResult<Key>>()?;

        self.delete_inner(txn_id, old_key).await?;
        self.btree.insert(txn_id, new_key).await
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

impl<F: File<Key = NodeId, Block = Node>, D: Dir, Txn: Transaction<D>> TableInstance
    for Index<F, D, Txn>
{
    fn key(&self) -> &[Column] {
        self.schema.key()
    }

    fn values(&self) -> &[Column] {
        self.schema.values()
    }

    fn schema(&self) -> TableSchema {
        self.schema.clone().into()
    }
}

impl<F, D, Txn> TableOrder for Index<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
{
    type OrderBy = IndexSlice<F, D, Txn>;
    type Reverse = IndexSlice<F, D, Txn>;

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

    fn reverse(self) -> TCResult<Self::Reverse> {
        Ok(IndexSlice::all(self.btree, self.schema, true).into())
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
}

#[async_trait]
impl<F, D, Txn> TableStream for Index<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    D::Write: DirCreateFile<F>,
{
    type Limit = Limited<F, D, Txn>;
    type Selection = Selection<F, D, Txn, Self>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.btree.count(txn_id).await
    }

    fn limit(self, limit: u64) -> Self::Limit {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        debug!("Index::rows");
        self.btree.keys(txn_id).await
    }
}

impl<F, D, Txn> TableSlice for Index<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    D::Write: DirCreateFile<F>,
{
    type Slice = IndexSlice<F, D, Txn>;

    fn slice(self, bounds: Bounds) -> TCResult<IndexSlice<F, D, Txn>> {
        self.index_slice(bounds).map(|is| is.into())
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
}

#[async_trait]
impl<F, D, Txn> Transact for Index<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node> + Transact,
    D: Dir,
    Txn: Transaction<D>,
{
    type Commit = <BTreeFile<F, D, Txn> as Transact>::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        self.btree.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.btree.finalize(txn_id).await
    }
}

#[async_trait]
impl<F, D, Txn> Persist<D> for Index<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
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
impl<F, D, Txn> Restore<D> for Index<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
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

impl<F: File<Key = NodeId, Block = Node>, D: Dir, Txn: Transaction<D>> TableIndex<F, D, Txn>
where
    D::Write: DirCreateFile<F>,
{
    /// Create a new `TableIndex` with the given [`TableSchema`].
    pub async fn create(
        context: &D,
        schema: TableSchema,
        txn_id: TxnId,
    ) -> TCResult<TableIndex<F, D, Txn>> {
        let mut dir = context.write(txn_id).await?;

        let primary_file = dir.create_file(PRIMARY_INDEX.into())?;
        let primary = Index::create(primary_file, schema.primary().clone(), txn_id).await?;

        let primary_schema = schema.primary();
        let mut auxiliary = Vec::with_capacity(schema.indices().len());
        for (name, column_names) in schema.indices() {
            if name == &PRIMARY_INDEX {
                return Err(TCError::bad_request(
                    "cannot create an auxiliary index with reserved name",
                    PRIMARY_INDEX,
                ));
            }

            let file = dir.create_file(name.clone())?;
            let index = Self::create_index(file, primary_schema, column_names.to_vec(), txn_id)
                .map_ok(move |index| (name.clone(), index))
                .await?;

            auxiliary.push(index);
        }

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

    /// Stream the rows within the given [`Bounds`] from the primary index of this `TableIndex`.
    pub async fn slice_rows<'a>(
        self,
        txn_id: TxnId,
        bounds: Bounds,
        reverse: bool,
    ) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        self.inner
            .primary
            .clone()
            .slice_rows(txn_id, bounds, reverse)
            .await
    }
}

impl<F, D, Txn> Instance for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
{
    type Class = TableType;

    fn class(&self) -> TableType {
        TableType::Table
    }
}

#[async_trait]
impl<F: File<Key = NodeId, Block = Node>, D: Dir, Txn: Transaction<D>> TableInstance
    for TableIndex<F, D, Txn>
{
    fn key(&self) -> &[Column] {
        self.inner.primary.key()
    }

    fn values(&self) -> &[Column] {
        self.inner.primary.values()
    }

    fn schema(&self) -> TableSchema {
        self.inner.schema.clone()
    }
}

impl<F, D, Txn> TableOrder for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    D::Write: DirCreateFile<F>,
{
    type OrderBy = Merged<F, D, Txn>;
    type Reverse = Merged<F, D, Txn>;

    fn order_by(self, columns: Vec<Id>, reverse: bool) -> TCResult<Self::OrderBy> {
        self.validate_order(&columns)?;

        let selection = Slice::new(self.clone(), Bounds::default())?;
        let merge_source = MergeSource::Table(selection);

        if self.primary().validate_order(&columns).is_ok() {
            debug!("primary key can order by {}", Tuple::from(columns.clone()));

            let index_slice = self.primary().clone().index_slice(Bounds::default())?;
            let merged = Merged::new(merge_source, index_slice)?;
            return if reverse {
                merged.reverse()
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
                        merged.reverse()
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

    fn reverse(self) -> TCResult<Self::Reverse> {
        Err(TCError::unsupported(
            "cannot reverse a Table itself, consider reversing a slice of the table instead",
        ))
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
}

#[async_trait]
impl<F, D, Txn> TableRead for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
{
    async fn read(&self, txn_id: &TxnId, key: &Key) -> TCResult<Option<Vec<Value>>> {
        let slice = self
            .inner
            .primary
            .btree
            .clone()
            .slice(tc_btree::Range::with_prefix(key.to_vec()), false)?;

        let mut keys = slice.keys(*txn_id).await?;
        keys.try_next().await
    }
}

#[async_trait]
impl<F: File<Key = NodeId, Block = Node>, D: Dir, Txn: Transaction<D>> TableStream
    for TableIndex<F, D, Txn>
where
    D::Write: DirCreateFile<F>,
{
    type Limit = Limited<F, D, Txn>;
    type Selection = Selection<F, D, Txn, Self>;

    async fn count(self, txn_id: TxnId) -> TCResult<u64> {
        self.inner.primary.clone().count(txn_id).await
    }

    fn limit(self, limit: u64) -> Self::Limit {
        Limited::new(self, limit)
    }

    fn select(self, columns: Vec<Id>) -> TCResult<Self::Selection> {
        Selection::new(self, columns)
    }

    async fn rows<'a>(self, txn_id: TxnId) -> TCResult<TCBoxTryStream<'a, Vec<Value>>> {
        self.inner.primary.clone().rows(txn_id).await
    }
}

impl<F, D, Txn> TableSlice for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    D::Write: DirCreateFile<F>,
{
    type Slice = Merged<F, D, Txn>;

    fn slice(self, bounds: Bounds) -> TCResult<Merged<F, D, Txn>> {
        debug!("TableIndex::slice {}", bounds);

        let primary = &self.inner.primary;
        let auxiliary = &self.inner.auxiliary;

        let columns: Vec<Id> = primary
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

        let selection = Slice::new(self.clone(), Bounds::default())?;
        let mut merge_source = MergeSource::Table(selection);

        let mut bounds = &bounds[..];
        loop {
            let initial = bounds.len();
            let mut i = bounds.len();
            while i > 0 {
                let subset: HashMap<Id, ColumnBound> = bounds[..i].to_vec().into_iter().collect();
                let subset = Bounds::from(subset);

                if primary.validate_bounds(&subset).is_ok() {
                    debug!("primary key can slice {}", subset);

                    let index_slice = primary.clone().index_slice(subset)?;
                    let merged = Merged::new(merge_source, index_slice)?;

                    bounds = &bounds[i..];
                    if bounds.is_empty() {
                        return Ok(merged);
                    }

                    merge_source = MergeSource::Merge(Box::new(merged));
                    break;
                } else {
                    let mut supported = false;
                    for (name, index) in auxiliary {
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

    fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        let primary = &self.inner.primary;
        let auxiliary = &self.inner.auxiliary;

        if primary.validate_bounds(bounds).is_ok() {
            return Ok(());
        }

        let bounds: Vec<(Id, ColumnBound)> = primary
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

                if primary.validate_bounds(&subset).is_ok() {
                    bounds = &bounds[i..];
                    break;
                }

                for (_, index) in auxiliary {
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
                let bounds = Tuple::<String>::from_iter(
                    bounds
                        .into_iter()
                        .map(|(id, bound)| format!("{}: {}", id, bound)),
                );

                return Err(TCError::unsupported(format!("this table has no index to support selection bounds on {}--available indices are {}", bounds, Tuple::<&Id>::from_iter(auxiliary.iter().map(|(id, _)| id)))));
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<F, D, Txn> TableWrite for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    D::Write: DirCreateFile<F>,
{
    async fn delete(&self, txn_id: TxnId, key: Key) -> TCResult<()> {
        let primary = &self.inner.primary;
        let aux = &self.inner.auxiliary;

        let key = primary.schema.validate_key(key)?;
        let row = match self.read(&txn_id, &key).await? {
            Some(row) => row,
            None => return Ok(()),
        };

        let row = primary.schema.row_from_values(row)?;

        let mut deletes = Vec::with_capacity(aux.len() + 1);
        for (_, index) in aux {
            deletes.push(index.delete(txn_id, row.clone()));
        }

        deletes.push(primary.delete(txn_id, row));
        try_join_all(deletes).await?;

        Ok(())
    }

    async fn update(&self, txn_id: TxnId, key: Key, values: Row) -> TCResult<()> {
        let columns_updated: HashSet<Id> = values.keys().cloned().collect();

        let primary = &self.inner.primary;
        let aux = &self.inner.auxiliary;

        let key = primary.schema.validate_key(key)?;
        let row = match self.read(&txn_id, &key).await? {
            Some(values) => primary.schema.row_from_values(values)?,
            None => return Ok(()),
        };

        let mut updates = Vec::with_capacity(aux.len() + 1);
        for (_, index) in aux {
            if !index
                .schema
                .column_names()
                .any(|name| columns_updated.contains(name))
            {
                continue;
            }

            updates.push(index.replace(txn_id, row.clone(), values.clone()));
        }

        updates.push(primary.replace(txn_id, row, values));
        try_join_all(updates).await?;

        Ok(())
    }

    async fn upsert(&self, txn_id: TxnId, key: Key, values: Values) -> TCResult<()> {
        let primary = &self.inner.primary;
        let aux = &self.inner.auxiliary;

        let key = primary.schema.validate_key(key)?;
        let values = primary.schema.validate_values(values)?;

        let columns: HashSet<Id> = primary
            .schema
            .values()
            .iter()
            .map(|col| &col.name)
            .cloned()
            .collect();

        let row = primary.schema.row_from_key_values(key, values)?;
        let update: Row = row
            .clone()
            .into_iter()
            .filter(|(id, _)| columns.contains(id))
            .collect();

        let mut upserts = Vec::with_capacity(aux.len() + 1);
        for (_name, index) in aux {
            upserts.push(index.replace(txn_id, row.clone(), update.clone()));
        }

        upserts.push(primary.replace(txn_id, row, update));
        try_join_all(upserts).await?;

        Ok(())
    }
}

#[async_trait]
impl<F, D, Txn> Transact for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node> + Transact,
    D: Dir,
    Txn: Transaction<D>,
{
    type Commit = <BTreeFile<F, D, Txn> as Transact>::Commit;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        let guard = self.inner.primary.commit(txn_id).await;

        let mut commits = Vec::with_capacity(self.inner.auxiliary.len());
        for (_, index) in &self.inner.auxiliary {
            commits.push(index.commit(txn_id));
        }

        join_all(commits).await;

        guard
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
impl<F, D, Txn> Persist<D> for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    D::Read: DirReadFile<F>,
    D::Write: DirCreateFile<F>,
{
    type Schema = TableSchema;
    type Store = D;
    type Txn = Txn;

    fn schema(&self) -> &Self::Schema {
        &self.inner.schema
    }

    async fn load(txn: &Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        let dir = store.read(*txn.id()).await?;

        let file = dir
            .get_file(&PRIMARY_INDEX.into())?
            .ok_or_else(|| TCError::internal("cannot load Table: primary index is missing"))?;

        let primary = Index::load(txn, schema.primary().clone(), file).await?;

        let mut auxiliary = Vec::with_capacity(schema.indices().len());
        for (name, columns) in schema.indices() {
            let file = dir.get_file(name)?.ok_or_else(|| {
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
impl<F, D, Txn> Restore<D> for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    D::Read: DirReadFile<F>,
    D::Write: DirCreateFile<F>,
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
impl<F, D, Txn, I> CopyFrom<D, I> for TableIndex<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
    I: TableStream + 'static,
    D::Read: DirReadFile<F>,
    D::Write: DirCreateFile<F>,
{
    async fn copy_from(source: I, dir: D, txn: &Txn) -> TCResult<Self> {
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

impl<F, D, Txn> From<TableIndex<F, D, Txn>> for Table<F, D, Txn>
where
    F: File<Key = NodeId, Block = Node>,
    D: Dir,
    Txn: Transaction<D>,
{
    fn from(table: TableIndex<F, D, Txn>) -> Self {
        Self::Table(table)
    }
}

impl<F: File<Key = NodeId, Block = Node>, D: Dir, Txn: Transaction<D>> fmt::Display
    for TableIndex<F, D, Txn>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Table")
    }
}
