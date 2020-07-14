use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, try_join_all};
use futures::stream::{StreamExt, TryStreamExt};

use crate::error;
use crate::state::btree::{self, BTree, BTreeRange, Key};
use crate::transaction::{Txn, TxnId};
use crate::value::class::Impl;
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::{Bounds, Row, Schema, Selection};

pub struct Index {
    btree: Arc<BTree>,
    schema: Schema,
}

impl Index {
    fn key_from_row(&self, mut row: Row) -> TCResult<Key> {
        let mut key = Vec::with_capacity(row.len());
        for column in &self.schema.columns()[0..row.len()] {
            if let Some(value) = row.remove(&column.name) {
                value.expect(column.dtype, &format!("for column {}", column.name))?;
                key.push(value)
            } else {
                return Err(error::bad_request(
                    "Update is missing a value for column",
                    &column.name,
                ));
            }
        }

        if !row.is_empty() {
            return Err(error::bad_request(
                "Tried to update unknown columns",
                row.keys()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(", "),
            ));
        }

        Ok(key)
    }

    pub async fn is_empty(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.btree.is_empty(txn_id).await
    }

    pub async fn len(&self, txn_id: TxnId) -> TCResult<u64> {
        self.btree.clone().len(txn_id, btree::Selector::all()).await
    }

    pub async fn contains(&self, txn_id: TxnId, key: Key) -> TCResult<bool> {
        Ok(self.btree.clone().len(txn_id, key.into()).await? > 0)
    }

    pub async fn reversed(
        &self,
        txn_id: TxnId,
        range: BTreeRange,
    ) -> TCResult<TCStream<Vec<Value>>> {
        self.btree
            .clone()
            .slice(txn_id, btree::Selector::reverse(range))
            .await
    }

    async fn delete_row(&self, txn_id: &TxnId, key: Key) -> TCResult<()> {
        self.btree.delete(txn_id, btree::Selector::Key(key)).await
    }

    async fn insert(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        let key = self.schema().into_key(row)?;
        self.btree.insert(txn_id, key).await
    }
}

#[async_trait]
impl Selection for Index {
    type Stream = TCStream<Vec<Value>>;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.len(txn_id).await
    }

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()> {
        self.btree.delete(&txn_id, btree::Selector::all()).await
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.btree
            .clone()
            .slice(txn_id, btree::Selector::all())
            .await
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()> {
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
    }

    async fn update(self: Arc<Self>, txn: Arc<Txn>, row: Row) -> TCResult<()> {
        let value = self.key_from_row(row)?;
        self.btree
            .update(txn.id(), &btree::Selector::all(), &value)
            .await
    }
}

pub struct ReadOnly {
    index: Arc<Index>,
}

impl ReadOnly {
    pub async fn copy_from<S: Selection>(
        source: Arc<S>,
        txn: Arc<Txn>,
        key_columns: Option<Vec<ValueId>>,
    ) -> TCResult<ReadOnly> {
        let btree_file = txn
            .clone()
            .subcontext_tmp()
            .await?
            .context()
            .create_file(txn.id().clone(), "index".parse()?)
            .await?;

        let (schema, btree) = if let Some(columns) = key_columns {
            let column_names: HashSet<&ValueId> = columns.iter().collect();
            let schema = source.schema().subset(column_names)?;
            let btree = BTree::create(txn.id().clone(), schema.clone().into(), btree_file).await?;

            let rows = source.select(columns)?.stream(txn.id().clone()).await?;
            btree.insert_from(txn.id(), rows).await?;
            (schema, btree)
        } else {
            let schema = source.schema().clone();
            let btree = BTree::create(txn.id().clone(), schema.clone().into(), btree_file).await?;
            let rows = source.stream(txn.id().clone()).await?;
            btree.insert_from(txn.id(), rows).await?;
            (schema, btree)
        };

        let index = Arc::new(Index {
            schema,
            btree: Arc::new(btree),
        });

        Ok(ReadOnly { index })
    }
}

#[async_trait]
impl Selection for ReadOnly {
    type Stream = TCStream<Vec<Value>>;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.index.clone().count(txn_id).await
    }

    async fn delete(self: Arc<Self>, _txn_id: TxnId) -> TCResult<()> {
        Err(error::method_not_allowed(
            "this is a transitive (read-only) index",
        ))
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.index.schema()
    }

    async fn stream(self: Arc<Self>, _txn_id: TxnId) -> TCResult<Self::Stream> {
        Err(error::not_implemented())
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()> {
        self.index.validate(bounds)
    }

    async fn update(self: Arc<Self>, _txn: Arc<Txn>, _value: Row) -> TCResult<()> {
        Err(error::method_not_allowed(
            "this is a transitive (read-only) index",
        ))
    }
}

pub struct IndexTable {
    index: Arc<Index>,
    auxiliary: BTreeMap<ValueId, Arc<Index>>,
}

impl IndexTable {
    async fn delete_row(&self, txn_id: &TxnId, row: &Row) -> TCResult<()> {
        self.schema().validate_row(row)?;

        let mut deletes = Vec::with_capacity(self.auxiliary.len() + 1);
        deletes.push(self.index.delete_row(txn_id, self.schema().key(row)?));
        for index in self.auxiliary.values() {
            deletes.push(index.delete_row(txn_id, index.schema().key(row)?));
        }
        try_join_all(deletes).await?;

        Ok(())
    }

    async fn upsert(self: Arc<Self>, txn_id: TxnId, row: Row) -> TCResult<()> {
        self.delete_row(&txn_id, &row).await?;

        let mut inserts = Vec::with_capacity(self.auxiliary.len() + 1);
        for index in self.auxiliary.values() {
            let columns = index.schema().column_names();
            let index_row: Row = row
                .clone()
                .into_iter()
                .filter(|(c, _)| columns.contains(c))
                .collect();
            inserts.push(index.insert(&txn_id, index_row));
        }
        inserts.push(self.index.insert(&txn_id, row));

        try_join_all(inserts).await?;
        Ok(())
    }
}

#[async_trait]
impl Selection for IndexTable {
    type Stream = <Index as Selection>::Stream;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.index.clone().count(txn_id).await
    }

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()> {
        let mut deletes = Vec::with_capacity(self.auxiliary.len() + 1);
        for index in self.auxiliary.values() {
            deletes.push(index.clone().delete(txn_id.clone()));
        }
        deletes.push(self.index.clone().delete(txn_id));

        try_join_all(deletes).await?;
        Ok(())
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.index.schema()
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.index.clone().stream(txn_id).await
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()> {
        if self.index.validate(bounds).is_ok() {
            return Ok(());
        }

        for index in self.auxiliary.values() {
            if index.validate(bounds).is_ok() {
                return Ok(());
            }
        }

        Err(error::bad_request(
            "This Table has no index which supports these bounds",
            bounds,
        ))
    }

    async fn update(self: Arc<Self>, txn: Arc<Txn>, value: Row) -> TCResult<()> {
        let schema = self.schema().clone();
        schema.validate_row_partial(&value)?;

        let txn_id = txn.id().clone();
        self.clone()
            .index(txn, None)
            .await?
            .stream(txn_id.clone())
            .await?
            .map(|row| schema.into_row(row))
            .map_ok(move |row| self.clone().upsert(txn_id.clone(), row))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }
}
