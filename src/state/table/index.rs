use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, try_join_all};
use futures::stream::{StreamExt, TryStreamExt};

use crate::error;
use crate::state::btree::{self, BTree, BTreeRange, Key};
use crate::state::dir::Dir;
use crate::transaction::lock::{Mutate, TxnLock};
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::view::Sliced;
use super::{Bounds, Column, Row, Schema, Selection};

pub struct Index {
    btree: Arc<BTree>,
    schema: Schema,
}

impl Index {
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

    async fn insert(&self, txn_id: &TxnId, row: Row, reject_extra_columns: bool) -> TCResult<()> {
        let key = self.schema().row_into_values(row, reject_extra_columns)?;
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

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        let key = self.schema.row_into_values(row, false)?;
        self.btree.delete(txn_id, btree::Selector::Key(key)).await
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

    async fn validate(&self, _txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
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
        let key: btree::Key = self.schema().row_into_values(row, false)?;
        self.btree
            .update(txn.id(), &btree::Selector::all(), &key)
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
    type Stream = <Index as Selection>::Stream;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.index.clone().count(txn_id).await
    }

    fn schema(&'_ self) -> &'_ Schema {
        self.index.schema()
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.index.clone().stream(txn_id).await
    }

    async fn validate(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
        self.index.validate(txn_id, bounds).await
    }
}

#[derive(Clone)]
struct Indices {
    dir: Arc<Dir>,
    primary: Arc<Index>,
    auxiliary: BTreeMap<ValueId, Arc<Index>>,
}

impl Indices {
    fn len(&self) -> usize {
        self.auxiliary.len() + 1
    }
}

#[async_trait]
impl Mutate for Indices {
    fn diverge(&self, _txn_id: &TxnId) -> Self {
        self.clone()
    }

    async fn converge(&mut self, mut new_value: Indices, txn_id: &TxnId) {
        let existing: HashSet<ValueId> = self.auxiliary.keys().cloned().collect();
        let new: HashSet<ValueId> = new_value.auxiliary.keys().cloned().collect();

        let dir = self.dir.clone();
        let delete_ops = existing
            .difference(&new)
            .map(move |name| dir.clone().delete_file(txn_id.clone(), name.clone()));
        for name in new.iter() {
            let index = new_value.auxiliary.remove(&name).unwrap();
            self.auxiliary.insert(name.clone(), index);
        }

        try_join_all(delete_ops).await.unwrap();
    }
}

pub struct IndexTable {
    indices: TxnLock<Indices>,
    schema: Schema,
}

impl IndexTable {
    pub async fn add_index(
        self: Arc<Self>,
        txn: Arc<Txn>,
        name: ValueId,
        key: Vec<ValueId>,
    ) -> TCResult<()> {
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

        let indices = self.indices.read(txn.id()).await?;

        let columns: HashMap<ValueId, Column> = self.schema().clone().into();
        let key: Vec<Column> = key
            .iter()
            .map(|c| {
                columns
                    .get(&c)
                    .cloned()
                    .ok_or(error::bad_request("No such column", c))
            })
            .collect::<TCResult<Vec<Column>>>()?;
        let values: Vec<Column> = self
            .schema()
            .key_columns()
            .iter()
            .filter(|c| !index_key_set.contains(&c.name))
            .cloned()
            .collect();
        let schema: Schema = (key, values).into();
        let btree_file = indices
            .dir
            .create_file(txn.id().clone(), name.clone())
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

        if let Ok(mut indices) = indices.upgrade().await {
            if indices.auxiliary.contains_key(&name) {
                indices
                    .dir
                    .clone()
                    .delete_file(txn.id().clone(), name.clone())
                    .await?;
                Err(error::bad_request(
                    "This table already has an index named",
                    name,
                ))
            } else {
                indices.auxiliary.insert(name, Arc::new(index));
                Ok(())
            }
        } else {
            self.indices
                .read(txn.id())
                .await?
                .dir
                .clone()
                .delete_file(txn.id().clone(), name)
                .await?;
            Err(error::conflict())
        }
    }

    pub fn slice(self: Arc<Self>, bounds: Bounds) -> TCResult<Sliced> {
        Sliced::new(self, bounds)
    }

    pub async fn stream_slice(
        self: Arc<Self>,
        _txn_id: TxnId,
        _bounds: Bounds,
    ) -> TCResult<TCStream<Vec<Value>>> {
        Err(error::not_implemented())
    }

    async fn upsert(self: Arc<Self>, txn_id: TxnId, row: Row) -> TCResult<()> {
        self.delete_row(&txn_id, row.clone()).await?;

        let indices = self.indices.read(&txn_id).await?;

        let mut inserts = Vec::with_capacity(indices.len());
        for index in indices.auxiliary.values() {
            inserts.push(index.insert(&txn_id, row.clone(), false));
        }
        inserts.push(indices.primary.insert(&txn_id, row, true));

        try_join_all(inserts).await?;
        Ok(())
    }
}

#[async_trait]
impl Selection for IndexTable {
    type Stream = <Index as Selection>::Stream;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.indices
            .read(&txn_id)
            .await?
            .primary
            .clone()
            .count(txn_id)
            .await
    }

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()> {
        let indices = self.indices.read(&txn_id).await?;
        let mut deletes = Vec::with_capacity(indices.len());
        for index in indices.auxiliary.values() {
            deletes.push(index.clone().delete(txn_id.clone()));
        }
        deletes.push(indices.primary.clone().delete(txn_id));

        try_join_all(deletes).await?;
        Ok(())
    }

    async fn delete_row(&self, txn_id: &TxnId, row: Row) -> TCResult<()> {
        self.schema().validate_row(&row)?;

        let indices = self.indices.read(txn_id).await?;
        let mut deletes = Vec::with_capacity(indices.len());
        for index in indices.auxiliary.values() {
            deletes.push(index.delete_row(txn_id, row.clone()));
        }
        deletes.push(indices.primary.delete_row(txn_id, row));
        try_join_all(deletes).await?;

        Ok(())
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream> {
        self.indices
            .read(&txn_id)
            .await?
            .primary
            .clone()
            .stream(txn_id)
            .await
    }

    async fn validate(&self, txn_id: &TxnId, bounds: &Bounds) -> TCResult<()> {
        let indices = self.indices.read(txn_id).await?;

        if indices.primary.validate(txn_id, bounds).await.is_ok() {
            return Ok(());
        }

        for index in indices.auxiliary.values() {
            if index.validate(txn_id, bounds).await.is_ok() {
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
            .map(|row| schema.values_into_row(row))
            .map_ok(move |row| self.clone().upsert(txn_id.clone(), row))
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }
}
