use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::state::btree::{self, BTree, BTreeRange, Key};
use crate::transaction::{Txn, TxnId};
use crate::value::class::Impl;
use crate::value::{TCResult, TCStream, Value, ValueId};

use super::{Bounds, Row, Schema, Selection};

struct Index {
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
}

#[async_trait]
impl Selection for Index {
    type Stream = TCStream<Vec<Value>>;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.len(txn_id).await
    }

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()> {
        self.btree.delete(txn_id, btree::Selector::all()).await
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

    async fn update(self: Arc<Self>, txn_id: TxnId, row: Row) -> TCResult<()> {
        let value = self.key_from_row(row)?;
        self.btree
            .update(&txn_id, &btree::Selector::all(), &value)
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

    async fn update(self: Arc<Self>, _txn_id: TxnId, _value: Row) -> TCResult<()> {
        Err(error::method_not_allowed(
            "this is a transitive (read-only) index",
        ))
    }
}
