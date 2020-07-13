use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::state::btree::{self, BTree, BTreeRange, Key};
use crate::transaction::TxnId;
use crate::value::class::Impl;
use crate::value::{TCResult, TCStream, Value};

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
    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64> {
        self.len(txn_id).await
    }

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()> {
        self.btree.delete(txn_id, btree::Selector::all()).await
    }

    fn schema(&'_ self) -> &'_ Schema {
        &self.schema
    }

    fn validate(&self, bounds: &Bounds) -> TCResult<()> {
        self.schema.validate(bounds)?;

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
    source: Index,
}
