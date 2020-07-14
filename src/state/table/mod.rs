use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;

use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, Value, ValueId};

mod base;
mod index;
mod view;

pub type Bounds = base::Bounds;
pub type Row = base::Row;
pub type Schema = base::Schema;

#[async_trait]
pub trait Selection: Sized + Send + Sync {
    type Stream: Stream<Item = Vec<Value>> + Send + Sync;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64>;

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()>;

    fn derive<M: Fn(Row) -> Value>(
        self: Arc<Self>,
        name: ValueId,
        map: M,
    ) -> Arc<view::Derived<Self, M>> {
        Arc::new((self, name, map).into())
    }

    fn filter<F: Fn(Row) -> bool>(self: Arc<Self>, filter: F) -> Arc<view::Filtered<Self, F>> {
        Arc::new((self, filter).into())
    }

    fn group_by(self: Arc<Self>, columns: Vec<ValueId>) -> Arc<view::Aggregate<Self>> {
        Arc::new((self, columns).into())
    }

    async fn index(
        self: Arc<Self>,
        txn: Arc<Txn>,
        columns: Option<Vec<ValueId>>,
    ) -> TCResult<Arc<index::ReadOnly>> {
        index::ReadOnly::copy_from(self, txn, columns)
            .await
            .map(Arc::new)
    }

    fn limit(self: Arc<Self>, limit: u64) -> Arc<view::Limit<Self>> {
        Arc::new((self, limit).into())
    }

    fn order_by(
        self: Arc<Self>,
        columns: Vec<ValueId>,
        reverse: bool,
    ) -> TCResult<Arc<view::Sorted<Self>>> {
        let sorted = (self, columns, reverse).try_into()?;
        Ok(Arc::new(sorted))
    }

    fn select(
        self: Arc<Self>,
        columns: Vec<ValueId>,
    ) -> TCResult<Arc<view::ColumnSelection<Self>>> {
        let selection = (self, columns).try_into()?;
        Ok(Arc::new(selection))
    }

    fn schema(&'_ self) -> &'_ Schema;

    fn slice(self: Arc<Self>, bounds: Bounds) -> TCResult<Arc<view::Slice<Self>>> {
        let slice = (self, bounds).try_into()?;
        Ok(Arc::new(slice))
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream>;

    fn validate(&self, bounds: &Bounds) -> TCResult<()>;

    async fn update(self: Arc<Self>, txn_id: TxnId, value: Row) -> TCResult<()>;
}
