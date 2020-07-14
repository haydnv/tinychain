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
pub type Column = base::Column;
pub type Row = base::Row;
pub type Schema = base::Schema;

#[async_trait]
pub trait Selection: Sized + Send + Sync + 'static {
    type Stream: Stream<Item = Vec<Value>> + Send + Sync + Unpin;

    async fn count(self: Arc<Self>, txn_id: TxnId) -> TCResult<u64>;

    async fn delete(self: Arc<Self>, txn_id: TxnId) -> TCResult<()>;

    async fn index(
        self: Arc<Self>,
        txn: Arc<Txn>,
        columns: Option<Vec<ValueId>>,
    ) -> TCResult<Arc<index::ReadOnly>> {
        index::ReadOnly::copy_from(self, txn, columns)
            .await
            .map(Arc::new)
    }

    fn limit(self: Arc<Self>, limit: u64) -> Arc<view::Limited<Self>> {
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

    fn slice(self: Arc<Self>, bounds: Bounds) -> TCResult<Arc<view::Sliced<Self>>> {
        let slice = (self, bounds).try_into()?;
        Ok(Arc::new(slice))
    }

    async fn stream(self: Arc<Self>, txn_id: TxnId) -> TCResult<Self::Stream>;

    fn validate(&self, bounds: &Bounds) -> TCResult<()>;

    async fn update(self: Arc<Self>, txn: Arc<Txn>, value: Row) -> TCResult<()>;
}
