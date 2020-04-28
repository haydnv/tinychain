use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::file::*;
use crate::internal::FsDir;
use crate::state::{Collection, Persistent};
use crate::transaction::{Transaction, TransactionId};
use crate::value::{TCResult, TCValue};

pub struct GraphConfig;

impl TryFrom<TCValue> for GraphConfig {
    type Error = error::TCError;

    fn try_from(_value: TCValue) -> TCResult<GraphConfig> {
        Err(error::not_implemented())
    }
}

#[derive(Debug)]
pub struct Graph {}

#[async_trait]
impl Collection for Graph {
    type Key = TCValue;
    type Value = TCValue;
    async fn get(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        _node_id: &TCValue,
    ) -> TCResult<Self::Value> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: Arc<Transaction>,
        _node_id: TCValue,
        _node: TCValue,
    ) -> TCResult<Arc<Self>> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl File for Graph {
    async fn copy_from(_reader: &mut FileReader, _dest: Arc<FsDir>) -> Arc<Self> {
        Arc::new(Graph {})
    }

    async fn copy_to(&self, _txn_id: TransactionId, _writer: &mut FileWriter) {
        // TODO
    }
}

#[async_trait]
impl Persistent for Graph {
    type Config = GraphConfig;

    async fn create(_txn: Arc<Transaction>, _config: GraphConfig) -> TCResult<Arc<Graph>> {
        Err(error::not_implemented())
    }

    async fn commit(&self, _txn_id: &TransactionId) {
        // TODO
    }
}
