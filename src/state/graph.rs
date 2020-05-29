use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::block::Store;
use crate::internal::file::*;
use crate::state::*;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCValue};

pub struct GraphConfig;

impl TryFrom<Args> for GraphConfig {
    type Error = error::TCError;

    fn try_from(_args: Args) -> TCResult<GraphConfig> {
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
        _txn: &Arc<Txn<'_>>,
        _node_id: &TCValue,
    ) -> TCResult<Self::Value> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _node_id: TCValue,
        _node: TCValue,
    ) -> TCResult<State> {
        Ok(self.into())
    }
}

#[async_trait]
impl File for Graph {
    async fn copy_from(_reader: &mut FileCopier, _txn_id: &TxnId, _dest: Arc<Store>) -> Arc<Self> {
        // TODO
        Arc::new(Graph {})
    }

    async fn copy_into(&self, _txn_id: TxnId, _writer: &mut FileCopier) {
        // TODO
    }

    async fn from_store(_txn_id: &TxnId, _store: Arc<Store>) -> Arc<Graph> {
        // TODO
        Arc::new(Graph {})
    }
}

#[async_trait]
impl Persistent for Graph {
    type Config = GraphConfig;

    async fn create(_txn: &Arc<Txn<'_>>, _config: GraphConfig) -> TCResult<Arc<Graph>> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Transact for Graph {
    async fn commit(&self, _txn_id: &TxnId) {
        // TODO
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // TODO
    }
}
