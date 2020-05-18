use std::convert::TryFrom;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::chain::{ChainBlock, Mutation};
use crate::internal::file::*;
use crate::object::actor::Token;
use crate::state::{Collection, Persistent, Transact};
use crate::transaction::{Txn, TxnId};
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
        _txn: Arc<Txn>,
        _node_id: &TCValue,
        _auth: &Option<Token>,
    ) -> TCResult<Self::Value> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: Arc<Txn>,
        _node_id: TCValue,
        _node: TCValue,
        _auth: &Option<Token>,
    ) -> TCResult<Arc<Self>> {
        Err(error::not_implemented())
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct GraphMutation;

impl Mutation for GraphMutation {}

#[async_trait]
impl File for Graph {
    type Block = ChainBlock<GraphMutation>;

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

    async fn create(_txn: Arc<Txn>, _config: GraphConfig) -> TCResult<Arc<Graph>> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Transact for Graph {
    async fn commit(&self, _txn_id: &TxnId) {
        // TODO
    }
}
