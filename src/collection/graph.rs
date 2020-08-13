use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;
use futures::try_join;

use crate::class::TCResult;
use crate::error;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::{label, Label, Value, ValueId};

use super::schema::GraphSchema;
use super::table::TableBase;
use super::tensor::{self, SparseTensor};

const ERR_CORRUPT: &str = "Graph corrupted! Please file a bug report.";

const NODE_FROM: Label = label("node_from");
const NODE_ID: Label = label("node_id");
const NODE_KEY: Label = label("node_key");
const NODE_TO: Label = label("node_to");
const NODE_TYPE: Label = label("node_type");

pub struct Graph {
    node_ids: TableBase,
    nodes: HashMap<ValueId, TableBase>,
    edges: HashMap<ValueId, TableBase>,
    max_id: TxnLock<Mutable<u64>>,
}

impl Graph {
    pub async fn create(_txn: Arc<Txn>, _schema: GraphSchema) -> TCResult<Graph> {
        Err(error::not_implemented())
    }

    async fn get_matrix(&self, label: &ValueId, txn_id: &TxnId) -> TCResult<SparseTensor> {
        if let Some(edges) = self.edges.get(label) {
            let max_id = self.max_id.read(txn_id).await?;
            let shape: tensor::Shape = vec![*max_id, *max_id].into();
            SparseTensor::try_from_table(edges.clone(), shape)
        } else {
            Err(error::bad_request("Graph has no such edge label", label))
        }
    }

    async fn get_node_id(
        &self,
        txn_id: TxnId,
        node_type: ValueId,
        node_key: Vec<Value>,
    ) -> TCResult<Option<u64>> {
        match self
            .node_ids
            .get(txn_id, vec![Value::from(node_type), Value::from(node_key)])
            .await?
        {
            Some(row) if row.len() == 2 => Ok(Some(row[1].clone().try_into()?)),
            None => Ok(None),
            _ => Err(error::internal(ERR_CORRUPT)),
        }
    }

    pub async fn add_node(
        &self,
        txn_id: TxnId,
        node_type: ValueId,
        key: Vec<Value>,
        value: Vec<Value>,
    ) -> TCResult<()> {
        let mut max_id = self.max_id.write(txn_id.clone()).await?;

        if let Some(table) = self.nodes.get(&node_type) {
            let node_insert = table.insert(txn_id.clone(), key.to_vec(), value);
            let node_id_insert =
                self.node_ids
                    .insert(txn_id, vec![Value::Tuple(key)], vec![Value::from(*max_id)]);
            try_join!(node_insert, node_id_insert)?;
            *max_id += 1;
            Ok(())
        } else {
            Err(error::bad_request(
                "This Graph has no such node type",
                node_type,
            ))
        }
    }

    pub async fn add_edge(
        &self,
        _txn_id: TxnId,
        _label: ValueId,
        _node_from: Vec<Value>,
        _node_to: Vec<Value>,
    ) -> TCResult<()> {
        Err(error::not_implemented())
    }

    pub async fn remove_edge(
        &self,
        _txn_id: &TxnId,
        _label: &ValueId,
        _node_from: &[Value],
        _node_to: &[Value],
    ) -> TCResult<()> {
        Err(error::not_implemented())
    }

    pub async fn remove_node(
        &self,
        _txn: Arc<Txn>,
        _node_type: ValueId,
        _node_key: Vec<Value>,
    ) -> TCResult<()> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Transact for Graph {
    async fn commit(&self, txn_id: &TxnId) {
        let commits = self
            .nodes
            .values()
            .map(|t| t.commit(txn_id))
            .chain(self.edges.values().map(|t| t.commit(txn_id)))
            .chain(iter::once(self.node_ids.commit(txn_id)));

        join_all(commits).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let rollbacks = self
            .nodes
            .values()
            .map(|t| t.rollback(txn_id))
            .chain(self.edges.values().map(|t| t.rollback(txn_id)))
            .chain(iter::once(self.node_ids.rollback(txn_id)));

        join_all(rollbacks).await;
    }
}
