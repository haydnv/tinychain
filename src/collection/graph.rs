use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, try_join_all, TryFutureExt};
use futures::try_join;

use crate::class::TCResult;
use crate::error;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::number::class::{NumberType, UIntType};
use crate::value::{label, Label, Value, ValueId, ValueType};

use super::schema::{GraphSchema, IndexSchema, TableSchema};
use super::table::TableBase;
use super::tensor::{self, SparseTensor, TensorIO};

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
    pub async fn create(txn: Arc<Txn>, schema: GraphSchema) -> TCResult<Graph> {
        let nodes = try_join_all(schema.nodes().iter().map(|(name, schema)| {
            TableBase::create(txn.clone(), schema.clone())
                .map_ok(move |table| (name.clone(), table))
        }))
        .await?
        .into_iter()
        .collect();

        let u64_type = ValueType::Number(NumberType::UInt(UIntType::U64));
        let edge_schema: IndexSchema = (
            vec![(NODE_FROM.into(), u64_type).into()],
            vec![(NODE_TO.into(), u64_type).into()],
        )
            .into();
        let edge_schema: TableSchema = (
            edge_schema,
            iter::once((NODE_TO.into(), vec![NODE_TO.into()])),
        )
            .into();
        let edges = try_join_all(schema.edges().iter().map(|name| {
            TableBase::create(txn.clone(), edge_schema.clone())
                .map_ok(move |table| (name.clone(), table))
        }))
        .await?
        .into_iter()
        .collect();

        let max_id = TxnLock::new(txn.id().clone(), 0u64.into());

        let node_id_schema: IndexSchema = (
            vec![(NODE_ID.into(), u64_type).into()],
            vec![(NODE_KEY.into(), ValueType::Tuple).into()],
        )
            .into();
        let node_id_schema: TableSchema = node_id_schema.into();
        let node_ids = TableBase::create(txn, node_id_schema).await?;

        Ok(Graph {
            node_ids,
            nodes,
            edges,
            max_id,
        })
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
        txn_id: TxnId,
        label: ValueId,
        node_from: (ValueId, Vec<Value>),
        node_to: (ValueId, Vec<Value>),
    ) -> TCResult<()> {
        let node_from_key = Value::Tuple(node_from.1);
        let node_from_id = self
            .nodes
            .get(&node_from.0)
            .ok_or(error::not_found(&node_from.0))?
            .get(txn_id.clone(), vec![node_from_key.clone()]);

        let node_to_key = Value::Tuple(node_to.1);
        let node_to_id = self
            .nodes
            .get(&node_to.0)
            .ok_or(error::not_found(&node_to.0))?
            .get(txn_id.clone(), vec![node_to_key.clone()]);

        let edges = self.get_matrix(&label, &txn_id);

        let (edges, node_from_id, node_to_id) = try_join!(edges, node_from_id, node_to_id)?;

        let node_from_id = try_unwrap_node_id(node_from_id, node_from_key)?;
        let node_to_id = try_unwrap_node_id(node_to_id, node_to_key)?;

        edges
            .write_value_at(txn_id, vec![node_from_id, node_to_id], true.into())
            .await
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

fn try_unwrap_node_id(node_id: Option<Vec<Value>>, node_key: Value) -> TCResult<u64> {
    node_id
        .ok_or(error::not_found(node_key))?
        .pop()
        .ok_or(error::internal(ERR_CORRUPT))?
        .try_into()
}
