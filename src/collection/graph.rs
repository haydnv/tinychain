use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, join_all, try_join_all, TryFutureExt};
use futures::stream::{FuturesOrdered, StreamExt, TryStreamExt};
use futures::try_join;

use crate::class::{TCResult, TCTryStream};
use crate::error;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::number::class::{NumberType, UIntType};
use crate::value::number::instance::{Number, UInt};
use crate::value::{label, Label, Value, ValueId, ValueType};

use super::schema::{GraphSchema, IndexSchema, TableSchema};
use super::table::TableBase;
use super::tensor::{self, einsum, SparseTable, SparseTensor, TensorBoolean, TensorIO};

const ERR_CORRUPT: &str = "Graph corrupted! Please file a bug report.";

const NODE_FROM: Label = label("node_from");
const NODE_ID: Label = label("node_id");
const NODE_KEY: Label = label("node_key");
const NODE_LABEL: Label = label("node_label");
const NODE_TO: Label = label("node_to");
const NODE_TYPE: Label = label("node_type");

pub struct Node {
    graph: Arc<Graph>,
    id: u64,
    edges: HashMap<ValueId, TCTryStream<Node>>,
}

pub struct Graph {
    nodes: HashMap<ValueId, TableBase>,
    edges: HashMap<ValueId, TableBase>,
    node_indices: TableBase,
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

        let edges = try_join_all(schema.edges().iter().map(|(name, dtype)| {
            SparseTable::create_table(txn.clone(), 2, *dtype)
                .map_ok(move |table| (name.clone(), table))
        }))
        .await?
        .into_iter()
        .collect();

        let max_id = TxnLock::new(txn.id().clone(), 0u64.into());

        let node_id_schema: IndexSchema = (
            vec![(NODE_KEY.into(), ValueType::Tuple).into()],
            vec![(NODE_ID.into(), u64_type).into()],
        )
            .into();
        let node_id_schema: TableSchema = node_id_schema.into();
        let node_indices = TableBase::create(txn.clone(), node_id_schema).await?;

        Ok(Graph {
            nodes,
            edges,
            node_indices,
            max_id,
        })
    }

    async fn get_matrix(&self, txn_id: &TxnId, label: &ValueId) -> TCResult<SparseTensor> {
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
            .node_indices
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
            let node_id_insert = self.node_indices.insert(
                txn_id,
                vec![Value::Tuple(key)],
                vec![Value::from(*max_id)],
            );
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

        let edges = self.get_matrix(&txn_id, &label);

        let (edges, node_from_id, node_to_id) = try_join!(edges, node_from_id, node_to_id)?;

        let node_from_id = try_unwrap_node_id(node_from_id, node_from_key)?;
        let node_to_id = try_unwrap_node_id(node_to_id, node_to_key)?;

        edges
            .write_value_at(txn_id, vec![node_from_id, node_to_id], true.into())
            .await
    }

    pub async fn bft(
        self: Arc<Self>,
        txn: Arc<Txn>,
        start_node: u64,
        relation: ValueId,
        limit: usize,
    ) -> TCResult<TCTryStream<u64>> {
        let edges = self.get_matrix(txn.id(), &relation);
        let max_id = self.max_id.read(txn.id());
        let (edges, max_id) = try_join!(edges, max_id)?;

        let visited = SparseTensor::create(txn.clone(), vec![*max_id].into(), NumberType::Bool);
        let adjacent = SparseTensor::create(txn.clone(), vec![*max_id].into(), NumberType::Bool);
        let (mut visited, mut adjacent) = try_join!(visited, adjacent)?;
        adjacent
            .write_value_at(txn.id().clone(), vec![start_node], true.into())
            .await?;

        let mut order = 0;
        // TODO: stream the search itself instead of buffering these futures
        let mut found = FuturesOrdered::new();

        while order < limit && adjacent.any(txn.clone()).await? {
            visited = visited.or(&adjacent)?;
            adjacent = einsum("ji,j->i", vec![edges.clone(), adjacent])?
                .copy(txn.subcontext_tmp().await?)
                .await?;

            adjacent.mask(&txn, visited.clone()).await?;
            let nodes = adjacent
                .clone()
                .filled(txn.clone())
                .await?
                .map_ok(|(id, _)| id[0]);

            found.push(future::ready(nodes));
            order += 1;
        }

        Ok(Box::pin(found.flatten()))
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
            .chain(iter::once(self.node_indices.commit(txn_id)));

        join_all(commits).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let rollbacks = self
            .nodes
            .values()
            .map(|t| t.rollback(txn_id))
            .chain(self.edges.values().map(|t| t.rollback(txn_id)))
            .chain(iter::once(self.node_indices.rollback(txn_id)));

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

fn unwrap_u64(row: Vec<Value>) -> u64 {
    if let &[Value::Number(Number::UInt(UInt::U64(u)))] = row.as_slice() {
        u
    } else {
        panic!(ERR_CORRUPT)
    }
}
