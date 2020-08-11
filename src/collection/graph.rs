use std::collections::HashMap;
use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, join_all, try_join};
use futures::stream::{FuturesOrdered, StreamExt, TryStreamExt};
use futures::try_join;

use crate::class::{TCResult, TCTryStream};
use crate::error;
use crate::transaction::lock::{Mutable, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::number::class::NumberType;
use crate::value::number::instance::{Number, UInt};
use crate::value::{Value, ValueId};

use super::table::{self, Selection, TableBase};
use super::tensor::{
    self, einsum, AxisBounds, SparseTensor, TensorBoolean, TensorIO, TensorTransform,
};

const ERR_CORRUPT: &str = "Graph corrupted! Please file a bug report.";

pub struct Graph {
    edges: TableBase,
    max_id: TxnLock<Mutable<u64>>,
    nodes: HashMap<ValueId, TableBase>,
    node_ids: TableBase,
}

impl Graph {
    pub async fn create(
        _txn: Arc<Txn>,
        _schema: HashMap<ValueId, table::Schema>,
    ) -> TCResult<Graph> {
        Err(error::not_implemented())
    }

    async fn get_matrix(&self, txn_id: &TxnId) -> TCResult<SparseTensor> {
        let max_id = self.max_id.read(txn_id).await?;
        let shape: tensor::Shape = vec![*max_id, *max_id].into();
        SparseTensor::try_from_table(self.edges.clone(), shape)
    }

    async fn get_node_id(
        &self,
        _txn_id: &TxnId,
        _node_type: ValueId,
        _node_key: Vec<Value>,
    ) -> TCResult<Option<u64>> {
        Err(error::not_implemented())
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
                    .insert(txn_id, vec![u64_value(*max_id)], vec![Value::Tuple(key)]);
            try_join(node_insert, node_id_insert).await?;
            *max_id += 1;
            Ok(())
        } else {
            Err(error::bad_request(
                "This Graph has no such node type",
                node_type,
            ))
        }
    }

    pub async fn add_edge(&self, txn_id: TxnId, node_from: u64, node_to: u64) -> TCResult<()> {
        let edges = self.get_matrix(&txn_id).await?;
        edges
            .write_value_at(txn_id, vec![node_from, node_to], true.into())
            .await
    }

    pub async fn bft(
        &self,
        txn: Arc<Txn>,
        node_type: ValueId,
        node_key: Vec<Value>,
    ) -> TCResult<TCTryStream<Vec<Value>>> {
        let start_node = self
            .get_node_id(txn.id(), node_type, node_key.to_vec())
            .await?
            .ok_or(error::not_found(Value::Tuple(node_key)))?;

        let edges = self.get_matrix(txn.id());
        let max_id = self.max_id.read(txn.id());
        let (edges, max_id) = try_join!(edges, max_id)?;

        let visited = SparseTensor::create(txn.clone(), vec![*max_id].into(), NumberType::Bool);
        let adjacent = SparseTensor::create(txn.clone(), vec![*max_id].into(), NumberType::Bool);
        let (mut visited, mut adjacent) = try_join!(visited, adjacent)?;
        adjacent
            .write_value_at(txn.id().clone(), vec![start_node], true.into())
            .await?;

        // TODO: stream the search itself instead of buffering these futures
        let mut found = FuturesOrdered::new();

        while adjacent.any(txn.clone()).await? {
            visited = visited.or(&adjacent)?;
            adjacent = einsum("ji,j->i", vec![edges.clone(), adjacent])?.and(&visited.not()?)?;

            let txn_id = txn.id().clone();
            let txn_id_clone = txn_id.clone();
            let nodes = self.nodes.clone();
            let node_ids = self.node_ids.clone();

            let adjacent_nodes = adjacent
                .clone()
                .filled(txn.clone())
                .await?
                .and_then(move |(id, _)| {
                    node_ids
                        .clone()
                        .get_owned(txn_id.clone(), vec![u64_value(id[0])])
                })
                .map(|r| match r {
                    Ok(Some(node_key)) if node_key.len() == 3 => {
                        Ok((node_key[1].clone(), node_key[2].clone()))
                    }
                    _ => Err(error::internal(ERR_CORRUPT)),
                })
                .map(|r| {
                    r.and_then(|(node_type, node_key)| {
                        let node_type: ValueId = node_type.try_into()?;
                        let node_key: Vec<Value> = node_key.try_into()?;
                        Ok((node_type, node_key))
                    })
                })
                .and_then(move |(node_type, node_key)| {
                    let nodes = nodes.clone();
                    let txn_id = txn_id_clone.clone();

                    Box::pin(async move {
                        let table = nodes
                            .get(&node_type)
                            .ok_or_else(|| error::internal(ERR_CORRUPT))?;

                        table
                            .clone()
                            .get_owned(txn_id, node_key)
                            .await?
                            .ok_or_else(|| error::internal(ERR_CORRUPT))
                    })
                });

            found.push(future::ready(adjacent_nodes));
        }

        let found: TCTryStream<Vec<Value>> = Box::pin(found.flatten());
        Ok(found)
    }

    pub async fn remove_edge(&self, txn_id: TxnId, node_from: u64, node_to: u64) -> TCResult<()> {
        let edges = self.get_matrix(&txn_id).await?;
        edges
            .write_value_at(txn_id, vec![node_from, node_to], false.into())
            .await
    }

    pub async fn remove_node(
        &self,
        txn: Arc<Txn>,
        node_type: ValueId,
        node_key: Vec<Value>,
    ) -> TCResult<()> {
        let node_id = self
            .get_node_id(txn.id(), node_type.clone(), node_key.to_vec())
            .await?
            .ok_or_else(|| error::not_found(Value::Tuple(node_key.clone())))?;

        let edges = self.get_matrix(txn.id()).await?;
        let max_id = self.max_id.read(txn.id()).await?;
        if edges
            .slice(vec![AxisBounds::all(*max_id), AxisBounds::At(node_id)].into())?
            .any(txn.clone())
            .await?
        {
            let node_key: Vec<String> = node_key.iter().map(|v| v.to_string()).collect();
            return Err(error::bad_request(
                "Tried to remove a graph node that still has edges",
                format!("[{}]", node_key.join(", ")),
            ));
        }

        let table = self.nodes.get(&node_type).unwrap();
        let row = table.schema().values_into_row(node_key)?;
        table.delete_row(txn.id(), row).await
    }
}

#[async_trait]
impl Transact for Graph {
    async fn commit(&self, txn_id: &TxnId) {
        let commits = self
            .nodes
            .values()
            .map(|t| t.commit(txn_id))
            .chain(iter::once(self.edges.commit(txn_id)))
            .chain(iter::once(self.node_ids.commit(txn_id)));

        join_all(commits).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        let commits = self
            .nodes
            .values()
            .map(|t| t.commit(txn_id))
            .chain(iter::once(self.edges.commit(txn_id)))
            .chain(iter::once(self.node_ids.commit(txn_id)));

        join_all(commits).await;
    }
}

fn u64_value(value: u64) -> Value {
    Value::Number(Number::UInt(UInt::U64(value)))
}
