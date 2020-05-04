use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join, join_all};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::cache::TransactionCache;
use crate::internal::file::*;
use crate::internal::Chain;
use crate::state::*;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{PathSegment, TCPath, TCResult, TCValue};

#[derive(Clone, Deserialize, Serialize)]
enum EntryType {
    Directory,
    Table,
    Graph,
}

#[derive(Clone)]
enum EntryState {
    Directory(Arc<Store>, Arc<Directory>),
    Table(Arc<Store>, Arc<Table>),
    Graph(Arc<Store>, Arc<Graph>),
}

pub struct Directory {
    context: Arc<Store>,
    chain: Arc<Chain>,
    txn_cache: TransactionCache<PathSegment, (EntryType, EntryState)>,
}

impl Directory {
    pub fn new(context: Arc<Store>) -> TCResult<Arc<Directory>> {
        Ok(Arc::new(Directory {
            context: context.clone(),
            chain: Chain::new(context.create("/contents")?),
            txn_cache: TransactionCache::new(),
        }))
    }
}

#[async_trait]
impl Collection for Directory {
    type Key = TCPath;
    type Value = State;

    async fn get(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        path: &Self::Key,
    ) -> TCResult<Self::Value> {
        println!("Directory::get {}", path);
        if path.is_empty() {
            return Err(error::bad_request(
                "You must specify a path to look up in a directory",
                path,
            ));
        } else if let Some((_, state)) = self.txn_cache.get(&txn.id(), &path[0]) {
            return match state {
                EntryState::Directory(_, dir) => dir.get(txn, &path.slice_from(1)).await,
                EntryState::Graph(_, graph) => Ok(graph.into()),
                EntryState::Table(_, table) => Ok(table.into()),
            };
        }

        let entry = self
            .chain
            .stream_into_until(txn.id())
            .filter_map(|entries: Vec<(PathSegment, EntryType)>| async move {
                let entries: Vec<(PathSegment, EntryType)> =
                    entries.iter().filter(|(p, _)| path == p).cloned().collect();
                if entries.is_empty() {
                    None
                } else {
                    Some(entries)
                }
            })
            .fold(
                None,
                |_, entries: Vec<(PathSegment, EntryType)>| async move { entries.last().cloned() },
            )
            .await;

        if let Some((_, entry_type)) = entry {
            if let Some(store) = self.context.get(&path[0].clone().into()) {
                match entry_type {
                    EntryType::Directory => {
                        let dir = Directory::from_store(store).await;
                        dir.get(txn, &path.slice_from(1)).await
                    }
                    EntryType::Graph => Ok(Graph::from_store(store).await.into()),
                    EntryType::Table => Ok(Table::from_store(store).await.into()),
                }
            } else {
                Err(error::internal(format!(
                    "Directory entry {} has no associated data",
                    path
                )))
            }
        } else {
            Err(error::not_found(path))
        }
    }

    async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Self::Key,
        state: Self::Value,
    ) -> TCResult<Arc<Self>> {
        println!("Directory::put {}", path);
        let path: PathSegment = path.try_into()?;

        let context = self.context.create(path.clone())?;
        let entry = match state {
            State::Graph(g) => (EntryType::Graph, EntryState::Graph(context, g)),
            State::Table(t) => (EntryType::Table, EntryState::Table(context, t)),
            State::Value(v) => {
                return Err(error::bad_request("Expected a persistent state, found", v))
            }
        };

        self.txn_cache.insert(txn.id(), path, entry);
        txn.mutate(self.clone());
        Ok(self)
    }
}

#[async_trait]
impl File for Directory {
    async fn copy_from(reader: &mut FileCopier, dest: Arc<Store>) -> Arc<Directory> {
        let (path, blocks) = reader.next().await.unwrap();
        let chain = Chain::copy_from(blocks, dest.create(path).unwrap()).await;

        Arc::new(Directory {
            context: dest,
            chain,
            txn_cache: TransactionCache::new(),
        })
    }

    async fn copy_into(&self, _txn_id: TransactionId, _writer: &mut FileCopier) {
        // TODO
    }

    async fn from_store(_store: Arc<Store>) -> Arc<Directory> {
        panic!("Directory::from_store is not implemented")
    }
}

#[async_trait]
impl Persistent for Directory {
    type Config = TCValue; // TODO: permissions

    async fn create(txn: Arc<Transaction>, _: TCValue) -> TCResult<Arc<Directory>> {
        Directory::new(txn.context())
    }
}

#[async_trait]
impl Transactable for Directory {
    async fn commit(&self, txn_id: &TransactionId) {
        let mutations = self.txn_cache.close(txn_id);
        let mut entries = Vec::with_capacity(mutations.len());
        let mut tasks = Vec::with_capacity(mutations.len());
        for (path, (entry_type, state)) in mutations.iter() {
            entries.push((path, entry_type));
            tasks.push(async move {
                match state {
                    EntryState::Directory(context, dir) => {
                        FileCopier::copy(txn_id.clone(), &*dir.clone(), context.clone()).await;
                    }
                    EntryState::Graph(context, graph) => {
                        FileCopier::copy(txn_id.clone(), &*graph.clone(), context.clone()).await;
                    }
                    EntryState::Table(context, table) => {
                        FileCopier::copy(txn_id.clone(), &*table.clone(), context.clone()).await;
                    }
                }
            });
        }

        join(join_all(tasks), self.chain.clone().put(txn_id, &entries)).await;
    }
}
