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
use crate::value::{Link, TCResult, TCValue};

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
    txn_cache: TransactionCache<Link, (EntryType, EntryState)>,
}

#[async_trait]
impl Collection for Directory {
    type Key = Link;
    type Value = State;

    async fn get(
        self: &Arc<Self>,
        txn: Arc<Transaction>,
        path: &Self::Key,
    ) -> TCResult<Self::Value> {
        if path.is_empty() {
            return Err(error::bad_request(
                "You must specify a path to look up in a directory",
                path,
            ));
        } else if let Some((_, state)) = self.txn_cache.get(&txn.id(), &path.nth(0)) {
            return match state {
                EntryState::Directory(_, dir) => dir.get(txn, &path.slice_from(1)).await,
                EntryState::Graph(_, graph) => Ok(graph.into()),
                EntryState::Table(_, table) => Ok(table.into()),
            };
        }

        let entry = self
            .chain
            .stream_into_until(txn.id())
            .filter_map(|entries: Vec<(Link, EntryType)>| async move {
                let entries: Vec<(Link, EntryType)> =
                    entries.iter().filter(|(p, _)| p == path).cloned().collect();
                if entries.is_empty() {
                    None
                } else {
                    Some(entries)
                }
            })
            .fold(None, |_, entries: Vec<(Link, EntryType)>| async move {
                entries.last().cloned()
            })
            .await;

        if let Some((_, entry_type)) = entry {
            if let Some(store) = self.context.get(&path.nth(0)) {
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
        if path.len() != 1 {
            return Err(error::not_found(path));
        }

        let context = self.context.create(&path)?;
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
        let chain = Chain::copy_from(blocks, dest.create(&path).unwrap()).await;

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
        Ok(Arc::new(Directory {
            context: txn.context(),
            chain: Chain::new(txn.context().create(&Link::to("/.contents")?)?),
            txn_cache: TransactionCache::new(),
        }))
    }
}

#[async_trait]
impl Transactable for Directory {
    async fn commit(&self, txn_id: &TransactionId) {
        let mutations = self.txn_cache.close(txn_id);
        let entries: Vec<(&Link, &EntryType)> = mutations
            .iter()
            .map(|(path, (entry_type, _state))| (path, entry_type))
            .collect();
        let tasks = mutations
            .iter()
            .map(|(_path, (_entry_type, state))| async move {
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

        join(join_all(tasks), self.chain.clone().put(txn_id, &entries)).await;
    }
}
