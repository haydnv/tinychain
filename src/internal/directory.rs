use std::collections::HashMap;
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
enum Entry {
    Directory(Link),
    Table(Link),
    Graph(Link),
}

impl Entry {
    fn path(&'_ self) -> &'_ Link {
        match self {
            Entry::Directory(p) => p,
            Entry::Table(p) => p,
            Entry::Graph(p) => p,
        }
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Entry::Directory(p1), Entry::Directory(p2)) => p1 == p2,
            (Entry::Table(p1), Entry::Table(p2)) => p1 == p2,
            (Entry::Graph(p1), Entry::Graph(p2)) => p1 == p2,
            _ => false,
        }
    }
}

impl Eq for Entry {}

impl std::hash::Hash for Entry {
    fn hash<T: std::hash::Hasher>(&self, h: &mut T) {
        self.path().hash(h)
    }
}

enum EntryState {
    Directory(Arc<Store>, Arc<Directory>),
    Table(Arc<Store>, Arc<Table>),
    Graph(Arc<Store>, Arc<Graph>),
}

pub struct Directory {
    context: Arc<Store>,
    chain: Arc<Chain>,
    txn_cache: TransactionCache<Entry, EntryState>,
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
        }

        let entry = self
            .chain
            .stream_into_until(txn.id())
            .filter_map(|entries: Vec<Entry>| async move {
                let entries: Vec<Entry> = entries
                    .iter()
                    .filter(|e| e.path() == path)
                    .cloned()
                    .collect();
                if entries.is_empty() {
                    None
                } else {
                    Some(entries)
                }
            })
            .fold(None, |_, entries: Vec<Entry>| async move {
                entries.last().cloned()
            })
            .await;

        match entry {
            Some(Entry::Directory(name)) => {
                let dir = Directory::from_store(self.context.reserve(&name)?).await;
                dir.get(txn, &path.slice_from(1)).await
            }
            Some(Entry::Graph(name)) => {
                Ok(Graph::from_store(self.context.reserve(&name)?).await.into())
            }
            Some(Entry::Table(name)) => {
                Ok(Table::from_store(self.context.reserve(&name)?).await.into())
            }
            None => Err(error::not_found(path)),
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

        let context = self.context.reserve(&path)?;
        let entry = match state {
            State::Graph(g) => (Entry::Graph(path), EntryState::Graph(context, g)),
            State::Table(t) => (Entry::Table(path), EntryState::Table(context, t)),
            State::Value(v) => {
                return Err(error::bad_request("Expected a persistent state, found", v))
            }
        };

        self.txn_cache.insert(txn.id(), entry.0, entry.1);
        Ok(self)
    }
}

#[async_trait]
impl File for Directory {
    async fn copy_from(reader: &mut FileCopier, dest: Arc<Store>) -> Arc<Directory> {
        let (path, blocks) = reader.next().await.unwrap();
        let chain = Chain::copy_from(blocks, dest.reserve(&path).unwrap()).await;

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

    async fn commit(&self, txn_id: &TransactionId) {
        let mutations = self.txn_cache.close(&txn_id);
        let entries: Vec<&Entry> = mutations.keys().collect();
        let tasks = mutations.values().map(|state| async move {
            match state {
                EntryState::Directory(context, dir) => {
                    FileCopier::copy(txn_id.clone(), dir.clone(), context.clone()).await;
                }
                EntryState::Graph(context, graph) => {
                    FileCopier::copy(txn_id.clone(), graph.clone(), context.clone()).await;
                }
                EntryState::Table(context, table) => {
                    FileCopier::copy(txn_id.clone(), table.clone(), context.clone()).await;
                }
            }
        });

        join(join_all(tasks), self.chain.clone().put(txn_id, &entries)).await;
    }

    async fn create(txn: Arc<Transaction>, _: TCValue) -> TCResult<Arc<Directory>> {
        Ok(Arc::new(Directory {
            context: txn.context(),
            chain: Chain::new(txn.context().reserve(&Link::to("/.contents")?)?),
            txn_cache: TransactionCache::new(),
        }))
    }
}
