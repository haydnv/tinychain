use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures::future::{join, join_all};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::file::*;
use crate::internal::{Chain, FsDir};
use crate::state::*;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCResult, TCValue};

#[derive(Deserialize, Serialize)]
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
    Directory(Arc<FsDir>, Arc<Directory>),
    Table(Arc<FsDir>, Arc<Table>),
    Graph(Arc<FsDir>, Arc<Graph>),
}

struct Directory {
    context: Arc<FsDir>,
    chain: Arc<Chain>,
    txn_cache: RwLock<HashMap<TransactionId, HashMap<Entry, EntryState>>>,
}

#[async_trait]
impl Collection for Directory {
    type Key = Link;
    type Value = State;

    async fn get(
        self: &Arc<Self>,
        _txn: Arc<Transaction>,
        _key: &Self::Key,
    ) -> TCResult<Self::Value> {
        Err(error::not_implemented())
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

        if let Some(mutations) = self.txn_cache.write().unwrap().get_mut(&txn.id()) {
            mutations.insert(entry.0, entry.1);
        } else {
            let mut mutations: HashMap<Entry, EntryState> = HashMap::new();
            mutations.insert(entry.0, entry.1);
            self.txn_cache.write().unwrap().insert(txn.id(), mutations);
        }

        Ok(self)
    }
}

#[async_trait]
impl File for Directory {
    async fn from_file(copier: &mut FileCopier, dest: Arc<FsDir>) -> Arc<Directory> {
        let (path, blocks) = copier.next().await.unwrap();
        let chain = Chain::from(blocks, dest.reserve(&path).unwrap()).await;

        Arc::new(Directory {
            context: dest,
            chain,
            txn_cache: RwLock::new(HashMap::new()),
        })
    }

    async fn copy_file(&self, _txn_id: TransactionId, _writer: &mut FileCopier) {
        // TODO
    }
}

#[async_trait]
impl Persistent for Directory {
    type Config = TCValue; // TODO: permissions

    async fn commit(&self, txn_id: &TransactionId) {
        let mutations = self.txn_cache.write().unwrap().remove(&txn_id);
        if let Some(mutations) = mutations {
            let entries: Vec<&Entry> = mutations.keys().collect();
            let tasks = mutations.values().map(|state| async move {
                match state {
                    EntryState::Directory(context, dir) => {
                        FileCopier::new()
                            .copy(txn_id.clone(), dir.clone(), context.clone())
                            .await;
                    }
                    EntryState::Graph(context, graph) => {
                        FileCopier::new()
                            .copy(txn_id.clone(), graph.clone(), context.clone())
                            .await;
                    }
                    EntryState::Table(context, table) => {
                        FileCopier::new()
                            .copy(txn_id.clone(), table.clone(), context.clone())
                            .await;
                    }
                }
            });

            join(join_all(tasks), self.chain.clone().put(txn_id, &entries)).await;
        }
    }

    async fn create(txn: Arc<Transaction>, _: TCValue) -> TCResult<Arc<Directory>> {
        Ok(Arc::new(Directory {
            context: txn.context(),
            chain: Chain::new(txn.context().reserve(&Link::to("/.contents")?)?),
            txn_cache: RwLock::new(HashMap::new()),
        }))
    }
}
