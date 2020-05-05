use std::convert::TryInto;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, join, join_all};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::cache::TransactionCache;
use crate::internal::chain::{Chain, ChainBlock, Mutation};
use crate::internal::file::*;
use crate::state::*;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{PathSegment, TCPath, TCResult, TCValue};

#[derive(Clone, Deserialize, Serialize)]
enum EntryType {
    Directory,
    Table,
    Graph,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct DirEntry {
    name: PathSegment,
    entry_type: EntryType,
}

impl Mutation for DirEntry {}

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
            chain: Chain::new(context.reserve("contents".parse()?)?),
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
            .stream_into::<DirEntry>(Some(txn.id()))
            .filter(|entry: &DirEntry| future::ready(&entry.name == path))
            .fold(None, |_, m| future::ready(Some(m)))
            .await;

        if let Some(entry) = entry {
            if let Some(store) = self.context.get_store(&path[0].clone().into()) {
                match entry.entry_type {
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

        let context = self.context.reserve(path.clone().into())?;
        let entry = match state {
            State::Graph(g) => (EntryType::Graph, EntryState::Graph(context, g)),
            State::Table(t) => (EntryType::Table, EntryState::Table(context, t)),
            State::Value(v) => {
                return Err(error::bad_request("Expected a persistent state, found", v))
            }
        };

        self.txn_cache.insert(txn.id(), path, entry);
        txn.mutate(self.clone());
        println!("Directory::put complete");
        Ok(self)
    }
}

#[async_trait]
impl File for Directory {
    type Block = ChainBlock<DirEntry>;

    async fn copy_from(reader: &mut FileCopier, dest: Arc<Store>) -> Arc<Directory> {
        let (path, blocks) = reader.next().await.unwrap();
        let chain = Chain::copy_from(blocks, dest.reserve(path).unwrap()).await;

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
impl Transact for Directory {
    async fn commit(&self, txn_id: &TransactionId) {
        println!("Directory::commit");
        let mutations = self.txn_cache.close(txn_id);
        println!("Directory closed transaction cache for {}", txn_id);
        let mut entries: Vec<DirEntry> = Vec::with_capacity(mutations.len());
        let mut tasks = Vec::with_capacity(mutations.len());
        for (name, (entry_type, state)) in mutations.iter() {
            entries.push(DirEntry {
                name: name.clone(),
                entry_type: entry_type.clone(),
            });
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

        println!("Directory::commit scheduled");
        join(join_all(tasks), self.chain.clone().put(txn_id, &entries)).await;
        println!("Directory::commit complete");
    }
}
