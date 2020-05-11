use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::{future, StreamExt};
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::chain::{Chain, ChainBlock, Mutation, PendingMutation};
use crate::internal::file::*;
use crate::state::*;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, PathSegment, TCPath, TCResult, TCValue};

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
    owner: Option<Link>,
}

impl DirEntry {
    fn new(name: PathSegment, entry_type: EntryType, owner: Option<Link>) -> DirEntry {
        DirEntry {
            name,
            entry_type,
            owner,
        }
    }
}

impl From<PendingDirEntry> for DirEntry {
    fn from(pending: PendingDirEntry) -> DirEntry {
        pending.0
    }
}

impl Mutation for DirEntry {}

#[derive(Clone)]
enum EntrySource {
    Directory(Arc<Store>, Arc<Directory>),
    Table(Arc<Store>, Arc<Table>),
    Graph(Arc<Store>, Arc<Graph>),
}

type PendingDirEntry = (DirEntry, EntrySource);

#[async_trait]
impl PendingMutation<DirEntry> for PendingDirEntry {
    async fn commit(self, txn_id: &TransactionId) -> DirEntry {
        use EntrySource::*;

        match &self.1 {
            Directory(context, dir) => {
                FileCopier::copy(txn_id.clone(), &*dir.clone(), context.clone()).await;
            }
            Graph(context, graph) => {
                FileCopier::copy(txn_id.clone(), &*graph.clone(), context.clone()).await;
            }
            Table(context, table) => {
                FileCopier::copy(txn_id.clone(), &*table.clone(), context.clone()).await;
            }
        }

        self.0
    }
}

pub struct Directory {
    context: Arc<Store>,
    chain: Arc<Chain<DirEntry, PendingDirEntry>>,
}

impl Directory {
    pub fn new(context: Arc<Store>) -> TCResult<Arc<Directory>> {
        Ok(Arc::new(Directory {
            context: context.clone(),
            chain: Chain::new(context.reserve("contents".parse()?)?),
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
        }

        let entry = self
            .chain
            .clone()
            .stream_into(Some(txn.id()))
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
        let path: PathSegment = path.try_into()?;

        let context = self.context.reserve(path.clone().into())?;
        let entry = match state {
            State::Graph(g) => (
                DirEntry::new(path.clone(), EntryType::Graph, None),
                EntrySource::Graph(context, g),
            ),
            State::Table(t) => (
                DirEntry::new(path.clone(), EntryType::Table, None),
                EntrySource::Table(context, t),
            ),
            State::Object(_) => return Err(error::not_implemented()),
            State::Value(v) => {
                return Err(error::bad_request(
                    "Directory::put expected a persistent state but found",
                    v,
                ))
            }
        };

        self.chain.put(txn.id(), iter::once(entry));
        txn.mutate(self.chain.clone());
        Ok(self)
    }
}

#[async_trait]
impl File for Directory {
    type Block = ChainBlock<DirEntry>;

    async fn copy_from(reader: &mut FileCopier, context: Arc<Store>) -> Arc<Directory> {
        let (path, blocks) = reader.next().await.unwrap();
        let chain = Chain::copy_from(blocks, context.reserve(path).unwrap()).await;

        Arc::new(Directory { context, chain })
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
