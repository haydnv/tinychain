use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, join, join_all};
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::chain::{Chain, ChainBlock, Mutation};
use crate::internal::file::*;
use crate::state::*;
use crate::transaction::{Transact, Transaction, TransactionId};
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

impl Mutation for DirEntry {}

pub struct Directory {
    context: Arc<Store>,
    chain: Arc<Chain<DirEntry>>,
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
        let context = self
            .context
            .reserve_or_get(&vec![txn.id().into(), path.clone()].into())?;

        let entry = match state {
            State::Graph(g) => {
                FileCopier::copy(txn.id(), &*g, context).await;
                DirEntry::new(path.clone(), EntryType::Graph, None)
            }
            State::Table(t) => {
                FileCopier::copy(txn.id(), &*t, context).await;
                DirEntry::new(path.clone(), EntryType::Table, None)
            }
            State::Object(_) => return Err(error::not_implemented()),
            State::Value(v) => {
                return Err(error::bad_request(
                    "Directory::put expected a persistent state but found",
                    v,
                ))
            }
        };

        // TODO: roll this back if the transaction fails elsewhere
        self.context.reserve(path.into())?;
        self.chain.put(txn.id(), iter::once(entry));
        txn.mutate(self.clone());
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
        panic!("Directory::copy_into is not implemented")
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
        let copies = join_all(self.chain.get_pending(txn_id).drain(..).map(|m: DirEntry| {
            let source = self
                .context
                .get_store(&vec![txn_id.clone().into(), m.name.clone()].into())
                .unwrap();
            let dest = self.context.get_store(&m.name.clone().into()).unwrap();

            async move {
                match m.entry_type {
                    EntryType::Directory => {
                        let d = Directory::from_store(source).await;
                        FileCopier::copy(txn_id.clone(), &*d, dest).await;
                    }
                    EntryType::Graph => {
                        let g = Graph::from_store(source).await;
                        FileCopier::copy(txn_id.clone(), &*g, dest).await;
                    }
                    EntryType::Table => {
                        let t = Table::from_store(source).await;
                        FileCopier::copy(txn_id.clone(), &*t, dest).await;
                    }
                }
            }
        }));

        join(copies, self.chain.commit(txn_id)).await;
    }
}
