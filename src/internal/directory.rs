use std::convert::TryInto;
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use futures::lock::Mutex;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::internal::block::Store;
use crate::internal::chain::{Chain, ChainBlock, Mutation};
use crate::internal::file::*;
use crate::object::actor::Token;
use crate::state::*;
use crate::transaction::{Transact, Txn, TxnId};
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
    chain: Mutex<Chain<DirEntry>>,
}

impl Directory {
    pub async fn new(txn_id: &TxnId, context: Arc<Store>) -> TCResult<Arc<Directory>> {
        println!("Directory::new");
        Ok(Arc::new(Directory {
            context: context.clone(),
            chain: Mutex::new(
                Chain::new(txn_id, context.reserve(txn_id, ".contents".parse()?).await?).await,
            ),
        }))
    }
}

#[async_trait]
impl Collection for Directory {
    type Key = TCPath;
    type Value = State;

    async fn get(
        self: &Arc<Self>,
        txn: Arc<Txn>,
        path: &Self::Key,
        auth: &Option<Token>,
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
            .lock()
            .await
            .stream_into(txn.id())
            .filter(|entry: &DirEntry| future::ready(&entry.name == path))
            .fold(None, |_, m| future::ready(Some(m)))
            .await;

        if let Some(entry) = entry {
            let txn_id = &txn.id();
            if let Some(store) = self
                .context
                .get_store(txn_id, &path[0].clone().into())
                .await
            {
                match entry.entry_type {
                    EntryType::Directory => {
                        let dir = Directory::from_store(txn_id, store).await;
                        dir.get(txn, &path.slice_from(1), auth).await
                    }
                    EntryType::Graph => Ok(Graph::from_store(txn_id, store).await.into()),
                    EntryType::Table => Ok(Table::from_store(txn_id, store).await.into()),
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
        txn: Arc<Txn>,
        path: Self::Key,
        state: Self::Value,
        _auth: &Option<Token>,
    ) -> TCResult<Arc<Self>> {
        let path: PathSegment = path.try_into()?;
        let context = self.context.reserve(&txn.id(), path.clone().into()).await?;
        let chain = self.chain.lock().await;

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

        println!("Directory::put new entry to {}", path);
        chain.put(txn.id(), iter::once(entry)).await?;
        txn.mutate(self.clone());
        Ok(self.clone())
    }
}

#[async_trait]
impl File for Directory {
    type Block = ChainBlock<DirEntry>;

    async fn copy_from(
        reader: &mut FileCopier,
        txn_id: &TxnId,
        context: Arc<Store>,
    ) -> Arc<Directory> {
        let (path, blocks) = reader.next().await.unwrap();
        let chain = Mutex::new(
            Chain::copy_from(blocks, txn_id, context.reserve(txn_id, path).await.unwrap()).await,
        );

        Arc::new(Directory { context, chain })
    }

    async fn copy_into(&self, _txn_id: TxnId, _writer: &mut FileCopier) {
        panic!("Directory::copy_into is not implemented")
    }

    async fn from_store(_txn_id: &TxnId, _store: Arc<Store>) -> Arc<Directory> {
        panic!("Directory::from_store is not implemented")
    }
}

#[async_trait]
impl Persistent for Directory {
    type Config = TCValue; // TODO: permissions

    async fn create(txn: Arc<Txn>, _: TCValue) -> TCResult<Arc<Directory>> {
        Directory::new(&txn.id(), txn.context()).await
    }
}

#[async_trait]
impl Transact for Directory {
    async fn commit(&self, txn_id: &TxnId) {
        let mut chain = self.chain.lock().await;
        self.context.commit(txn_id).await;
        chain.commit(txn_id).await;
    }
}
