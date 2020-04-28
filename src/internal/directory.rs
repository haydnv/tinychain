use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use futures::future::{join, join_all};
use futures::{FutureExt, StreamExt};

use crate::error;
use crate::internal::file;
use crate::internal::{Chain, FsDir};
use crate::state::*;
use crate::transaction::{Transaction, TransactionId};
use crate::value::{Link, TCResult, TCValue};

struct Directory {
    context: Arc<FsDir>,
    chain: Arc<Chain>,
    txn_cache: RwLock<HashMap<TransactionId, HashMap<Link, (Arc<FsDir>, State)>>>,
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
        } else if state.is_value() {
            return Err(error::bad_request(
                "Expected a persistent state, found",
                state,
            ));
        }

        let entry = (self.context.reserve(&path)?, state);
        if let Some(mutations) = self.txn_cache.write().unwrap().get_mut(&txn.id()) {
            mutations.insert(path, entry);
        } else {
            let mut mutations: HashMap<Link, (Arc<FsDir>, State)> = HashMap::new();
            mutations.insert(path, entry);
            self.txn_cache.write().unwrap().insert(txn.id(), mutations);
        }

        Ok(self)
    }
}

#[async_trait]
impl file::File for Directory {
    async fn from_file(copier: &mut file::FileCopier, dest: Arc<FsDir>) -> Arc<Directory> {
        let (path, blocks) = copier.next().await.unwrap();
        let chain = Chain::from(blocks, dest.reserve(&path).unwrap()).await;

        Arc::new(Directory {
            context: dest,
            chain,
            txn_cache: RwLock::new(HashMap::new()),
        })
    }

    async fn copy_file(&self, _txn_id: TransactionId, _writer: &mut file::FileCopier) {
        // TODO
    }
}

#[async_trait]
impl Persistent for Directory {
    type Config = TCValue; // TODO: permissions

    async fn commit(&self, txn_id: &TransactionId) {
        let mutations = self.txn_cache.write().unwrap().remove(&txn_id);
        if let Some(mutations) = mutations {
            let paths: Vec<Link> = mutations.keys().cloned().collect();
            let tasks = mutations.values().map(|(context, state)| async move {
                match state {
                    State::Graph(graph) => {
                        file::FileCopier::new()
                            .copy(txn_id.clone(), graph.clone(), context.clone())
                            .await;
                    }
                    State::Table(table) => {
                        file::FileCopier::new()
                            .copy(txn_id.clone(), table.clone(), context.clone())
                            .await;
                    }
                    State::Value(_) => {
                        panic!("Tried to file::copy a Value! This should never happen")
                    }
                }
            });

            join(join_all(tasks), self.chain.clone().put(txn_id, &paths)).await;
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
