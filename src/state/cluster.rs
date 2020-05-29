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
use crate::internal::chain::{Chain, Mutation};
use crate::internal::file::*;
use crate::state::*;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::link::{PathSegment, TCPath};
use crate::value::TCResult;

pub struct ClusterConfig;

impl TryFrom<Args> for ClusterConfig {
    type Error = error::TCError;

    fn try_from(_args: Args) -> TCResult<ClusterConfig> {
        Ok(ClusterConfig)
    }
}

#[derive(Clone, Deserialize, Serialize)]
enum StateType {
    Cluster,
    Table,
    Graph,
}

#[derive(Clone, Deserialize, Serialize)]
struct ClusterState {
    name: PathSegment,
    state_type: StateType,
}

impl ClusterState {
    fn new(name: PathSegment, state_type: StateType) -> ClusterState {
        ClusterState { name, state_type }
    }
}

impl Mutation for ClusterState {}

pub struct Cluster {
    context: Arc<Store>,
    chain: Mutex<Chain<ClusterState>>,
}

impl Cluster {
    pub async fn new(txn_id: &TxnId, context: Arc<Store>) -> TCResult<Arc<Cluster>> {
        println!("Cluster::new");

        let chain = Chain::new(txn_id, context.reserve(txn_id, ".contents".parse()?).await?).await;

        Ok(Arc::new(Cluster {
            context: context.clone(),
            chain: Mutex::new(chain),
        }))
    }

    async fn current_entries(&self, txn_id: TxnId) -> Vec<ClusterState> {
        self.chain
            .lock()
            .await
            .stream_into(txn_id)
            .fold(vec![], |mut entries: Vec<ClusterState>, entry| {
                entries.push(entry);
                future::ready(entries)
            })
            .await
    }
}

#[async_trait]
impl Collection for Cluster {
    type Key = TCPath;
    type Value = State;

    async fn get(self: &Arc<Self>, txn: &Arc<Txn<'_>>, path: &Self::Key) -> TCResult<Self::Value> {
        println!("Cluster::get {}", path);
        if path.is_empty() {
            return Err(error::bad_request(
                "You must specify a path to look up in a Cluster",
                path,
            ));
        }

        let entry = self
            .chain
            .lock()
            .await
            .stream_into(txn.id())
            .filter(|entry: &ClusterState| future::ready(entry.name == path[0]))
            .fold(None, |_, m| future::ready(Some(m)))
            .await;

        if let Some(entry) = entry {
            let txn_id = &txn.id();

            if let Some(store) = self
                .context
                .get_store(txn_id, &path[0].clone().into())
                .await
            {
                match entry.state_type {
                    StateType::Cluster => {
                        let cluster = Cluster::from_store(txn_id, store).await;
                        if path.len() == 1 {
                            Ok(cluster.into())
                        } else {
                            cluster.get(txn, &path.slice_from(1)).await
                        }
                    }
                    StateType::Graph if path.len() == 1 => {
                        Ok(Graph::from_store(txn_id, store).await.into())
                    }
                    StateType::Table if path.len() == 1 => {
                        Ok(table::Table::from_store(txn_id, store).await.into())
                    }
                    _ => Err(error::not_found(path)),
                }
            } else {
                Err(error::internal(format!(
                    "Cluster entry {} has no associated data",
                    path
                )))
            }
        } else {
            Err(error::not_found(path))
        }
    }

    async fn put(
        self: Arc<Self>,
        txn: &Arc<Txn<'_>>,
        path: Self::Key,
        state: Self::Value,
    ) -> TCResult<State> {
        let path: PathSegment = path.try_into()?;
        if path.starts_with(".") {
            return Err(error::bad_request(
                "Cluster entry name may not start with a '.'",
                path,
            ));
        }

        let context = self.context.reserve(&txn.id(), path.clone().into()).await?;
        let path_clone = path.clone();
        let entry = match state {
            State::Cluster(d) => {
                FileCopier::copy(txn.id(), &*d, context).await;
                ClusterState::new(path_clone, StateType::Cluster)
            }
            State::Graph(g) => {
                FileCopier::copy(txn.id(), &*g, context).await;
                ClusterState::new(path_clone, StateType::Graph)
            }
            State::Table(t) => {
                FileCopier::copy(txn.id(), &*t, context).await;
                ClusterState::new(path_clone, StateType::Table)
            }
            other => {
                return Err(error::bad_request(
                    "Cluster::put expected a persistent state but found",
                    other,
                ))
            }
        };

        println!("Cluster::put new entry to {}", path);
        self.chain
            .lock()
            .await
            .put(txn.id(), iter::once(entry))
            .await?;
        txn.mutate(self.clone());

        self.get(txn, &path.into()).await
    }
}

#[async_trait]
impl File for Cluster {
    async fn copy_from(
        reader: &mut FileCopier,
        txn_id: &TxnId,
        context: Arc<Store>,
    ) -> Arc<Cluster> {
        let (path, blocks) = reader.next().await.unwrap();

        let chain =
            Chain::copy_from(blocks, txn_id, context.reserve(txn_id, path).await.unwrap()).await;
        let chain = Mutex::new(chain);
        let cluster = Arc::new(Cluster { context, chain });

        for entry in cluster.current_entries(txn_id.clone()).await.drain(..) {
            println!("Cluster::copy_from entry {}", entry.name);

            let dest = cluster
                .context
                .reserve(txn_id, entry.name.into())
                .await
                .unwrap();

            match entry.state_type {
                StateType::Cluster => {
                    Cluster::copy_from(reader, txn_id, dest).await;
                }
                StateType::Graph => {
                    Graph::copy_from(reader, txn_id, dest).await;
                }
                StateType::Table => {
                    table::Table::copy_from(reader, txn_id, dest).await;
                }
            }
        }

        cluster
    }

    async fn copy_into(&self, txn_id: TxnId, writer: &mut FileCopier) {
        println!("Cluster::copy_into copying chain");
        writer.write_file(
            ".contents".parse().unwrap(),
            Box::new(self.chain.lock().await.stream_bytes(txn_id.clone()).boxed()),
        );

        for entry in self.current_entries(txn_id.clone()).await.drain(..) {
            println!("Cluster::copy_into copying entry {}", entry.name);

            let txn_id = txn_id.clone();
            let store = self
                .context
                .get_store(&txn_id, &entry.name.into())
                .await
                .unwrap();

            match entry.state_type {
                StateType::Cluster => {
                    Cluster::from_store(&txn_id, store)
                        .await
                        .copy_into(txn_id, writer)
                        .await
                }
                StateType::Graph => {
                    Graph::from_store(&txn_id, store)
                        .await
                        .copy_into(txn_id, writer)
                        .await
                }
                StateType::Table => {
                    table::Table::from_store(&txn_id, store)
                        .await
                        .copy_into(txn_id, writer)
                        .await
                }
            }
        }
    }

    async fn from_store(txn_id: &TxnId, store: Arc<Store>) -> Arc<Cluster> {
        println!("Cluster::from_store");

        let chain = Chain::from_store(
            txn_id,
            store
                .get_store(txn_id, &".contents".parse().unwrap())
                .await
                .unwrap(),
        )
        .await
        .unwrap();

        Arc::new(Cluster {
            context: store,
            chain: Mutex::new(chain),
        })
    }
}

#[async_trait]
impl Persistent for Cluster {
    type Config = ClusterConfig; // TODO: permissions

    async fn create(txn: &Arc<Txn<'_>>, _: ClusterConfig) -> TCResult<Arc<Cluster>> {
        Cluster::new(&txn.id(), txn.context()).await
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        let mut chain = self.chain.lock().await;
        self.context.commit(txn_id).await;
        chain.commit(txn_id).await;
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.context.rollback(txn_id).await;
        self.chain.lock().await.rollback(txn_id).await;
    }
}
