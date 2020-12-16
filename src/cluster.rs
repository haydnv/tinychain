use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::try_join;
use log::debug;

use crate::block::Dir;
use crate::chain::Chain;
use crate::class::{State, TCResult};
use crate::error;
use crate::handler::Public;
use crate::object::InstanceExt;
use crate::request::Request;
use crate::scalar::*;
use crate::transaction::lock::{Mutable, Mutate, TxnLock};
use crate::transaction::{Transact, Txn, TxnId};

const ERR_ID: &str = "Invalid Id for Cluster member";

#[derive(Clone)]
enum ClusterReplica {
    Director(HashSet<LinkHost>), // set of all hosts replicating this cluster
    Actor(LinkHost),             // link to the director
}

impl Default for ClusterReplica {
    fn default() -> ClusterReplica {
        ClusterReplica::Director(HashSet::new())
    }
}

#[derive(Clone)]
struct ClusterState {
    replica: ClusterReplica,
}

#[async_trait]
impl Mutate for ClusterState {
    type Pending = Self;

    fn diverge(&self, _txn_id: &TxnId) -> Self::Pending {
        self.clone()
    }

    async fn converge(&mut self, new_value: Self::Pending) {
        *self = new_value
    }
}

#[derive(Clone)]
pub struct Cluster {
    path: TCPathBuf,
    data_dir: Arc<Dir>,
    workspace: Arc<Dir>,
    state: TxnLock<ClusterState>,
    chains: TxnLock<Mutable<HashMap<Id, InstanceExt<Chain>>>>,
    methods: TxnLock<Mutable<Object>>,
}

impl Cluster {
    pub fn create(path: TCPathBuf, data_dir: Arc<Dir>, workspace: Arc<Dir>) -> TCResult<Cluster> {
        let replica = ClusterReplica::default();
        let state = TxnLock::new(
            format!("State of Cluster at {}", path),
            ClusterState { replica },
        );

        let chains = TxnLock::new(
            format!("Chains of Cluster at {}", path),
            HashMap::new().into(),
        );

        let methods = TxnLock::new(
            format!("Object of Cluster at {}", path),
            Object::default().into(),
        );

        Ok(Cluster {
            path,
            data_dir,
            workspace,
            state,
            chains,
            methods,
        })
    }
}

#[async_trait]
impl Public for Cluster {
    async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State> {
        if path.is_empty() {
            let key: Id = key.try_cast_into(|v| error::bad_request(ERR_ID, v))?;

            let methods = self.methods.read(txn.id()).await?;
            if let Some(method) = (**methods).get(&key) {
                return Ok(State::Scalar(method.clone()));
            }

            let chains = self.chains.read(txn.id()).await?;
            if let Some(chain) = chains.get(&key) {
                return Ok(State::from(chain.clone()));
            }

            Err(error::not_found(key))
        } else {
            let methods = self.methods.read(txn.id()).await?;
            if methods.contains_key(&path[0]) {
                return methods.get(request, txn, path, key).await;
            }

            let chains = self.chains.read(txn.id()).await?;
            if let Some(chain) = chains.get(&path[0]) {
                return chain.get(request, txn, &path[1..], key).await;
            }

            Err(error::path_not_found(path))
        }
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        if path.is_empty() {
            if key.is_none() {
                Err(error::not_implemented("Cluster::PUT /"))
            } else {
                let key: Id = key.try_cast_into(|v| error::bad_request(ERR_ID, v))?;

                match value {
                    State::Chain(chain) => {
                        let mut chains = self.chains.write(*txn.id()).await?;
                        chains.insert(key, chain.into());
                        Ok(())
                    }
                    State::Object(crate::object::Object::Instance(chain)) => {
                        let mut chains = self.chains.write(*txn.id()).await?;
                        let chain = chain.try_as()?;
                        chains.insert(key, chain);
                        Ok(())
                    }
                    State::Scalar(Scalar::Op(op_def)) => {
                        let mut methods = self.methods.write(*txn.id()).await?;
                        methods.insert(key, Scalar::Op(op_def));
                        Ok(())
                    }
                    other => Err(error::bad_request(
                        "Cluster member must be a Chain or OpDef, not",
                        other,
                    )),
                }
            }
        } else {
            let methods = self.methods.read(txn.id()).await?;
            if methods.contains_key(&path[0]) {
                return methods.put(request, txn, path, key, value).await;
            } else {
                debug!("Cluster at {} has no method called {}", self.path, path[0]);
            }

            let chains = self.chains.read(txn.id()).await?;
            if let Some(chain) = chains.get(&path[0]) {
                return chain.put(request, txn, &path[1..], key, value).await;
            } else {
                debug!("Cluster at {} has no chain called {}", self.path, path[0]);
            }

            Err(error::path_not_found(path))
        }
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        if path.is_empty() {
            return Err(error::method_not_allowed(self));
        }

        let methods = self.methods.read(txn.id()).await?;
        if methods.contains_key(&path[0]) {
            return methods.post(request, txn, path, params).await;
        }

        let chains = self.chains.read(txn.id()).await?;
        if let Some(chain) = chains.get(&path[0]) {
            return chain.post(request, txn, &path[1..], params).await;
        }

        Err(error::path_not_found(path))
    }

    async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<()> {
        let txn_id = *txn.id();

        if path.is_empty() {
            if key.is_none() {
                let (mut methods, mut chains) =
                    try_join!(self.methods.write(txn_id), self.chains.write(txn_id))?;
                *methods = Object::default();
                *chains = HashMap::new();
                Ok(())
            } else {
                let key: Id = key.try_cast_into(|v| error::bad_request(ERR_ID, v))?;

                let mut methods = self.methods.write(txn_id).await?;
                if methods.remove(&key).is_some() {
                    return Ok(());
                }

                let mut chains = self.chains.write(txn_id).await?;
                if chains.remove(&key).is_some() {
                    return Ok(());
                }

                Err(error::not_found(key))
            }
        } else {
            let methods = self.methods.read(txn.id()).await?;
            if methods.contains_key(&path[0]) {
                return methods.delete(request, txn, path, key).await;
            }

            let chains = self.chains.read(txn.id()).await?;
            if let Some(chain) = chains.get(&path[0]) {
                return chain.delete(request, txn, &path[1..], key).await;
            }

            Err(error::path_not_found(path))
        }
    }
}

#[async_trait]
impl Transact for Cluster {
    async fn commit(&self, txn_id: &TxnId) {
        debug!("Cluster::commit!");
        self.state.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        debug!("Cluster::rollback!");
        self.state.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("Cluster::finalize!");
        self.state.finalize(txn_id).await
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", self.path)
    }
}
