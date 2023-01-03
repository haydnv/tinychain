use std::collections::HashSet;
use std::iter::FromIterator;
use std::sync::Arc;

use futures::future::FutureExt;
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, warn};
use tokio::sync::RwLock;

use tc_transact::Transaction;
use tc_value::{Link, Value};
use tcgeneric::Map;

use crate::state::State;
use crate::txn::Txn;

#[derive(Clone)]
pub struct Leader {
    mutated: Arc<RwLock<HashSet<Link>>>,
}

impl Leader {
    pub fn new() -> Self {
        Self {
            mutated: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub async fn mutate(&self, participant: Link) {
        let mut mutated = self.mutated.write().await;
        if mutated.contains(&participant) {
            log::info!("got duplicate participant {}", participant);
        } else {
            mutated.insert(participant);
        }
    }

    pub async fn commit(&self, txn: &Txn) {
        let mut mutated = self.mutated.write().await;

        if mutated.is_empty() {
            debug!("no dependencies to commit");
            return;
        }

        let mut commits = FuturesUnordered::from_iter(mutated.drain().map(|dep| {
            debug!("sending commit message to dependency at {}", dep);
            txn.post(dep.clone(), State::Map(Map::default()))
                .map(|result| (dep, result))
        }));

        while let Some((dep, result)) = commits.next().await {
            if let Err(cause) = result {
                warn!(
                    "cluster at {} failed commit of transaction {}: {}",
                    dep,
                    txn.id(),
                    cause
                );
            }
        }
    }

    pub async fn rollback(&self, txn: &Txn) {
        let mut mutated = self.mutated.write().await;

        if mutated.is_empty() {
            debug!("no dependencies to roll back");
            return;
        }

        let mut rollbacks = FuturesUnordered::from_iter(mutated.drain().map(|dependent| {
            debug!("sending rollback message to dependency at {}", dependent);
            txn.delete(dependent.clone(), Value::default())
                .map(|result| (dependent, result))
        }));

        while let Some((dep, result)) = rollbacks.next().await {
            if let Err(cause) = result {
                warn!(
                    "cluster at {} failed rollback of transaction {}: {}",
                    dep,
                    txn.id(),
                    cause
                );
            }
        }
    }
}
