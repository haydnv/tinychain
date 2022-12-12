use std::collections::HashSet;
use std::iter::FromIterator;
use std::sync::Arc;

use futures::future::{try_join_all, FutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use log::{debug, info, warn};
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::Transaction;
use tc_value::{Link, Value};
use tcgeneric::Map;

use crate::state::State;
use crate::txn::Txn;

#[derive(Clone)]
pub struct Owner {
    mutated: Arc<RwLock<HashSet<Link>>>,
}

impl Owner {
    pub fn new() -> Self {
        Self {
            mutated: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub async fn mutate(&self, participant: Link) {
        let mut mutated = self.mutated.write().await;
        if mutated.contains(&participant) {
            log::warn!("got duplicate participant {}", participant);
        } else {
            mutated.insert(participant);
        }
    }

    pub async fn commit(&self, txn: &Txn) -> TCResult<()> {
        let mut mutated = self.mutated.write().await;

        if mutated.is_empty() {
            debug!("no dependencies to commit");
        }

        let mutated = mutated.drain();
        try_join_all(mutated.into_iter().map(|link| {
            info!("sending commit message to dependency at {}", link);
            txn.post(link, Map::<State>::default().into())
        }))
        .await?;

        Ok(())
    }

    pub async fn rollback(&self, txn: &Txn) {
        let mut mutated = self.mutated.write().await;

        if mutated.is_empty() {
            debug!("no dependencies to roll back");
            return;
        }

        let mut rollbacks = FuturesUnordered::from_iter(mutated.drain().map(|dependent| {
            debug!("sending commit message to dependency at {}", dependent);
            txn.delete(dependent.clone(), Value::default())
                .map(|result| (dependent, result))
        }));

        while let Some((dependent, result)) = rollbacks.next().await {
            if let Err(cause) = result {
                warn!(
                    "cluster at {} failed rollback of transaction {}: {}",
                    dependent,
                    txn.id(),
                    cause
                );
            }
        }
    }
}
