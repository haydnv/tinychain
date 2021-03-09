use std::collections::HashSet;

use futures::future::try_join_all;
use log::debug;

use uplock::RwLock;

use tc_error::*;
use tcgeneric::Map;

use crate::scalar::{Link, Value};
use crate::state::State;
use crate::txn::Txn;

#[derive(Clone)]
pub struct Owner {
    mutated: RwLock<HashSet<Link>>,
}

impl Owner {
    pub fn new() -> Self {
        Self {
            mutated: RwLock::new(HashSet::new()),
        }
    }

    pub async fn mutate(&self, participant: Link) {
        let mut mutated = self.mutated.write().await;
        mutated.insert(participant);
    }

    pub async fn commit(&self, txn: &Txn) -> TCResult<()> {
        let mut mutated = self.mutated.write().await;

        if mutated.is_empty() {
            debug!("no dependencies to commit");
        }

        let mutated = mutated.drain();
        try_join_all(mutated.into_iter().map(|link| {
            debug!("sending commit message to dependency at {}", link);
            txn.post(link, Map::<State>::default().into())
        }))
        .await?;

        Ok(())
    }

    pub async fn rollback(&self, txn: &Txn) -> TCResult<()> {
        let mut mutated = self.mutated.write().await;

        if mutated.is_empty() {
            debug!("no dependencies to roll back");
        }

        let mutated = mutated.drain();
        try_join_all(mutated.into_iter().map(|link| {
            debug!("sending commit message to dependency at {}", link);
            txn.delete(link, Value::None)
        }))
        .await?;

        Ok(())
    }
}
