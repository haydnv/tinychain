use std::collections::HashSet;

use futures::future::try_join_all;
use log::debug;

use uplock::RwLock;

use error::*;
use generic::Map;

use crate::scalar::Link;
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

    pub async fn mutate(&self, peer: Link) -> TCResult<()> {
        let mut mutated = self.mutated.write().await;
        mutated.insert(peer);
        Ok(())
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
}
