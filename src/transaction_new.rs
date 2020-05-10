use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, RwLock};

use futures::future::{join_all, try_join_all};

use crate::error;
use crate::state::State;
use crate::transaction::{Transact, TransactionId};
use crate::value::{Op, TCRef, TCResult, TCValue, ValueId};

struct TransactionState {
    known: BTreeSet<TCRef>,
    queue: VecDeque<(ValueId, Op)>,
}

impl TransactionState {
    fn push(&mut self, op: (ValueId, Op)) -> TCResult<()> {
        let required = op.1.deps();
        let unknown: Vec<&TCRef> = required.difference(&self.known).collect();
        if !unknown.is_empty() {
            let unknown: TCValue = unknown.into_iter().cloned().collect();
            Err(error::bad_request(
                "Some required values were not provided",
                unknown,
            ))
        } else {
            if required.is_empty() {
                self.queue.push_front(op);
            } else {
                self.queue.push_back(op);
            }

            Ok(())
        }
    }

    async fn resolve(
        &mut self,
        txn: &Arc<Transaction>,
        capture: BTreeSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, State>> {
        let mut resolved: HashMap<ValueId, State> = HashMap::new();
        while !self.queue.is_empty() {
            let known: BTreeSet<TCRef> = resolved.keys().cloned().map(|id| id.into()).collect();
            let mut ready = vec![];
            let mut value_ids = vec![];
            loop {
                let (value_id, op) = self.queue.pop_front().unwrap();
                if op.deps().is_subset(&known) {
                    ready.push(txn.execute(op));
                    value_ids.push(value_id);
                } else {
                    self.queue.push_front((value_id, op));
                    break;
                }
            }

            resolved.extend(
                try_join_all(ready)
                    .await?
                    .into_iter()
                    .map(|s| (value_ids.remove(0), s)),
            );
        }

        Ok(resolved
            .into_iter()
            .filter(|(id, _)| capture.contains(id))
            .collect())
    }
}

struct Transaction {
    id: TransactionId,
    mutated: RwLock<Vec<Arc<dyn Transact>>>,
    state: RwLock<TransactionState>,
}

impl Transaction {
    pub async fn commit(&self) {
        println!("commit!");

        let mut mutated = self.mutated.write().unwrap();
        let mut tasks = Vec::with_capacity(mutated.len());
        while let Some(state) = mutated.pop() {
            tasks.push(async move { state.commit(&self.id).await });
        }
        join_all(tasks).await;
    }

    pub async fn execute(self: &Arc<Self>, _op: Op) -> TCResult<State> {
        Err(error::not_implemented())
    }

    pub fn mutate(self: &Arc<Self>, state: Arc<dyn Transact>) {
        self.mutated.write().unwrap().push(state)
    }

    pub fn push(&self, op: (ValueId, Op)) -> TCResult<()> {
        self.state.write().unwrap().push(op)
    }

    pub async fn resolve(
        self: &Arc<Self>,
        capture: BTreeSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, State>> {
        self.state.write().unwrap().resolve(self, capture).await
    }
}
