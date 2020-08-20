use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use futures::future::{self, TryFutureExt};
use futures::stream::{FuturesUnordered, Stream, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::block::dir::Dir;
use crate::class::{State, TCBoxTryFuture, TCResult};
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::value::link::PathSegment;
use crate::value::{Op, TCRef, TCString, Value, ValueId};

use super::Transact;

const ERR_CORRUPT: &str = "Transaction corrupted! Please file a bug report.";

#[derive(Clone, Debug, Eq, PartialEq, Hash, Deserialize, Serialize)]
pub struct TxnId {
    timestamp: u128, // nanoseconds since Unix epoch
    nonce: u16,
}

impl TxnId {
    pub fn new(time: NetworkTime) -> TxnId {
        TxnId {
            timestamp: time.as_nanos(),
            nonce: rand::thread_rng().gen(),
        }
    }
}

impl Ord for TxnId {
    fn cmp(&self, other: &TxnId) -> std::cmp::Ordering {
        if self.timestamp == other.timestamp {
            self.nonce.cmp(&other.nonce)
        } else {
            self.timestamp.cmp(&other.timestamp)
        }
    }
}

impl PartialOrd for TxnId {
    fn partial_cmp(&self, other: &TxnId) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Into<PathSegment> for TxnId {
    fn into(self) -> PathSegment {
        self.to_string().parse().unwrap()
    }
}

impl Into<String> for TxnId {
    fn into(self) -> String {
        format!("{}-{}", self.timestamp, self.nonce)
    }
}

impl fmt::Display for TxnId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.nonce)
    }
}

pub struct Txn {
    id: TxnId,
    context: Arc<Dir>,
    gateway: Arc<Gateway>,
    mutated: Arc<RwLock<Vec<Arc<dyn Transact>>>>,
}

impl Txn {
    pub async fn new(gateway: Arc<Gateway>, workspace: Arc<Dir>) -> TCResult<Arc<Txn>> {
        let id = TxnId::new(Gateway::time());
        let context: PathSegment = id.clone().try_into()?;
        let context = workspace.create_dir(&id, &context.into()).await?;

        Ok(Arc::new(Txn {
            id,
            context,
            gateway,
            mutated: Arc::new(RwLock::new(vec![])),
        }))
    }

    pub fn context(self: &Arc<Self>) -> Arc<Dir> {
        self.context.clone()
    }

    pub async fn subcontext(&self, subcontext: ValueId) -> TCResult<Arc<Txn>> {
        let subcontext: Arc<Dir> = self
            .context
            .create_dir(&self.id, &subcontext.into())
            .await?;

        Ok(Arc::new(Txn {
            id: self.id.clone(),
            context: subcontext,
            gateway: self.gateway.clone(),
            mutated: self.mutated.clone(),
        }))
    }

    pub fn subcontext_tmp<'a>(&'a self) -> TCBoxTryFuture<'a, Arc<Txn>> {
        Box::pin(async move {
            let id = self.context.unique_id(self.id()).await?;
            self.subcontext(id).await
        })
    }

    pub fn id(&'_ self) -> &'_ TxnId {
        &self.id
    }

    pub async fn execute<S: Stream<Item = (ValueId, Value)> + Unpin>(
        self: Arc<Self>,
        mut parameters: S,
    ) -> TCResult<HashMap<ValueId, State>> {
        let mut resolved = HashMap::new();
        let mut pending = FuturesUnordered::new();
        let mut awaiting = HashSet::<ValueId>::new();

        while let Some((name, value)) = parameters.next().await {
            if resolved.contains_key(&name) || awaiting.contains(&name) {
                return Err(error::bad_request("Duplicate state identifier", name));
            }

            match value {
                Value::TCString(TCString::Ref(tc_ref)) => {
                    return Err(error::bad_request("Cannot assign to a reference", tc_ref));
                }
                Value::Op(op) => {
                    let mut unresolved: HashSet<ValueId> = op
                        .requires()
                        .difference(&resolved.keys().cloned().map(TCRef::from).collect())
                        .cloned()
                        .map(ValueId::from)
                        .collect();

                    while !unresolved.is_empty() && !awaiting.is_empty() {
                        if let Some(result) = pending.next().await {
                            let (name, state) = result?;
                            awaiting.remove(&name);
                            unresolved.remove(&name);
                            resolved.insert(name, state);
                        } else {
                            return Err(error::bad_request(
                                "Some dependencies could not be resolved",
                                Value::Tuple(
                                    unresolved
                                        .into_iter()
                                        .map(TCString::from)
                                        .map(Value::from)
                                        .collect(),
                                ),
                            ));
                        }
                    }

                    if unresolved.is_empty() {
                        awaiting.insert(name.clone());
                        pending.push(
                            self.clone()
                                .resolve(resolved.clone(), *op)
                                .map_ok(|state| (name, state)),
                        );
                    } else {
                        return Err(error::bad_request(
                            "Some dependencies could not be resolved",
                            Value::Tuple(
                                unresolved
                                    .into_iter()
                                    .map(TCString::from)
                                    .map(Value::from)
                                    .collect(),
                            ),
                        ));
                    }
                }
                value => {
                    resolved.insert(name, value.into());
                }
            }
        }

        Ok(resolved)
    }

    pub async fn commit(&self) {
        println!("commit!");

        future::join_all(self.mutated.write().unwrap().drain(..).map(|s| async move {
            s.commit(&self.id).await;
        }))
        .await;
    }

    pub async fn rollback(&self) {
        println!("rollback!");

        future::join_all(self.mutated.write().unwrap().drain(..).map(|s| async move {
            s.rollback(&self.id).await;
        }))
        .await;
    }

    async fn resolve(
        self: Arc<Self>,
        _provided: HashMap<ValueId, State>,
        _provider: Op,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }

    fn mutate(self: &Arc<Self>, state: Arc<dyn Transact>) {
        self.mutated.write().unwrap().push(state)
    }
}
