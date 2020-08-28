use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use futures::future::{self, TryFutureExt};
use futures::stream::{FuturesUnordered, Stream, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::auth::Auth;
use crate::block::dir::{Dir, DirEntry};
use crate::block::file::File;
use crate::block::BlockData;
use crate::class::{State, TCBoxTryFuture, TCResult};
use crate::collection::Collection;
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::lock::RwLock;
use crate::value::link::PathSegment;
use crate::value::op::{Op, Subject};
use crate::value::{TCRef, TCString, Value, ValueId};

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
    dir: Arc<Dir>,
    context: ValueId,
    gateway: Arc<Gateway>,
    mutated: RwLock<Vec<Collection>>,
}

impl Txn {
    pub async fn new(gateway: Arc<Gateway>, workspace: Arc<Dir>) -> TCResult<Arc<Txn>> {
        let id = TxnId::new(Gateway::time());
        let context: PathSegment = id.clone().try_into()?;
        let dir = workspace.create_dir(&id, &context.clone().into()).await?;

        println!("new Txn: {}", id);

        Ok(Arc::new(Txn {
            id,
            dir,
            context,
            gateway,
            mutated: RwLock::new(vec![]),
        }))
    }

    pub async fn context<T: BlockData>(&self) -> TCResult<Arc<File<T>>>
    where
        Arc<File<T>>: Into<DirEntry>,
    {
        self.dir
            .create_file(self.id.clone(), self.context.clone())
            .await
    }

    pub async fn subcontext(&self, subcontext: ValueId) -> TCResult<Arc<Txn>> {
        let dir = self
            .dir
            .get_or_create_dir(&self.id, &self.context.clone().into())
            .await?;

        Ok(Arc::new(Txn {
            id: self.id.clone(),
            dir,
            context: subcontext,
            gateway: self.gateway.clone(),
            mutated: self.mutated.clone(),
        }))
    }

    pub fn subcontext_tmp<'a>(&'a self) -> TCBoxTryFuture<'a, Arc<Txn>> {
        Box::pin(async move {
            let id = self.dir.unique_id(self.id()).await?;
            self.subcontext(id).await
        })
    }

    pub fn id(&'_ self) -> &'_ TxnId {
        &self.id
    }

    pub async fn execute<S: Stream<Item = (ValueId, Value)> + Unpin>(
        self: Arc<Self>,
        auth: &Auth,
        mut parameters: S,
    ) -> TCResult<HashMap<ValueId, State>> {
        println!("Txn::execute");

        let mut resolved = HashMap::new();
        let mut pending = FuturesUnordered::new();
        let mut awaiting = HashSet::<ValueId>::new();

        while let Some((name, value)) = parameters.next().await {
            if resolved.contains_key(&name) || awaiting.contains(&name) {
                return Err(error::bad_request("Duplicate state identifier", name));
            } else {
                println!("param {}: {}", name, value);
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
                                .resolve(resolved.clone(), *op, auth.clone())
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

        while let Some(result) = pending.next().await {
            let (name, state) = result?;
            resolved.insert(name, state);
        }

        Ok(resolved)
    }

    pub async fn commit(&self) {
        println!("commit!");

        future::join_all(self.mutated.write().await.drain(..).map(|s| async move {
            s.commit(&self.id).await;
        }))
        .await;
    }

    pub async fn rollback(&self) {
        println!("rollback!");

        future::join_all(self.mutated.write().await.drain(..).map(|s| async move {
            s.rollback(&self.id).await;
        }))
        .await;
    }

    async fn resolve(
        self: Arc<Self>,
        provided: HashMap<ValueId, State>,
        provider: Op,
        auth: Auth,
    ) -> TCResult<State> {
        match provider {
            Op::Get(subject, object) => match subject {
                Subject::Link(link) => {
                    self.gateway
                        .get(&link, object, &auth, Some(self.clone()))
                        .await
                }
                Subject::Ref(tc_ref) => {
                    let subject = provided
                        .get(&tc_ref.clone().into())
                        .ok_or_else(|| error::not_found(tc_ref))?;
                    if let State::Collection(collection) = subject {
                        collection.get(self.clone(), object).await
                    } else {
                        Err(error::bad_request("Value does not support GET", subject))
                    }
                }
            },
            Op::Put(subject, object, value) => {
                let value: State = if let Value::TCString(TCString::Id(tc_ref)) = value {
                    provided
                        .get(&tc_ref.clone())
                        .cloned()
                        .ok_or_else(|| error::not_found(tc_ref))?
                } else {
                    value.into()
                };

                match subject {
                    Subject::Link(link) => {
                        self.gateway
                            .put(&link, object, value, &auth, Some(self.id.clone()))
                            .await
                    }
                    Subject::Ref(tc_ref) => {
                        let subject = provided
                            .get(&tc_ref.clone().into())
                            .ok_or_else(|| error::not_found(tc_ref))?;

                        if let State::Collection(collection) = subject {
                            self.mutate(collection.clone()).await;

                            collection
                                .put(&self, &object, value)
                                .await
                                .map(State::Collection)
                        } else {
                            Err(error::bad_request("Value does not support GET", subject))
                        }
                    }
                }
            }
        }
    }

    async fn mutate(self: &Arc<Self>, state: Collection) {
        self.mutated.write().await.push(state)
    }
}
