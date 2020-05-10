use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::join_all;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::host::{Host, NetworkTime};
use crate::internal::block::Store;
use crate::internal::cache::{Deque, Map};
use crate::state::State;
use crate::value::*;

#[async_trait]
pub trait Transact: Send + Sync {
    async fn commit(&self, txn_id: &TransactionId);
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Deserialize, Serialize)]
pub struct TransactionId {
    timestamp: u128, // nanoseconds since Unix epoch
    nonce: u16,
}

impl TransactionId {
    fn new(time: NetworkTime) -> TransactionId {
        TransactionId {
            timestamp: time.as_nanos(),
            nonce: rand::thread_rng().gen(),
        }
    }
}

impl From<Bytes> for TransactionId {
    fn from(b: Bytes) -> TransactionId {
        if b.len() != 18 {
            panic!("TransactionId should be exactly 18 bytes");
        }

        TransactionId {
            timestamp: u128::from_be_bytes(b[..16].try_into().expect("Bad transaction timestamp")),
            nonce: u16::from_be_bytes(b[16..18].try_into().expect("Bad transaction nonce")),
        }
    }
}

impl PartialOrd for TransactionId {
    fn partial_cmp(&self, other: &TransactionId) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TransactionId {
    fn cmp(&self, other: &TransactionId) -> std::cmp::Ordering {
        if self.timestamp == other.timestamp {
            self.nonce.cmp(&other.nonce)
        } else {
            self.timestamp.cmp(&other.timestamp)
        }
    }
}

impl Into<PathSegment> for TransactionId {
    fn into(self) -> PathSegment {
        self.to_string().parse().unwrap()
    }
}

impl Into<String> for TransactionId {
    fn into(self) -> String {
        format!("{}-{}", self.timestamp, self.nonce)
    }
}

impl Into<Bytes> for TransactionId {
    fn into(self) -> Bytes {
        Bytes::from(
            [
                &self.timestamp.to_be_bytes()[..],
                &self.nonce.to_be_bytes()[..],
            ]
            .concat(),
        )
    }
}

impl Into<Bytes> for &TransactionId {
    fn into(self) -> Bytes {
        Bytes::from(
            [
                &self.timestamp.to_be_bytes()[..],
                &self.nonce.to_be_bytes()[..],
            ]
            .concat(),
        )
    }
}

impl fmt::Display for TransactionId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.nonce)
    }
}

fn calc_deps(
    value_id: ValueId,
    op: Op,
    state: &mut HashMap<ValueId, State>,
    queue: &Deque<(ValueId, Op)>,
) -> TCResult<()> {
    if let Op::Post { requires, .. } = &op {
        let mut required_value_ids: Vec<(ValueId, ValueId)> = vec![];
        for (id, provider) in requires {
            match provider {
                TCValue::Op(dep) => {
                    calc_deps(id.clone(), dep.clone(), state, queue)?;
                    required_value_ids.push((id.clone(), id.clone()));
                }
                TCValue::Ref(r) => {
                    required_value_ids.push((id.clone(), r.value_id().clone()));
                }
                value => {
                    if state.contains_key(id) {
                        return Err(error::bad_request("Duplicate values provided for", id));
                    }

                    state.insert(id.clone(), State::Value(value.clone()));
                    required_value_ids.push((id.clone(), id.clone()));
                }
            }
        }
    }

    queue.push_back((value_id, op));
    Ok(())
}

pub struct Transaction {
    host: Arc<Host>,
    id: TransactionId,
    context: Arc<Store>,
    state: Map<ValueId, State>,
    queue: Deque<(ValueId, Op)>,
    mutated: Deque<Arc<dyn Transact>>,
}

impl Transaction {
    pub fn new(host: Arc<Host>, root: Arc<Store>) -> TCResult<Arc<Transaction>> {
        let id = TransactionId::new(host.time());
        let context: PathSegment = id.clone().try_into()?;
        let context = root.reserve(context.into())?;

        println!();
        println!("Transaction::new");

        Ok(Arc::new(Transaction {
            host,
            id,
            context,
            state: Map::new(),
            queue: Deque::new(),
            mutated: Deque::new(),
        }))
    }

    pub fn of(host: Arc<Host>, root: Arc<Store>, op: Op) -> TCResult<Arc<Transaction>> {
        let id = TransactionId::new(host.time());
        let context: PathSegment = id.clone().try_into()?;
        let context = root.reserve(context.into())?;

        let mut state: HashMap<ValueId, State> = HashMap::new();
        let queue: Deque<(ValueId, Op)> = Deque::new();
        calc_deps("_".parse()?, op, &mut state, &queue)?;

        println!();
        println!("Transaction::of");

        Ok(Arc::new(Transaction {
            host,
            id,
            context,
            state: state.into_iter().collect(),
            queue,
            mutated: Deque::new(),
        }))
    }

    pub fn context(self: &Arc<Self>) -> Arc<Store> {
        self.context.clone()
    }

    pub fn id(self: &Arc<Self>) -> TransactionId {
        self.id.clone()
    }

    pub async fn commit(self: &Arc<Self>) {
        println!("commit!");
        let mut tasks = Vec::with_capacity(self.mutated.len());
        while let Some(state) = self.mutated.pop_front() {
            tasks.push(async move { state.commit(&self.id).await });
        }
        join_all(tasks).await;
    }

    pub async fn execute<'a>(
        self: &Arc<Self>,
        capture: HashSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, State>> {
        // TODO: use FuturesUnordered to parallelize tasks

        while let Some((value_id, op)) = self.queue.pop_front() {
            let state = match op {
                Op::Get { subject, key } => match subject {
                    Subject::Link(l) => self.clone().get(l, *key).await,
                    Subject::Ref(r) => match self.state.get(&r.value_id()) {
                        Some(s) => s.get(self.clone(), *key).await,
                        None => Err(error::bad_request(
                            "Required value not provided",
                            r.value_id(),
                        )),
                    },
                },
                Op::Put {
                    subject,
                    key,
                    value,
                } => match subject {
                    Subject::Link(l) => self.clone().put(l, *key, self.resolve_val(*value)?).await,
                    Subject::Ref(r) => {
                        let subject = self.resolve(&r.value_id())?;
                        let value = self.resolve_val(*value)?;
                        subject.put(self.clone(), *key, value.try_into()?).await
                    }
                },
                Op::Post {
                    subject,
                    action,
                    requires,
                } => {
                    let mut deps: Vec<(ValueId, TCValue)> = Vec::with_capacity(requires.len());
                    for (dest_id, id) in requires {
                        let dep = self.resolve_val(id)?;
                        deps.push((dest_id, dep.try_into()?));
                    }

                    match subject {
                        Some(r) => {
                            let subject = self.resolve(&r.value_id())?;
                            subject
                                .post(self.clone(), &action.try_into()?, deps.into())
                                .await
                        }
                        None => {
                            self.host
                                .post(self.clone(), &action.clone().into(), deps.into())
                                .await
                        }
                    }
                }
            };

            self.state.insert(value_id, state?);
        }

        let mut results: HashMap<ValueId, State> = HashMap::new();
        for value_id in capture {
            match self.state.get(&value_id) {
                Some(state) => {
                    results.insert(value_id, state);
                }
                None => {
                    return Err(error::bad_request(
                        "There is no such value to capture",
                        value_id,
                    ));
                }
            }
        }

        Ok(results)
    }

    fn resolve(self: &Arc<Self>, id: &ValueId) -> TCResult<State> {
        match self.state.get(id) {
            Some(s) => Ok(s),
            None => Err(error::bad_request("Required value not provided", id)),
        }
    }

    fn resolve_val(self: &Arc<Self>, value: TCValue) -> TCResult<State> {
        match value {
            TCValue::Ref(r) => self.resolve(&r.value_id()),
            _ => Ok(value.into()),
        }
    }

    pub fn mutate(self: &Arc<Self>, state: Arc<dyn Transact>) {
        // TODO: don't queue state if it's already in the queue
        self.mutated.push_back(state)
    }

    pub fn time(&self) -> NetworkTime {
        NetworkTime::from_nanos(self.id.timestamp)
    }

    pub async fn get(self: &Arc<Self>, link: Link, key: TCValue) -> TCResult<State> {
        println!("txn::get {} {}", link, key);
        self.host.get(self.clone(), &link, key).await
    }

    pub async fn put(self: &Arc<Self>, dest: Link, key: TCValue, state: State) -> TCResult<State> {
        println!("txn::put {} {}", dest, key);
        self.host.put(self.clone(), dest, key, state).await
    }

    pub async fn post(self: &Arc<Self>, dest: &Link, args: Args) -> TCResult<State> {
        println!("txn::post {}", dest);
        self.host.post(self.clone(), dest, args).await
    }
}
