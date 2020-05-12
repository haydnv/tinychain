use std::collections::{HashMap, HashSet, VecDeque};
use std::convert::TryInto;
use std::fmt;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{self, join_all, try_join_all, Future, FutureExt};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::host::{Host, NetworkTime};
use crate::internal::block::Store;
use crate::internal::cache::{Map, Single};
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
    pub fn new(time: NetworkTime) -> TransactionId {
        TransactionId {
            timestamp: time.as_nanos(),
            nonce: rand::thread_rng().gen(),
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

struct TransactionState {
    known: HashSet<TCRef>,
    queue: VecDeque<(ValueId, Op)>,
    resolved: HashMap<ValueId, State>,
}

impl TransactionState {
    fn new() -> TransactionState {
        TransactionState {
            known: HashSet::new(),
            queue: VecDeque::new(),
            resolved: HashMap::new(),
        }
    }

    fn push(&mut self, value: (ValueId, TCValue)) -> TCResult<()> {
        if self.resolved.contains_key(&value.0) {
            return Err(error::bad_request("Duplicate value provided for", value.0));
        }
        self.known.insert(value.0.clone().into());

        match value.1 {
            TCValue::Op(op) => {
                let required = op.deps();
                let unknown: Vec<&TCRef> = required.difference(&self.known).collect();
                if !unknown.is_empty() {
                    let unknown: TCValue = unknown.into_iter().cloned().collect();
                    Err(error::bad_request(
                        "Some required values were not provided",
                        unknown,
                    ))
                } else {
                    if required.is_empty() {
                        self.queue.push_front((value.0, op));
                    } else {
                        self.queue.push_back((value.0, op));
                    }

                    Ok(())
                }
            }
            _ => {
                self.resolved.insert(value.0, value.1.into());
                Ok(())
            }
        }
    }

    async fn resolve(
        &mut self,
        txn: Arc<Transaction>,
        capture: HashSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, State>> {
        let resolved: Map<ValueId, State> = self.resolved.drain().collect();
        while !self.queue.is_empty() {
            let known: HashSet<TCRef> = resolved.keys().drain().map(|id| id.into()).collect();
            let mut ready = vec![];
            let mut value_ids = vec![];
            while let Some((value_id, op)) = self.queue.pop_front() {
                if op.deps().is_subset(&known) {
                    ready.push(txn.resolve_value(&resolved, value_id.clone(), op));
                    println!("ready: {}", value_id);
                    value_ids.push(value_id);
                } else {
                    self.queue.push_front((value_id, op));
                    break;
                }
            }

            let values = try_join_all(ready).await?.into_iter().map(|s| {
                println!("resolved {}", value_ids[0]);
                (value_ids.remove(0), s)
            });
            resolved.extend(values);
            println!("{} remaining to resolve", self.queue.len());
        }

        let resolved = resolved
            .drain()
            .drain()
            .filter(|(id, _)| capture.contains(id))
            .collect();

        Ok(resolved)
    }
}

pub struct Transaction {
    id: TransactionId,
    context: Arc<Store>,
    host: Arc<Host>,
    mutated: Arc<RwLock<Vec<Arc<dyn Transact>>>>,
    state: RwLock<Single<TransactionState>>,
}

impl Transaction {
    pub async fn from_iter<I: IntoIterator<Item = (ValueId, TCValue)>>(
        host: Arc<Host>,
        root: Arc<Store>,
        iter: I,
    ) -> TCResult<Arc<Transaction>> {
        println!();
        println!("Transaction::from_iter");

        let mut state = TransactionState::new();
        for item in iter {
            state.push(item)?;
        }

        Transaction::with_state(host, root, state).await
    }

    pub async fn new(host: Arc<Host>, root: Arc<Store>) -> TCResult<Arc<Transaction>> {
        Transaction::with_state(host, root, TransactionState::new()).await
    }

    async fn with_state(
        host: Arc<Host>,
        root: Arc<Store>,
        txn_state: TransactionState,
    ) -> TCResult<Arc<Transaction>> {
        let id = TransactionId::new(host.time());
        let context: PathSegment = id.clone().try_into()?;
        let context = root.reserve(&id, context.into()).await?;
        let state = RwLock::new(Single::new(Some(txn_state)));

        Ok(Arc::new(Transaction {
            id,
            context,
            host,
            mutated: Arc::new(RwLock::new(vec![])),
            state,
        }))
    }

    pub fn context(self: &Arc<Self>) -> Arc<Store> {
        self.context.clone()
    }

    async fn extend(self: &Arc<Self>, context: ValueId) -> TCResult<Arc<Transaction>> {
        let context: Arc<Store> = self.context.reserve(&self.id, context.into()).await?;

        Ok(Arc::new(Transaction {
            id: self.id.clone(),
            context,
            host: self.host.clone(),
            mutated: self.mutated.clone(),
            state: RwLock::new(Single::new(None)),
        }))
    }

    pub fn id(self: &Arc<Self>) -> TransactionId {
        self.id.clone()
    }

    pub fn commit(&self) -> impl Future<Output = ()> + '_ {
        println!("commit!");
        join_all(self.mutated.write().unwrap().drain(..).map(|s| async move {
            s.commit(&self.id).await;
        }))
        .then(|_| future::ready(()))
    }

    pub fn mutate(self: &Arc<Self>, state: Arc<dyn Transact>) {
        self.mutated.write().unwrap().push(state)
    }

    pub async fn resolve(
        self: &Arc<Self>,
        capture: HashSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, State>> {
        let mut state = self.state.write().unwrap().replace(None).unwrap();
        state.resolve(self.clone(), capture).await
    }

    async fn resolve_value(
        self: &Arc<Self>,
        resolved: &Map<ValueId, State>,
        value_id: ValueId,
        op: Op,
    ) -> TCResult<State> {
        let extension = self.extend(value_id).await?;

        match op {
            Op::Get { subject, key } => match subject {
                Subject::Link(l) => extension.get(l, *key).await,
                Subject::Ref(r) => match resolved.get(&r.value_id()) {
                    Some(s) => s.get(extension, *key).await,
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
                Subject::Link(l) => extension.put(l, *key, resolve_val(resolved, *value)?).await,
                Subject::Ref(r) => {
                    let subject = resolve_id(resolved, &r.value_id())?;
                    let value = resolve_val(resolved, *value)?;
                    subject.put(extension, *key, value.try_into()?).await
                }
            },
            Op::Post {
                subject,
                action,
                requires,
            } => {
                let mut deps: Vec<(ValueId, TCValue)> = Vec::with_capacity(requires.len());
                for (dest_id, id) in requires {
                    let dep = resolve_val(resolved, id)?;
                    deps.push((dest_id, dep.try_into()?));
                }

                match subject {
                    Some(r) => {
                        let subject = resolve_id(resolved, &r.value_id())?;
                        subject
                            .post(extension, &action.try_into()?, deps.into())
                            .await
                    }
                    None => {
                        self.host
                            .post(extension, &action.clone().into(), deps.into())
                            .await
                    }
                }
            }
        }
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

fn resolve_id(resolved: &Map<ValueId, State>, id: &ValueId) -> TCResult<State> {
    match resolved.get(id) {
        Some(s) => Ok(s),
        None => Err(error::bad_request("Required value not provided", id)),
    }
}

fn resolve_val(resolved: &Map<ValueId, State>, value: TCValue) -> TCResult<State> {
    match value {
        TCValue::Ref(r) => resolve_id(resolved, &r.value_id()),
        _ => Ok(value.into()),
    }
}
