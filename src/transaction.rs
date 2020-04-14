use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock};

use futures::future::try_join_all;
use futures_util::future::FutureExt;
use rand::Rng;

use crate::cache::Map;
use crate::context::*;
use crate::error;
use crate::host::Host;
use crate::state::TCState;
use crate::value::*;

#[derive(Clone)]
pub struct TransactionId {
    timestamp: u128, // nanoseconds since Unix epoch
    nonce: u16,
}

impl TransactionId {
    fn new(timestamp: u128) -> TransactionId {
        let nonce: u16 = rand::thread_rng().gen();
        TransactionId { timestamp, nonce }
    }
}

impl Into<Link> for TransactionId {
    fn into(self) -> Link {
        Link::to(&format!("/{}-{}", self.timestamp, self.nonce)).unwrap()
    }
}

impl Into<String> for TransactionId {
    fn into(self) -> String {
        format!("{}-{}", self.timestamp, self.nonce)
    }
}

impl Into<Vec<u8>> for TransactionId {
    fn into(self) -> Vec<u8> {
        [
            &self.timestamp.to_be_bytes()[..],
            &self.nonce.to_be_bytes()[..],
        ]
        .concat()
    }
}

impl fmt::Display for TransactionId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.nonce)
    }
}

pub struct Transaction {
    host: Arc<Host>,
    id: TransactionId,
    context: Link,
    state: RwLock<HashMap<ValueId, TCState>>,
    pending: Map<ValueId, _Op>,
}

#[derive(Debug, Hash)]
enum _Op {
    Get(Subject, TCValue),
    Put(Subject, TCValue, TCValue),
    Post(Option<ValueId>, Link, Vec<(String, ValueId)>),
}

fn calc_deps(
    op: Op,
    state: &mut HashMap<ValueId, TCState>,
    pending: &Map<ValueId, _Op>,
) -> TCResult<_Op> {
    let _op = match op {
        Op::Get { subject, key } => _Op::Get(subject, *key),
        Op::Put {
            subject,
            key,
            value,
        } => _Op::Put(subject, *key, *value),
        Op::Post {
            subject,
            action,
            requires,
        } => {
            let mut required_value_ids: Vec<(String, ValueId)> = vec![];
            for (id, provider) in requires {
                match provider {
                    TCValue::Op(dep) => {
                        let pending_dep = calc_deps(dep.clone(), state, pending)?;
                        pending.insert(id.clone(), Arc::new(pending_dep));
                        required_value_ids.push((id.clone(), id.clone()));
                    }
                    TCValue::Ref(r) => {
                        required_value_ids.push((id, r.value_id()));
                    }
                    value => {
                        if state.contains_key(&id) {
                            return Err(error::bad_request("Duplicate values provided for", id));
                        }

                        state.insert(id.clone(), TCState::Value(value));
                        required_value_ids.push((id.clone(), id.clone()));
                    }
                }
            }

            _Op::Post(subject.map(|s| s.value_id()), action, required_value_ids)
        }
    };

    Ok(_op)
}

impl Transaction {
    pub fn of(host: Arc<Host>, op: Op) -> TCResult<Arc<Transaction>> {
        let id = TransactionId::new(host.time());
        let context: Link = id.clone().into();

        let mut state: HashMap<ValueId, TCState> = HashMap::new();
        let pending: Map<ValueId, _Op> = Map::new();
        calc_deps(op, &mut state, &pending)?;

        Ok(Arc::new(Transaction {
            host,
            id,
            context,
            state: RwLock::new(state),
            pending,
        }))
    }

    fn extend(
        self: Arc<Self>,
        context: Link,
        required: HashMap<String, TCState>,
    ) -> Arc<Transaction> {
        Arc::new(Transaction {
            host: self.host.clone(),
            id: self.id.clone(),
            context: self.context.append(&context),
            state: RwLock::new(required),
            pending: Map::new(),
        })
    }

    pub fn context(self: Arc<Self>) -> Link {
        self.context.clone()
    }

    fn enqueue(
        self: Arc<Self>,
        mut unvisited: Vec<ValueId>,
        queue: &mut Vec<ValueId>,
    ) -> TCResult<()> {
        let mut visited: HashSet<ValueId> = HashSet::new();
        while let Some(value_id) = unvisited.pop() {
            if visited.contains(&value_id) {
                continue;
            }

            if let Some(op) = self.pending.get(&value_id) {
                queue.push(value_id.clone());

                match &*op {
                    _Op::Get(subject, key) => {
                        if let Subject::Ref(r) = subject {
                            unvisited.push(r.value_id());
                        }
                        if let TCValue::Ref(r) = key {
                            unvisited.push(r.value_id());
                        }
                    }
                    _Op::Put(subject, key, state) => {
                        if let Subject::Ref(r) = subject {
                            unvisited.push(r.value_id());
                        }
                        if let TCValue::Ref(r) = key {
                            unvisited.push(r.value_id())
                        }
                        if let TCValue::Ref(r) = state {
                            unvisited.push(r.value_id());
                        }
                    }
                    _Op::Post(subject, _, requires) => {
                        if let Some(id) = subject {
                            unvisited.push(id.clone());
                        }

                        for (_, id) in requires {
                            if self.pending.contains_key(&id) {
                                unvisited.push(id.clone());
                            }
                        }
                    }
                }
            } else if !self.state.read().unwrap().contains_key(&value_id) {
                return Err(error::bad_request("Required value not provided", value_id));
            }

            visited.insert(value_id);
        }

        Ok(())
    }

    pub async fn execute<'a>(
        self: Arc<Self>,
        capture: HashSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, TCState>> {
        // TODO: add a TCValue::Ref type and it to support discrete namespaces for each child txn

        let unvisited: Vec<ValueId> = capture.clone().into_iter().collect();
        let mut queue: Vec<ValueId> = vec![];
        self.clone().enqueue(unvisited, &mut queue)?;

        enum QueuedOp {
            Get(Link, TCValue),
            Put(Link, TCValue, TCState),
            Post(Link, Arc<Transaction>),
            GetFrom(TCState, TCValue),
            PutTo(TCState, TCValue, TCState),
            PostTo(TCState, Link, Arc<Transaction>),
        }

        while !queue.is_empty() {
            let mut ready: Vec<(ValueId, QueuedOp)> = vec![];
            while let Some(value_id) = queue.last() {
                let state = self.state.read().unwrap();
                let capture_state = |id| {
                    if let Some(s) = state.get(id) {
                        Some(s)
                    } else {
                        None
                    }
                };

                let op = if let Some(op) = self.pending.get(&value_id) {
                    match &*op {
                        _Op::Get(subject, key) => match subject {
                            Subject::Link(l) => QueuedOp::Get(l.clone(), key.clone()),
                            Subject::Ref(r) => {
                                if let Some(s) = capture_state(&r.value_id()) {
                                    QueuedOp::GetFrom(s.clone(), key.clone())
                                } else {
                                    break;
                                }
                            }
                        },
                        _Op::Put(subject, key, value) => match subject {
                            Subject::Link(l) => QueuedOp::Put(l.clone(), key.clone(), value.into()),
                            Subject::Ref(r) => {
                                if let Some(s) = capture_state(&r.value_id()) {
                                    QueuedOp::PutTo(s.clone(), key.clone(), value.into())
                                } else {
                                    break;
                                }
                            }
                        },
                        _Op::Post(subject, action, requires) => {
                            let mut captured: HashMap<String, TCState> = HashMap::new();
                            for (subjective_id, current_id) in requires {
                                if let Some(s) = capture_state(current_id) {
                                    captured.insert(subjective_id.to_owned(), s.clone());
                                } else {
                                    break;
                                }
                            }
                            let txn = self.clone().extend(
                                self.context.append(&Link::to(&format!("/{}", value_id))?),
                                captured,
                            );

                            if let Some(subject) = subject {
                                if let Some(subject) = capture_state(subject) {
                                    QueuedOp::PostTo(subject.clone(), action.clone(), txn)
                                } else {
                                    break;
                                }
                            } else {
                                QueuedOp::Post(action.clone(), txn)
                            }
                        }
                    }
                } else {
                    break;
                };

                self.pending.remove(&value_id);
                ready.push((queue.pop().unwrap(), op));
            }

            if ready.is_empty() {
                return Err(error::bad_request(
                    "Transaction graph stalled before completing",
                    format!("{:?}", queue),
                ));
            }

            let mut futures = vec![];
            for (_, op) in &ready {
                let f = match op {
                    QueuedOp::Get(path, key) => self.clone().get(path.clone(), key.clone()).boxed(),
                    QueuedOp::Put(path, key, state) => self
                        .clone()
                        .put(path.clone(), key.clone(), state.clone())
                        .boxed(),
                    QueuedOp::Post(path, txn) => self.host.clone().post(txn.clone(), path).boxed(),
                    QueuedOp::GetFrom(subject, key) => {
                        subject.get(self.clone(), key.clone()).boxed()
                    }
                    QueuedOp::PutTo(subject, key, value) => subject
                        .put(self.clone(), key.clone(), value.clone())
                        .boxed(),
                    QueuedOp::PostTo(subject, method, txn) => {
                        subject.post(txn.clone(), method).boxed()
                    }
                };

                futures.push(f)
            }

            let results: Vec<TCState> = try_join_all(futures).await?;

            let mut state = self.state.write().unwrap();
            for i in 0..results.len() {
                state.insert(ready[i].0.to_string(), results[i].clone());
            }
        }

        let state = self.state.read().unwrap();
        let mut responses: HashMap<String, TCState> = HashMap::new();
        for value_id in capture {
            match state.get(&value_id) {
                Some(r) => {
                    responses.insert(value_id, r.clone());
                }
                None => {
                    return Err(error::bad_request(
                        "Tried to capture unknown value",
                        value_id,
                    ));
                }
            }
        }

        Ok(responses)
    }

    pub fn require(self: Arc<Self>, value_id: &str) -> TCResult<TCValue> {
        match self.state.read().unwrap().get(value_id) {
            Some(response) => match response {
                TCState::Value(value) => Ok(value.clone()),
                other => Err(error::bad_request(
                    &format!("Required value {} is not serialiable", value_id),
                    other,
                )),
            },
            None => Err(error::bad_request(
                &format!("{}: no value was provided for", self.context),
                value_id,
            )),
        }
    }

    pub async fn get(self: Arc<Self>, path: Link, key: TCValue) -> TCResult<TCState> {
        self.host.clone().get(self.clone(), path, key).await
    }

    pub async fn put(
        self: Arc<Self>,
        path: Link,
        key: TCValue,
        state: TCState,
    ) -> TCResult<TCState> {
        self.host.clone().put(self.clone(), path, key, state).await
    }

    pub async fn post(
        self: Arc<Self>,
        path: &Link,
        args: Vec<(&str, TCValue)>,
    ) -> TCResult<TCState> {
        let txn = self.clone().extend(
            self.context.clone(),
            args.iter()
                .map(|(k, v)| ((*k).to_string(), TCState::Value(v.clone())))
                .collect(),
        );

        self.host.clone().post(txn, path).await
    }
}
