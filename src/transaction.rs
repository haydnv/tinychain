use std::collections::{BTreeMap, HashMap, HashSet};
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

#[derive(Clone, Debug)]
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

#[derive(Hash)]
enum _Op {
    Get(Subject, TCValue),
    Put(Subject, TCValue, TCValue),
    Post(Option<ValueId>, Link, Vec<(String, ValueId)>),
}

impl _Op {
    fn deps(&self) -> Vec<ValueId> {
        let mut value_ids = Vec::with_capacity(3);

        match self {
            _Op::Get(subject, key) => {
                if let Subject::Ref(sr) = subject {
                    value_ids.push(sr.value_id());
                }
                if let TCValue::Ref(kr) = key {
                    value_ids.push(kr.value_id());
                }
            }
            _Op::Put(subject, key, value) => {
                if let Subject::Ref(sr) = subject {
                    value_ids.push(sr.value_id());
                }
                if let TCValue::Ref(kr) = key {
                    value_ids.push(kr.value_id());
                }
                if let TCValue::Ref(vr) = value {
                    value_ids.push(vr.value_id());
                }
            }
            _Op::Post(subject, _, requires) => {
                if let Some(value_id) = subject {
                    value_ids.push(value_id.clone());
                }
                for (_, value_id) in requires {
                    value_ids.push(value_id.clone());
                }
            }
        }

        value_ids
    }
}

enum QueuedOp {
    Get(Link, TCValue),
    Put(Link, TCValue, TCState),
    Post(Link, Arc<Transaction>),
    GetFrom(TCState, TCValue),
    PutTo(TCState, TCValue, TCState),
    PostTo(TCState, Link, Arc<Transaction>),
}

impl QueuedOp {
    fn from(
        parent_txn: Arc<Transaction>,
        value_id: &ValueId,
        op: &_Op,
        state: &HashMap<ValueId, TCState>,
    ) -> Option<QueuedOp> {
        match op {
            _Op::Get(subject, key) => match subject {
                Subject::Link(l) => Some(QueuedOp::Get(l.clone(), key.clone())),
                Subject::Ref(r) => {
                    if let Some(s) = state.get(&r.value_id()) {
                        Some(QueuedOp::GetFrom(s.clone(), key.clone()))
                    } else {
                        println!("{} not provided", r);
                        None
                    }
                }
            },
            _Op::Put(subject, key, value) => match subject {
                Subject::Link(l) => Some(QueuedOp::Put(l.clone(), key.clone(), value.into())),
                Subject::Ref(r) => {
                    if let Some(s) = state.get(&r.value_id()) {
                        Some(QueuedOp::PutTo(s.clone(), key.clone(), value.into()))
                    } else {
                        println!("{} not provided", r);
                        None
                    }
                }
            },
            _Op::Post(subject, action, requires) => {
                let mut captured: HashMap<String, TCState> = HashMap::new();
                for (subjective_id, current_id) in requires {
                    if let Some(s) = state.get(current_id) {
                        captured.insert(subjective_id.to_owned(), s.clone());
                    } else {
                        println!("{} not provided", current_id);
                        return None;
                    }
                }
                let txn = parent_txn.clone().extend(
                    parent_txn
                        .context
                        .append(&Link::to(&format!("/{}", value_id)).unwrap()),
                    captured,
                );

                if let Some(subject) = subject {
                    if let Some(subject) = state.get(subject) {
                        Some(QueuedOp::PostTo(subject.clone(), action.clone(), txn))
                    } else {
                        println!("{} not provided", subject);
                        None
                    }
                } else {
                    Some(QueuedOp::Post(action.clone(), txn))
                }
            }
        }
    }
}

fn calc_deps(
    value_id: ValueId,
    op: Op,
    state: &mut HashMap<ValueId, TCState>,
    providers: &Map<ValueId, _Op>,
) -> TCResult<()> {
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
                        calc_deps(id.clone(), dep.clone(), state, providers)?;
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

    providers.insert(value_id, Arc::new(_op));
    Ok(())
}

pub struct Transaction {
    host: Arc<Host>,
    id: TransactionId,
    context: Link,
    state: RwLock<HashMap<ValueId, TCState>>,
    providers: Map<ValueId, _Op>,
}

impl Transaction {
    pub fn of(host: Arc<Host>, op: Op) -> TCResult<Arc<Transaction>> {
        let id = TransactionId::new(host.time());
        let context: Link = id.clone().into();

        let mut state: HashMap<ValueId, TCState> = HashMap::new();
        let providers: Map<ValueId, _Op> = Map::new();
        calc_deps(String::from("_"), op, &mut state, &providers)?;

        Ok(Arc::new(Transaction {
            host,
            id,
            context,
            state: RwLock::new(state),
            providers,
        }))
    }

    fn extend(
        self: &Arc<Self>,
        context: Link,
        required: HashMap<String, TCState>,
    ) -> Arc<Transaction> {
        Arc::new(Transaction {
            host: self.host.clone(),
            id: self.id.clone(),
            context: self.context.append(&context),
            state: RwLock::new(required),
            providers: Map::new(),
        })
    }

    pub fn context(self: &Arc<Self>) -> Link {
        self.context.clone()
    }

    fn enqueue(&self, capture: &HashSet<ValueId>) -> TCResult<BTreeMap<ValueId, Arc<_Op>>> {
        let mut providers = BTreeMap::new();
        let mut unresolved = Vec::with_capacity(capture.len());
        unresolved.extend(capture.iter().cloned());

        let state = self.state.read().unwrap();
        while let Some(value_id) = unresolved.pop() {
            if !state.contains_key(&value_id) {
                if let Some(provider) = self.providers.remove(&value_id) {
                    providers.insert(value_id.clone(), provider.clone());
                    for dep in provider.deps() {
                        unresolved.push(dep);
                    }
                } else {
                    return Err(error::bad_request("Required value not provided", value_id));
                }
            }
        }

        Ok(providers)
    }

    pub async fn execute<'a>(
        self: &Arc<Self>,
        capture: HashSet<ValueId>,
    ) -> TCResult<HashMap<ValueId, TCState>> {
        let mut providers = self.enqueue(&capture)?;

        while !providers.is_empty() {
            let mut ready: Vec<(ValueId, QueuedOp)> = vec![];
            for (value_id, provider) in providers.iter().rev() {
                if let Some(op) = QueuedOp::from(
                    self.clone(),
                    &value_id,
                    &provider,
                    &*self.state.read().unwrap(),
                ) {
                    println!("ready: {}", &value_id);
                    ready.push((value_id.clone(), op));
                } else {
                    println!("not ready: {}", value_id);
                    break;
                }
            }

            if ready.is_empty() {
                return Err(error::bad_request(
                    "Transaction graph stalled before completing",
                    "",
                ));
            } else {
                println!(
                    "calculating {:?}",
                    ready.iter().map(|(id, _)| id).collect::<Vec<&ValueId>>()
                );
            }

            let mut futures = vec![];
            for (value_id, op) in &ready {
                providers.remove(value_id);

                let f = match op {
                    QueuedOp::Get(path, key) => self.get(path.clone(), key.clone()).boxed(),
                    QueuedOp::Put(path, key, state) => {
                        self.put(path.clone(), key.clone(), state.clone()).boxed()
                    }
                    QueuedOp::Post(path, txn) => self.host.post(txn.clone(), path).boxed(),
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

    pub fn require(self: &Arc<Self>, value_id: &str) -> TCResult<TCValue> {
        match self.state.read().unwrap().get(value_id) {
            Some(response) => match response {
                TCState::Value(value) => Ok(value.clone()),
                other => Err(error::bad_request(
                    &format!("Required value {} is not serializable", value_id),
                    other,
                )),
            },
            None => Err(error::bad_request(
                &format!("{}: no value was provided for", self.context),
                value_id,
            )),
        }
    }

    pub async fn get(self: &Arc<Self>, path: Link, key: TCValue) -> TCResult<TCState> {
        println!("txn::get {}", path);
        self.host.clone().get(self.clone(), path, key).await
    }

    pub async fn put(
        self: &Arc<Self>,
        path: Link,
        key: TCValue,
        state: TCState,
    ) -> TCResult<TCState> {
        println!("txn::put {} {}", path, key);
        self.host.clone().put(self.clone(), path, key, state).await
    }

    pub async fn post(
        self: &Arc<Self>,
        path: &Link,
        args: Vec<(&str, TCValue)>,
    ) -> TCResult<TCState> {
        println!("txn::post {} {:?}", path, args);

        let txn = self.clone().extend(
            self.context.clone(),
            args.iter()
                .map(|(k, v)| ((*k).to_string(), TCState::Value(v.clone())))
                .collect(),
        );

        self.host.clone().post(txn, path).await
    }
}
