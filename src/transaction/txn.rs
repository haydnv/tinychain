use std::collections::{HashMap, HashSet};
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use futures::future::{self, try_join_all, TryFutureExt};
use futures::stream::{self, FuturesUnordered, Stream, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::auth::Auth;
use crate::block::dir::{Dir, DirEntry};
use crate::block::file::File;
use crate::block::BlockData;
use crate::class::{ResponseStream, State, TCBoxTryFuture, TCResult, TCStream};
use crate::collection::class::CollectionInstance;
use crate::collection::Collection;
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::lock::RwLock;
use crate::value::link::PathSegment;
use crate::value::op::{Op, Subject};
use crate::value::{TCRef, TCString, Value, ValueId};

use super::Transact;

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
        mut parameters: S,
        capture: &HashSet<ValueId>,
        auth: &Auth,
    ) -> TCResult<HashMap<ValueId, State>> {
        println!("Txn::execute");

        let mut providers = HashMap::new();
        let mut pending = vec![];
        let mut ongoing = FuturesUnordered::new();
        let mut awaiting = HashSet::new();
        let mut resolved = HashMap::new();

        while let Some((name, value)) = parameters.next().await {
            if providers.contains_key(&name) {
                return Err(error::bad_request("Duplicate state identifier", name));
            } else {
                println!("param {}: {}", name, value);
            }

            if capture.contains(&name) {
                if let Value::Op(provider) = value {
                    pending.push((name, *provider));
                } else if let Value::TCString(TCString::Ref(tc_ref)) = value {
                    return Err(error::bad_request("Cannot assign a reference", tc_ref));
                } else {
                    resolved.insert(name, State::Value(value));
                }
            } else {
                providers.insert(name, value);
            }

            while let Some((name, provider)) = pending.pop() {
                while awaiting.contains(&name) {
                    if let Some(result) = ongoing.next().await {
                        let (name, state) = result?;
                        awaiting.remove(&name);
                        resolved.insert(name, state);
                    } else {
                        return Err(error::not_found(name));
                    }
                }

                if resolved.contains_key(&name) {
                    continue;
                }

                let resolved_ids: HashSet<ValueId> = resolved.keys().cloned().collect();
                let mut required: HashSet<ValueId> =
                    requires(&provider).drain().map(ValueId::from).collect();
                if required.is_subset(&resolved_ids) {
                    awaiting.insert(name.clone());
                    ongoing.push(
                        self.clone()
                            .resolve(resolved.clone(), provider, auth.clone())
                            .map_ok(|state| (name, state)),
                    );
                } else {
                    pending.push((name, provider));

                    for dep in required.drain() {
                        if resolved.contains_key(&dep) {
                            continue;
                        }

                        if let Some(provider) = providers.remove(&dep) {
                            if let Value::Op(provider) = provider {
                                pending.push((dep, *provider));
                            } else {
                                resolved.insert(dep, State::Value(provider));
                            }
                        } else {
                            return Err(error::not_found(dep));
                        }
                    }
                }
            }
        }

        while let Some(result) = ongoing.next().await {
            let (name, state) = result?;
            awaiting.remove(&name);
            resolved.insert(name, state);
        }

        Ok(resolved)
    }

    pub async fn execute_and_stream<S: Stream<Item = (ValueId, Value)> + Unpin>(
        self: Arc<Self>,
        parameters: S,
        mut capture: HashSet<ValueId>,
        auth: &Auth,
    ) -> TCResult<ResponseStream> {
        let mut txn_state = self.clone().execute(parameters, &capture, auth).await?;
        let mut streams = Vec::with_capacity(capture.len());
        for value_id in capture.drain() {
            let this = self.clone();
            let state = txn_state
                .remove(&value_id)
                .ok_or_else(|| error::not_found(&value_id))?;

            streams.push(async move {
                match state {
                    State::Collection(collection) => {
                        let stream: TCStream<Value> = collection.to_stream(this).await?;
                        TCResult::Ok((value_id, stream))
                    }
                    State::Value(value) => {
                        let stream: TCStream<Value> = Box::pin(stream::once(future::ready(value)));
                        TCResult::Ok((value_id, stream))
                    }
                }
            });
        }

        let response = try_join_all(streams).await?;
        let response: ResponseStream = Box::pin(stream::iter(response));
        Ok(response)
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
        println!("Txn::resolve {}", provider);

        match provider {
            Op::Get(subject, object) => match subject {
                Subject::Link(link) => {
                    let object = resolve_value(&provided, &object)?.clone();
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

fn resolve_value<'a>(
    provided: &'a HashMap<ValueId, State>,
    object: &'a Value,
) -> TCResult<&'a Value> {
    match object {
        Value::TCString(TCString::Ref(object)) => match provided.get(object.value_id()) {
            Some(State::Value(object)) => Ok(object),
            Some(other) => Err(error::bad_request("Expected Value but found", other)),
            None => Err(error::not_found(object)),
        },
        other => Ok(other),
    }
}

fn requires(op: &Op) -> HashSet<TCRef> {
    let mut requires = HashSet::with_capacity(3);

    if let Subject::Ref(subject) = op.subject() {
        requires.insert(subject.clone());
    }

    if let Value::TCString(TCString::Ref(object)) = op.object() {
        requires.insert(object.clone());
    }

    if let Op::Put(_, _, Value::TCString(TCString::Ref(value))) = op {
        requires.insert(value.clone());
    }

    requires
}
