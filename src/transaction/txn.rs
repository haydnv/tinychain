use std::collections::{BTreeSet, HashMap, HashSet};
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
use crate::value::{TCString, Value, ValueId};

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
        // TODO: use a Graph here and queue every op absolutely as soon as it's ready

        println!("Txn::execute");

        let mut graph = HashMap::new();
        while let Some((name, value)) = parameters.next().await {
            if let Value::TCString(TCString::Ref(tc_ref)) = value {
                return Err(error::bad_request("Cannot assign a Ref", tc_ref));
            }

            graph.insert(name, State::Value(value));
        }

        for name in capture {
            if !graph.contains_key(&name) {
                return Err(error::not_found(&name));
            }
        }

        let mut pending = FuturesUnordered::new();

        loop {
            let mut visited = HashSet::new();
            let mut unvisited = Vec::with_capacity(graph.len());

            let start = capture
                .iter()
                .filter_map(|name| graph.get_key_value(name))
                .filter_map(|(name, state)| match state {
                    State::Value(Value::Op(_)) => Some(name),
                    _ => None,
                })
                .next();

            let start = if let Some(start) = start {
                start
            } else {
                break;
            };

            unvisited.push(start.clone());
            while let Some(name) = unvisited.pop() {
                visited.insert(name.clone());

                let state = graph.get(&name).ok_or_else(|| error::not_found(&name))?;
                if let State::Value(Value::Op(op)) = state {
                    let mut ready = true;
                    for dep in requires(op) {
                        let dep_state =
                            graph.get(&dep).ok_or_else(|| error::not_found(&dep))?;
                        let resolved = if let State::Value(Value::Op(_)) = dep_state {
                            false
                        } else {
                            true
                        };

                        if !resolved {
                            ready = false;
                            if !visited.contains(&dep) {
                                unvisited.push(dep);
                            }
                        }
                    }

                    if ready {
                        pending.push(
                            self.clone()
                                .resolve(graph.clone(), *op.clone(), auth.clone())
                                .map_ok(|state| (name, state)),
                        );
                    }
                }
            }

            while let Some(result) = pending.next().await {
                let (name, state) = result?;
                graph.insert(name, state);
            }
        }

        Ok(graph
            .drain()
            .filter(|(name, _state)| capture.contains(name))
            .collect())
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

fn requires(op: &Op) -> BTreeSet<ValueId> {
    let mut requires = BTreeSet::new();

    if let Subject::Ref(subject) = op.subject() {
        requires.insert(subject.value_id().clone());
    }

    if let Value::TCString(TCString::Ref(object)) = op.object() {
        requires.insert(object.value_id().clone());
    }

    if let Op::Put(_, _, Value::TCString(TCString::Ref(value))) = op {
        requires.insert(value.value_id().clone());
    }

    requires
}
