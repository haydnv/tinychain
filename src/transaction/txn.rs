use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt;
use std::hash::Hash;
use std::iter;
use std::slice;
use std::str::FromStr;
use std::sync::Arc;

use futures::future::{self, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, Stream, StreamExt};
use rand::Rng;
use serde::de;
use tokio::sync::mpsc;

use crate::block::{BlockData, Dir, DirEntry, File};
use crate::chain::ChainInstance;
use crate::class::{State, TCBoxTryFuture, TCResult};
use crate::collection::class::CollectionInstance;
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::lock::RwLock;
use crate::request::Request;
use crate::scalar::*;

use super::Transact;

const INVALID_ID: &str = "Invalid transaction ID";

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
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

    pub fn to_path(&self) -> PathSegment {
        self.to_string().parse().unwrap()
    }

    pub fn zero() -> TxnId {
        TxnId {
            timestamp: 0,
            nonce: 0,
        }
    }
}

impl FromStr for TxnId {
    type Err = error::TCError;

    fn from_str(s: &str) -> TCResult<TxnId> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() == 2 {
            let timestamp = parts[0]
                .parse()
                .map_err(|e| error::bad_request(INVALID_ID, e))?;
            let nonce = parts[1]
                .parse()
                .map_err(|e| error::bad_request(INVALID_ID, e))?;
            Ok(TxnId { timestamp, nonce })
        } else {
            Err(error::bad_request(INVALID_ID, s))
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

impl<'de> de::Deserialize<'de> for TxnId {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s = String::deserialize(d)?;
        Self::from_str(&s).map_err(de::Error::custom)
    }
}

impl fmt::Display for TxnId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.nonce)
    }
}

struct Inner {
    id: TxnId,
    workspace: Arc<Dir>,
    dir: Arc<Dir>,
    context: ValueId,
    gateway: Arc<Gateway>,
    mutated: RwLock<Vec<Box<dyn Transact>>>,
    txn_server: mpsc::UnboundedSender<TxnId>,
}

#[derive(Clone)]
pub struct Txn {
    inner: Arc<Inner>,
}

impl Txn {
    pub async fn new(
        gateway: Arc<Gateway>,
        workspace: Arc<Dir>,
        id: TxnId,
        txn_server: mpsc::UnboundedSender<TxnId>,
    ) -> TCResult<Txn> {
        let context = id.to_path();
        let dir = workspace
            .create_dir(id.clone(), slice::from_ref(&context))
            .await?;

        println!("new Txn: {}", id);

        let inner = Arc::new(Inner {
            id,
            workspace,
            dir,
            context,
            gateway,
            mutated: RwLock::new(vec![]),
            txn_server,
        });

        Ok(Txn { inner })
    }

    pub async fn context<T: BlockData>(&self) -> TCResult<Arc<File<T>>>
    where
        Arc<File<T>>: Into<DirEntry>,
    {
        self.inner
            .dir
            .create_file(self.inner.id.clone(), self.inner.context.clone())
            .await
    }

    pub async fn subcontext(&self, subcontext: ValueId) -> TCResult<Txn> {
        let dir = self
            .inner
            .dir
            .get_or_create_dir(&self.inner.id, slice::from_ref(&self.inner.context))
            .await?;

        let subcontext = Arc::new(Inner {
            id: self.inner.id.clone(),
            workspace: self.inner.dir.clone(),
            dir,
            context: subcontext,
            gateway: self.inner.gateway.clone(),
            mutated: self.inner.mutated.clone(),
            txn_server: self.inner.txn_server.clone(),
        });

        Ok(Txn { inner: subcontext })
    }

    pub fn subcontext_tmp(&self) -> TCBoxTryFuture<Txn> {
        Box::pin(async move {
            let id = self.inner.dir.unique_id(self.id()).await?;
            self.subcontext(id).await
        })
    }

    pub fn id(&'_ self) -> &'_ TxnId {
        &self.inner.id
    }

    pub async fn execute<I: Into<State>, S: Stream<Item = (ValueId, I)> + Unpin>(
        &self,
        request: &Request,
        mut parameters: S,
    ) -> TCResult<State> {
        validate_id(request, &self.inner.id)?;

        println!("Txn::execute");

        let mut graph: HashMap<ValueId, State> = HashMap::new();
        let mut capture = None;
        while let Some((name, state)) = parameters.next().await {
            let state: State = state.into();
            println!("pending: {}: {}", name, state);
            capture = Some(name.clone());
            graph.insert(name, state);
        }

        let capture =
            capture.ok_or_else(|| error::unsupported("Cannot execute empty operation"))?;

        let mut pending = FuturesUnordered::new();

        while !is_resolved(
            graph
                .get(&capture)
                .ok_or_else(|| error::not_found(&capture))?,
        ) {
            let mut visited = HashSet::new();
            let mut unvisited = Vec::with_capacity(graph.len());
            unvisited.push(capture.clone());
            while let Some(name) = unvisited.pop() {
                if visited.contains(&name) {
                    println!("Already visited {}", name);
                    continue;
                } else {
                    visited.insert(name.clone());
                }

                println!("Txn::execute {} (#{})", &name, visited.len());

                let state = graph.get(&name).ok_or_else(|| error::not_found(&name))?;
                if let State::Scalar(scalar) = state {
                    let mut ready = true;

                    if let Scalar::Op(op) = scalar {
                        if op.is_def() {
                            continue;
                        }

                        println!("Provider: {}", &op);
                        for dep in requires(op, &graph)? {
                            if dep == name {
                                return Err(error::bad_request("Dependency cycle", dep));
                            }

                            println!("requires {}", dep);
                            let dep_state =
                                graph.get(&dep).ok_or_else(|| error::not_found(&dep))?;

                            if !is_resolved(dep_state) {
                                ready = false;
                                unvisited.push(dep);
                            }
                        }

                        if ready {
                            println!("queueing dep {}: {}", name, state);
                            pending.push(
                                self.resolve(request, graph.clone(), *op.clone())
                                    .map(|r| (name, r)),
                            );
                        }
                    } else if let Scalar::Tuple(tuple) = scalar {
                        let mut ready = true;
                        for dep in tuple {
                            if let Scalar::Value(Value::TCString(TCString::Ref(dep))) = dep {
                                if dep.value_id() == &name {
                                    return Err(error::bad_request("Dependency cycle", dep));
                                }

                                let dep_state = graph
                                    .get(dep.value_id())
                                    .ok_or_else(|| error::not_found(&dep))?;

                                if !is_resolved(dep_state) {
                                    ready = false;
                                    unvisited.push(dep.value_id().clone());
                                }
                            }
                        }

                        if ready {
                            let resolved = dereference_state(&graph, scalar)?;
                            graph.insert(name, resolved);
                        }
                    } else if let Scalar::Value(Value::Tuple(tuple)) = scalar {
                        let mut ready = true;
                        for dep in tuple {
                            if let Value::TCString(TCString::Ref(dep)) = dep {
                                if dep.value_id() == &name {
                                    return Err(error::bad_request("Dependency cycle", dep));
                                }

                                let dep_state = graph
                                    .get(dep.value_id())
                                    .ok_or_else(|| error::not_found(&dep))?;

                                if !is_resolved(dep_state) {
                                    ready = false;
                                    unvisited.push(dep.value_id().clone());
                                }
                            }
                        }

                        if ready {
                            let resolved = dereference_state(&graph, scalar)?;
                            graph.insert(name, resolved);
                        }
                    }
                }
            }

            while let Some((name, result)) = pending.next().await {
                if let Err(cause) = &result {
                    println!("Error resolving {}: {}", name, cause);
                }

                let state = result?;
                graph.insert(name, state);
            }
        }

        println!("Txn::execute complete, returning {}...", capture);
        graph
            .remove(&capture)
            .ok_or_else(|| error::not_found(capture))
    }

    pub async fn commit(&self) {
        println!("commit!");

        future::join_all(
            self.inner
                .mutated
                .write()
                .await
                .drain(..)
                .map(|s| async move {
                    s.commit(&self.inner.id).await;
                }),
        )
        .await;
    }

    pub async fn rollback(&self) {
        println!("rollback!");

        future::join_all(
            self.inner
                .mutated
                .write()
                .await
                .drain(..)
                .map(|s| async move {
                    s.rollback(&self.inner.id).await;
                }),
        )
        .await;
    }

    pub async fn finalize(&self) {
        println!("finalize!");

        self.inner
            .workspace
            .delete(self.id().clone(), self.id().to_path())
            .await
            .unwrap();

        future::join_all(
            self.inner
                .mutated
                .write()
                .await
                .drain(..)
                .map(|s| async move {
                    s.finalize(self.id()).await;
                }),
        )
        .await;

        self.inner.workspace.finalize(self.id()).await;
    }

    pub async fn resolve(
        &self,
        request: &Request,
        provided: HashMap<ValueId, State>,
        provider: Op,
    ) -> TCResult<State> {
        validate_id(request, &self.inner.id)?;

        println!("Txn::resolve {}", provider);

        match provider {
            Op::Def(op_def) => Err(error::not_implemented(format!("Txn::resolve {}", op_def))),
            Op::Ref(OpRef::If(cond, then, or_else)) => {
                let cond = provided
                    .get(cond.value_id())
                    .ok_or_else(|| error::not_found(cond))?;

                if let State::Scalar(Scalar::Value(Value::Number(Number::Bool(cond)))) = cond {
                    if cond.into() {
                        Ok(State::Scalar(then))
                    } else {
                        Ok(State::Scalar(or_else))
                    }
                } else {
                    Err(error::bad_request(
                        "Expected a boolean condition but found",
                        cond,
                    ))
                }
            }
            Op::Ref(OpRef::Get((link, key))) => {
                let key = dereference_value_state(&provided, &key)
                    .and_then(Scalar::try_from)
                    .and_then(Value::try_from)?;

                self.inner.gateway.get(request, self, &link, key).await
            }
            Op::Method(Method::Get((tc_ref, path), key)) => {
                let subject = provided
                    .get(tc_ref.value_id())
                    .ok_or_else(|| error::not_found(tc_ref))?;

                let key = dereference_value_state(&provided, &key)
                    .and_then(Scalar::try_from)
                    .and_then(Value::try_from)?;

                println!("Method::Get subject {} {}: {}", subject, path, key);

                match subject {
                    State::Chain(chain) => {
                        println!("Txn::resolve Chain {}: {}", path, key);
                        chain.get(request, self, &path[..], key.clone()).await
                    }
                    State::Cluster(cluster) => {
                        println!("Txn::resolve Cluster {}: {}", path, key);
                        cluster
                            .get(request, &self.inner.gateway, self, &path[..], key.clone())
                            .await
                    }
                    State::Collection(collection) => {
                        println!("Txn::resolve Collection {}: {}", path, key);
                        collection
                            .get(self.clone(), &path[..], key)
                            .await
                            .map(State::from)
                    }
                    State::Object(object) => object.get(request, self, &path[..], key).await,
                    State::Scalar(scalar) => match scalar {
                        Scalar::Object(object) => object.get(request, self, &path[..], key).await,
                        Scalar::Op(op) => match &**op {
                            Op::Def(op_def) => {
                                if !&path[..].is_empty() {
                                    return Err(error::not_found(path));
                                }

                                op_def.get(request, self, key, None).await
                            }
                            other => Err(error::method_not_allowed(other)),
                        },
                        Scalar::Value(value) => value
                            .get(path.as_slice(), key.clone())
                            .map(Scalar::Value)
                            .map(State::Scalar),
                        other => Err(error::method_not_allowed(format!("GET: {}", other))),
                    },
                }
            }
            Op::Ref(OpRef::Put((link, key, value))) => {
                let value = dereference_state(&provided, &value)?;
                self.inner
                    .gateway
                    .put(&request, self, &link, key, value)
                    .await?;

                Ok(().into())
            }
            Op::Method(Method::Put((tc_ref, path), (key, value))) => {
                let subject = provided
                    .get(&tc_ref.clone().into())
                    .ok_or_else(|| error::not_found(tc_ref))?;

                let value = dereference_state(&provided, &value)?;

                println!(
                    "Txn::resolve Method::Put {}{}: {} <- {}",
                    subject, path, key, value
                );

                match subject {
                    State::Scalar(scalar) => Err(error::method_not_allowed(scalar)),
                    State::Chain(chain) => {
                        self.mutate(chain.clone().into()).await;

                        chain
                            .put(&request, self, path.as_slice(), key, value)
                            .map_ok(State::from)
                            .await
                    }
                    other => Err(error::not_implemented(format!(
                        "Txn::resolve Method::Put for {}",
                        other
                    ))),
                }
            }
            Op::Ref(OpRef::Post((link, data))) => {
                println!("Txn::resolve POST {} <- {}", link, data);

                self.inner
                    .gateway
                    .post(request, self, link, data.into())
                    .map_ok(State::from)
                    .await
            }
            Op::Method(Method::Post((subject, path), data)) => {
                let subject = provided
                    .get(&subject.clone().into())
                    .ok_or_else(|| error::not_found(subject))?;

                match subject {
                    State::Scalar(scalar) => match scalar {
                        Scalar::Op(op) => match &**op {
                            Op::Def(op_def) => {
                                if !path.as_slice().is_empty() {
                                    return Err(error::not_found(path));
                                }

                                op_def.post(request, self, data).await
                            }
                            other => Err(error::method_not_allowed(other)),
                        },
                        other => Err(error::method_not_allowed(other)),
                    },
                    _ => Err(error::not_implemented("Txn::resolve Method::Post")),
                }
            }
        }
    }

    pub async fn mutate(&self, state: State) {
        let state: Box<dyn Transact> = match state {
            State::Chain(chain) => Box::new(chain),
            State::Cluster(cluster) => Box::new(cluster),
            State::Collection(collection) => Box::new(collection),
            State::Object(_) => panic!("Objects do not support transactional mutations!"),
            State::Scalar(_) => panic!("Scalar values do not support transactional mutations!"),
        };

        self.inner.mutated.write().await.push(state)
    }
}

impl Drop for Txn {
    fn drop(&mut self) {
        // There will still be one reference in TxnServer when all others are dropped, plus this one
        if Arc::strong_count(&self.inner) == 2 {
            self.inner.txn_server.send(self.inner.id.clone()).unwrap();
        }
    }
}

fn dereference_value_state(provided: &HashMap<ValueId, State>, object: &Value) -> TCResult<State> {
    match object {
        Value::TCString(TCString::Ref(object)) => match provided.get(object.value_id()) {
            Some(state) => Ok(state.clone()),
            None => Err(error::not_found(object)),
        },
        Value::Tuple(tuple) => {
            let tuple: TCResult<Vec<State>> = tuple
                .iter()
                .map(|val| dereference_value_state(provided, val))
                .collect();

            let tuple: TCResult<Vec<Value>> = tuple?.drain(..).map(Value::try_from).collect();
            tuple
                .map(Value::Tuple)
                .map(Scalar::Value)
                .map(State::Scalar)
        }
        other => Ok(State::Scalar(Scalar::Value(other.clone()))),
    }
}

fn dereference_state(provided: &HashMap<ValueId, State>, object: &Scalar) -> TCResult<State> {
    match object {
        Scalar::Value(value) => dereference_value_state(provided, value),
        Scalar::Tuple(tuple) => {
            let tuple: TCResult<Vec<State>> = tuple
                .iter()
                .map(|val| dereference_state(provided, val))
                .collect();

            let tuple: TCResult<Vec<Scalar>> = tuple?.drain(..).map(Scalar::try_from).collect();
            tuple.map(Scalar::Tuple).map(State::Scalar)
        }
        other => Ok(State::Scalar(other.clone())),
    }
}

fn is_resolved(state: &State) -> bool {
    match state {
        State::Scalar(scalar) => is_resolved_scalar(scalar),
        _ => true,
    }
}

fn is_resolved_scalar(scalar: &Scalar) -> bool {
    match scalar {
        Scalar::Object(object) => object.data().values().all(is_resolved_scalar),
        Scalar::Op(op) => match **op {
            Op::Ref(_) => false,
            Op::Method(_) => false,
            _ => true,
        },
        Scalar::Value(value) => is_resolved_value(value),
        Scalar::Tuple(tuple) => tuple.iter().all(is_resolved_scalar),
    }
}

fn is_resolved_value(value: &Value) -> bool {
    match value {
        Value::TCString(TCString::Ref(_)) => false,
        Value::Tuple(tuple) => tuple.iter().all(is_resolved_value),
        _ => true,
    }
}

fn requires(op: &Op, txn_state: &HashMap<ValueId, State>) -> TCResult<HashSet<ValueId>> {
    let mut deps = HashSet::new();

    match op {
        Op::Def(OpDef::Get((tc_ref, _))) => {
            deps.insert(tc_ref.clone());
        }
        Op::Def(OpDef::Put((tc_ref, _, _))) => {
            deps.insert(tc_ref.clone());
        }
        Op::Def(OpDef::Post(_)) => {}
        Op::Method(method) => match method {
            Method::Get((subject, _path), key) => {
                deps.insert(subject.value_id().clone());
                deps.extend(value_requires(key)?);
            }
            Method::Put((subject, _path), (key, value)) => {
                deps.insert(subject.value_id().clone());
                deps.extend(value_requires(key)?);
                deps.extend(scalar_requires(value, txn_state)?);
            }
            Method::Post((subject, _path), params) => {
                deps.insert(subject.value_id().clone());
                deps.extend(params.data().iter().filter_map(|(name, dep)| {
                    if is_resolved_scalar(dep) {
                        None
                    } else {
                        Some(name.clone())
                    }
                }));
            }
        },
        Op::Ref(op_ref) => match op_ref {
            OpRef::If(cond, then, or_else) => {
                let cond_state = txn_state
                    .get(cond.value_id())
                    .ok_or_else(|| error::not_found(cond))?;
                if is_resolved(cond_state) {
                    if let State::Scalar(Scalar::Value(Value::Number(Number::Bool(b)))) = cond_state
                    {
                        if b.into() {
                            deps.extend(scalar_requires(then, txn_state)?);
                        } else {
                            deps.extend(scalar_requires(or_else, txn_state)?);
                        }
                    } else {
                        return Err(error::bad_request(
                            "Expected a Boolean condition but found",
                            cond_state,
                        ));
                    }
                } else {
                    deps.insert(cond.value_id().clone());
                }
            }
            OpRef::Get((_path, key)) => {
                deps.extend(value_requires(key)?);
            }
            OpRef::Put((_path, key, value)) => {
                deps.extend(value_requires(key)?);
                deps.extend(scalar_requires(value, txn_state)?);
            }
            OpRef::Post((_path, params)) => {
                for provider in params.data().values() {
                    deps.extend(scalar_requires(provider, txn_state)?);
                }
            }
        },
    }

    Ok(deps)
}

fn scalar_requires(
    scalar: &Scalar,
    txn_state: &HashMap<ValueId, State>,
) -> TCResult<HashSet<ValueId>> {
    match scalar {
        Scalar::Object(object) => {
            let mut required = HashSet::new();
            for s in object.data().values() {
                required.extend(scalar_requires(s, txn_state)?);
            }
            Ok(required)
        }
        Scalar::Op(op) => match &**op {
            Op::Def(_) => Ok(HashSet::new()),
            other => requires(other, txn_state),
        },
        Scalar::Value(value) => value_requires(value),
        Scalar::Tuple(tuple) => {
            let mut required = HashSet::new();
            for s in tuple {
                required.extend(scalar_requires(s, txn_state)?);
            }
            Ok(required)
        }
    }
}

fn value_requires(value: &Value) -> TCResult<HashSet<ValueId>> {
    match value {
        Value::TCString(TCString::Ref(tc_ref)) => {
            Ok(iter::once(tc_ref.value_id().clone()).collect())
        }
        Value::Tuple(tuple) => {
            let mut required = HashSet::new();
            for s in tuple {
                required.extend(value_requires(s)?);
            }
            Ok(required)
        }
        _ => Ok(HashSet::new()),
    }
}

fn validate_id(request: &Request, expected: &TxnId) -> TCResult<()> {
    match request.txn_id() {
        Some(txn_id) if txn_id != expected => Err(error::unsupported(format!(
            "Cannot access Transaction {} from {}",
            expected, txn_id
        ))),
        _ => Ok(()),
    }
}
