use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::hash::Hash;
use std::slice;
use std::str::FromStr;
use std::sync::Arc;

use futures::future::{self, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, Stream, StreamExt};
use log::debug;
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
    context: Id,
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

        debug!("new Txn: {}", id);

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

    pub async fn subcontext(&self, subcontext: Id) -> TCResult<Txn> {
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

    pub async fn execute<I: Into<State>, S: Stream<Item = (Id, I)> + Unpin>(
        &self,
        request: &Request,
        mut parameters: S,
    ) -> TCResult<State> {
        validate_id(request, &self.inner.id)?;

        debug!("Txn::execute");

        let mut graph: HashMap<Id, State> = HashMap::new();
        let mut capture = None;
        while let Some((name, state)) = parameters.next().await {
            let state: State = state.into();
            debug!("pending: {}: {}", name, state);
            capture = Some(name.clone());
            graph.insert(name, state);
        }

        let capture =
            capture.ok_or_else(|| error::unsupported("Cannot execute empty operation"))?;

        let mut pending = FuturesUnordered::new();

        while !is_resolved(dereference_state(&graph, &capture)?) {
            let mut visited = HashSet::new();
            let mut unvisited = Vec::with_capacity(graph.len());
            unvisited.push(capture.clone());
            while let Some(name) = unvisited.pop() {
                if visited.contains(&name) {
                    debug!("Already visited {}", name);
                    continue;
                } else {
                    visited.insert(name.clone());
                }

                debug!("Txn::execute {} (#{})", &name, visited.len());

                let state = dereference_state(&graph, &name)?;
                if let State::Scalar(scalar) = state {
                    let mut ready = true;

                    for dep in requires(&scalar, &graph)? {
                        debug!("requires {}", dep);
                        let dep_state = dereference_state(&graph, dep.id())?;

                        if !is_resolved(dep_state) {
                            ready = false;
                            unvisited.push(dep.id().clone());
                        }
                    }

                    if ready {
                        if let Scalar::Ref(tc_ref) = scalar {
                            let tc_ref = (&**tc_ref).clone();
                            pending.push(
                                self.resolve(request, graph.clone(), tc_ref)
                                    .map(|r| (name, r)),
                            );
                        } else if let Scalar::Object(object) = scalar {
                            let object = dereference_object(&graph, object)?;
                            graph.insert(name, Scalar::Object(object).into());
                        } else if let Scalar::Tuple(tuple) = scalar {
                            let tuple = dereference_tuple(&graph, tuple)?;
                            graph.insert(name, Scalar::Tuple(tuple).into());
                        }
                    }
                }
            }

            while let Some((name, result)) = pending.next().await {
                if let Err(cause) = &result {
                    debug!("Error resolving {}: {}", name, cause);
                }

                let state = result?;
                graph.insert(name, state);
            }
        }

        debug!("Txn::execute complete, returning {}...", capture);
        graph
            .remove(&capture)
            .ok_or_else(|| error::not_found(capture))
    }

    pub async fn commit(&self) {
        debug!("commit!");

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
        debug!("rollback!");

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
        debug!("finalize!");

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

    pub fn resolve<'a>(
        &'a self,
        request: &'a Request,
        provided: HashMap<Id, State>,
        provider: TCRef,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            validate_id(request, &self.inner.id)?;

            debug!("Txn::resolve {}", provider);

            match provider {
                TCRef::Flow(control) => match *control {
                    FlowControl::If(cond, then, or_else) => {
                        let cond = self.resolve(request, provided.clone(), cond).await?;

                        if let State::Scalar(Scalar::Value(Value::Number(Number::Bool(cond)))) =
                            cond
                        {
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
                },
                TCRef::Id(id_ref) => dereference_state_owned(&provided, id_ref.id()),
                TCRef::Op(OpRef::Get((link, key))) => {
                    let key = dereference_value(&provided, key)?;
                    self.inner.gateway.get(request, self, &link, key).await
                }
                TCRef::Method(Method::Get((id_ref, path), key)) => {
                    let subject = dereference_state(&provided, id_ref.id())?;
                    let key = dereference_value(&provided, key)?;

                    match subject {
                        State::Chain(chain) => {
                            debug!("Txn::resolve Chain {}: {}", path, key);
                            chain.get(request, self, &path[..], key).await
                        }
                        State::Cluster(cluster) => {
                            debug!("Txn::resolve Cluster {}: {}", path, key);
                            cluster
                                .get(request, &self.inner.gateway, self, &path[..], key)
                                .await
                        }
                        State::Collection(collection) => {
                            debug!("Txn::resolve Collection {}: {}", path, key);
                            collection
                                .get(request, self, &path[..], key)
                                .await
                                .map(State::from)
                        }
                        State::Object(object) => object.get(request, self, &path[..], key).await,
                        State::Scalar(scalar) => match scalar {
                            Scalar::Object(object) => {
                                object.get(request, self, &path[..], key).await
                            }
                            Scalar::Op(op_def) => {
                                if !&path[..].is_empty() {
                                    return Err(error::not_found(path));
                                }

                                op_def.get(request, self, key, None).await
                            }
                            Scalar::Value(value) => value
                                .get(path.as_slice(), key.clone())
                                .map(Scalar::Value)
                                .map(State::Scalar),
                            other => Err(error::method_not_allowed(format!("GET: {}", other))),
                        },
                    }
                }
                TCRef::Op(OpRef::Put((link, key, value))) => {
                    let key = dereference_value(&provided, key)?;
                    let value = dereference(&provided, &value)?;
                    self.inner
                        .gateway
                        .put(&request, self, &link, key, value)
                        .await?;

                    Ok(().into())
                }
                TCRef::Method(Method::Put((id_ref, path), (key, value))) => {
                    let subject = dereference_state(&provided, id_ref.id())?;
                    let key = dereference_value(&provided, key)?;
                    let value = dereference(&provided, &value)?;

                    debug!(
                        "Txn::resolve Method::Put {}{}: {} <- {}",
                        subject, path, key, value
                    );

                    match subject {
                        State::Scalar(scalar) => Err(error::method_not_allowed(scalar)),
                        State::Chain(chain) => {
                            self.mutate(chain.clone().into()).await;

                            chain
                                .put(&request, self, &path[..], key, value)
                                .map_ok(State::from)
                                .await
                        }
                        State::Collection(collection) => {
                            let value = value.try_into()?;

                            collection
                                .put(&request, self, &path[..], key, value)
                                .await?;
                            Ok(State::Collection(collection.clone()))
                        }
                        other => Err(error::not_implemented(format!(
                            "Txn::resolve Method::Put for {}",
                            other
                        ))),
                    }
                }
                TCRef::Op(OpRef::Post((link, data))) => {
                    debug!("Txn::resolve POST {} <- {}", link, data);

                    self.inner
                        .gateway
                        .post(request, self, link, data.into())
                        .map_ok(State::from)
                        .await
                }
                TCRef::Method(Method::Post((id_ref, path), data)) => {
                    let subject = dereference_state(&provided, id_ref.id())?;

                    match subject {
                        State::Scalar(scalar) => match scalar {
                            Scalar::Op(op_def) => {
                                if !path.as_slice().is_empty() {
                                    return Err(error::not_found(path));
                                }

                                op_def.post(request, self, data).await
                            }
                            other => Err(error::method_not_allowed(other)),
                        },
                        _ => Err(error::not_implemented("Txn::resolve Method::Post")),
                    }
                }
            }
        })
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

fn dereference(provided: &HashMap<Id, State>, scalar: &Scalar) -> TCResult<State> {
    match scalar {
        Scalar::Ref(tc_ref) => match &**tc_ref {
            TCRef::Id(id_ref) => dereference_state_owned(provided, id_ref.id()),
            other => Err(error::not_implemented(format!("dereference {}", other))),
        },
        other => dereference_scalar(provided, other).map(State::Scalar),
    }
}

fn dereference_scalar(provided: &HashMap<Id, State>, scalar: &Scalar) -> TCResult<Scalar> {
    match scalar {
        Scalar::Object(object) => dereference_object(provided, object).map(Scalar::Object),
        Scalar::Op(op_ref) => Err(error::not_implemented(format!(
            "dereference_scalar {}",
            op_ref
        ))),
        Scalar::Ref(tc_ref) => match &**tc_ref {
            TCRef::Id(id_ref) => {
                dereference_state_owned(provided, id_ref.id()).and_then(Scalar::try_from)
            }
            other => Err(error::not_implemented(format!("dereference {}", other))),
        },
        Scalar::Tuple(tuple) => dereference_tuple(provided, tuple).map(Scalar::Tuple),
        other => Ok(other.clone()),
    }
}

fn dereference_state<'a>(provided: &'a HashMap<Id, State>, id: &'a Id) -> TCResult<&'a State> {
    provided.get(id).ok_or_else(|| error::not_found(id))
}

fn dereference_state_owned(provided: &HashMap<Id, State>, id: &Id) -> TCResult<State> {
    provided
        .get(id)
        .cloned()
        .ok_or_else(|| error::not_found(id))
}

fn dereference_object(_provided: &HashMap<Id, State>, _object: &Object) -> TCResult<Object> {
    Err(error::not_implemented("dereference_object"))
}

fn dereference_tuple(provided: &HashMap<Id, State>, tuple: &[Scalar]) -> TCResult<Vec<Scalar>> {
    let mut dereferenced = Vec::with_capacity(tuple.len());
    for item in tuple {
        dereferenced.push(dereference_scalar(provided, item)?);
    }
    Ok(dereferenced)
}

fn dereference_value(provided: &HashMap<Id, State>, key: Key) -> TCResult<Value> {
    match key {
        Key::Value(value) => Ok(value),
        Key::Ref(id_ref) => {
            dereference_state_owned(provided, id_ref.id()).and_then(Value::try_from)
        }
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
        Scalar::Object(object) => object.values().all(is_resolved_scalar),
        Scalar::Ref(_) => false,
        Scalar::Tuple(tuple) => tuple.iter().all(is_resolved_scalar),
        _ => true,
    }
}

fn requires<'a>(
    scalar: &'a Scalar,
    txn_state: &'a HashMap<Id, State>,
) -> TCResult<HashSet<&'a IdRef>> {
    let mut deps = HashSet::new();

    match scalar {
        Scalar::Ref(tc_ref) => {
            deps.extend(ref_requires(&**tc_ref, txn_state)?);
        }
        Scalar::Tuple(tuple) => {
            for item in &tuple[..] {
                deps.extend(requires(item, txn_state)?);
            }
        }
        _ => {}
    }

    Ok(deps)
}

fn ref_requires<'a>(
    tc_ref: &'a TCRef,
    txn_state: &'a HashMap<Id, State>,
) -> TCResult<HashSet<&'a IdRef>> {
    let mut deps = HashSet::new();

    match tc_ref {
        TCRef::Flow(control) => {
            deps.extend(flow_requires(&**control, txn_state)?);
        }
        TCRef::Id(id_ref) => {
            deps.insert(id_ref);
        }
        TCRef::Method(method) => {
            deps.extend(method_requires(method, txn_state)?);
        }
        TCRef::Op(op_ref) => {
            deps.extend(op_requires(op_ref, txn_state)?);
        }
    }

    Ok(deps)
}

fn flow_requires<'a>(
    control: &'a FlowControl,
    txn_state: &'a HashMap<Id, State>,
) -> TCResult<HashSet<&'a IdRef>> {
    let mut deps = HashSet::new();

    match control {
        FlowControl::If(cond, _, _) => {
            deps.extend(ref_requires(cond, txn_state)?);
        }
    }

    Ok(deps)
}

fn method_requires<'a>(
    method: &'a Method,
    txn_state: &'a HashMap<Id, State>,
) -> TCResult<HashSet<&'a IdRef>> {
    let mut deps = HashSet::new();

    match method {
        Method::Get((subject, _path), key) => {
            deps.insert(subject);

            if let Key::Ref(key) = key {
                deps.insert(key);
            }
        }
        Method::Put((subject, _path), (key, value)) => {
            deps.insert(subject);

            if let Key::Ref(key) = key {
                deps.insert(key);
            }

            deps.extend(requires(value, txn_state)?);
        }
        Method::Post((subject, _path), _params) => {
            deps.insert(subject);
        }
    }

    Ok(deps)
}

fn op_requires<'a>(
    op_ref: &'a OpRef,
    txn_state: &'a HashMap<Id, State>,
) -> TCResult<HashSet<&'a IdRef>> {
    let mut deps = HashSet::new();

    match op_ref {
        OpRef::Get((_path, key)) => {
            if let Key::Ref(tc_ref) = key {
                deps.insert(tc_ref);
            }
        }
        OpRef::Put((_path, key, value)) => {
            if let Key::Ref(tc_ref) = key {
                deps.insert(tc_ref);
            }

            deps.extend(requires(value, txn_state)?);
        }
        OpRef::Post((_path, params)) => {
            for provider in params.values() {
                deps.extend(requires(provider, txn_state)?);
            }
        }
    }

    Ok(deps)
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
