use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt;
use std::hash::Hash;
use std::iter::FromIterator;
use std::slice;
use std::str::FromStr;
use std::sync::Arc;

use futures::future::{self, try_join_all, FutureExt, TryFutureExt};
use futures::stream::{FuturesUnordered, Stream, StreamExt};
use log::debug;
use rand::Rng;
use serde::de;
use tokio::sync::mpsc;

use crate::block::{BlockData, Dir, DirEntry, File};
use crate::class::{State, TCBoxTryFuture, TCResult};
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::handler::Public;
use crate::lock::RwLock;
use crate::request::Request;
use crate::scalar::*;

use super::Transact;

const INVALID_ID: &str = "Invalid transaction ID";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
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

#[derive(Clone)]
struct TxnState {
    context: Option<State>,
    provided: HashMap<Id, State>,
}

impl TxnState {
    fn new(context: Option<State>) -> Self {
        let provided = HashMap::new();
        Self { context, provided }
    }

    fn dereference(&self, scalar: Scalar) -> TCResult<State> {
        match scalar {
            Scalar::Ref(ref tc_ref) => match &**tc_ref {
                TCRef::Id(id_ref) => self.dereference_state_owned(id_ref.id()),
                _ => self.dereference_scalar(scalar).map(State::Scalar),
            },
            _ => self.dereference_scalar(scalar).map(State::Scalar),
        }
    }

    fn dereference_state(&'_ self, id: &'_ Id) -> TCResult<&'_ State> {
        if let Some(state) = self.provided.get(id) {
            Ok(state)
        } else if id == "self" {
            if let Some(context) = &self.context {
                Ok(context)
            } else {
                Err(error::not_found(id))
            }
        } else {
            self.provided.get(id).ok_or_else(|| error::not_found(id))
        }
    }

    fn dereference_state_owned(&self, id: &Id) -> TCResult<State> {
        self.dereference_state(id).map(Clone::clone)
    }

    fn dereference_object(&self, _object: &Object) -> TCResult<Object> {
        Err(error::not_implemented("TxnState::dereference_object"))
    }

    fn dereference_scalar(&'_ self, scalar: Scalar) -> TCResult<Scalar> {
        match scalar {
            Scalar::Object(object) => self.dereference_object(&object).map(Scalar::Object),
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => self
                    .dereference_state_owned(id_ref.id())
                    .and_then(Scalar::try_from),
                _ => Ok(Scalar::Ref(tc_ref)),
            },
            Scalar::Tuple(tuple) => self.dereference_tuple(tuple).map(Scalar::Tuple),
            other => Ok(other),
        }
    }

    fn dereference_tuple(&'_ self, tuple: Vec<Scalar>) -> TCResult<Vec<Scalar>> {
        let mut dereferenced = Vec::with_capacity(tuple.len());
        for item in tuple {
            dereferenced.push(self.dereference_scalar(item)?);
        }
        Ok(dereferenced)
    }

    fn dereference_value(&'_ self, key: Key) -> TCResult<Value> {
        match key {
            Key::Value(value) => Ok(value),
            Key::Ref(id_ref) => match self.dereference_state(id_ref.id())? {
                State::Scalar(Scalar::Value(value)) => Ok(value.clone()),
                other => Err(error::bad_request("Expected Value but found", other)),
            },
        }
    }
}

struct Inner {
    id: TxnId,
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
        dir: Arc<Dir>,
        id: TxnId,
        txn_server: mpsc::UnboundedSender<TxnId>,
    ) -> TCResult<Txn> {
        let context = id.to_path();

        debug!("new Txn: {}", id);

        let inner = Arc::new(Inner {
            id,
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
            .create_file(self.inner.id, self.inner.context.clone())
            .await
    }

    pub async fn subcontext(&self, subcontext: Id) -> TCResult<Txn> {
        let dir = self
            .inner
            .dir
            .get_or_create_dir(&self.inner.id, slice::from_ref(&self.inner.context))
            .await?;

        let subcontext = Arc::new(Inner {
            id: self.inner.id,
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

        let mut graph = TxnState::new(None);
        let mut capture = None;
        while let Some((name, state)) = parameters.next().await {
            let state: State = state.into();
            debug!("pending: {}: {}", name, state);
            capture = Some(name.clone());
            graph.provided.insert(name, state);
        }

        let capture =
            capture.ok_or_else(|| error::unsupported("Cannot execute empty operation"))?;

        while !is_resolved(graph.dereference_state(&capture)?) {
            let mut pending = vec![];
            let mut visited = HashSet::new();
            let mut unvisited = Vec::with_capacity(graph.provided.len());
            unvisited.push(capture.clone());
            while let Some(name) = unvisited.pop() {
                if visited.contains(&name) {
                    debug!("Already visited {}", name);
                    continue;
                } else {
                    visited.insert(name.clone());
                }

                debug!("Txn::execute {} (#{})", &name, visited.len());

                let state = graph.dereference_state(&name)?;
                if let State::Scalar(scalar) = state {
                    let mut ready = true;

                    for dep in requires(&scalar, &graph)? {
                        debug!("requires {}", dep);
                        let dep_state = graph.dereference_state(dep.id())?;

                        if !is_resolved(dep_state) {
                            debug!("{} is not resolved (state is {})", dep.id(), dep_state);
                            ready = false;
                            unvisited.push(dep.id().clone());
                        }
                    }

                    if ready {
                        if let Scalar::Ref(tc_ref) = scalar {
                            let tc_ref = (&**tc_ref).clone();
                            pending.push((name, tc_ref));
                        } else if let Scalar::Object(object) = scalar {
                            let object = graph.dereference_object(object)?;
                            graph.provided.insert(name, Scalar::Object(object).into());
                        } else if let Scalar::Tuple(tuple) = scalar {
                            let tuple = Scalar::Tuple(graph.dereference_tuple(tuple.to_vec())?);
                            graph.provided.insert(name, State::Scalar(tuple));
                        }
                    }
                }
            }

            if pending.is_empty() && !is_resolved(graph.dereference_state(&capture)?) {
                return Err(error::bad_request(
                    "Cannot resolve all dependencies of",
                    capture,
                ));
            }

            let current_state = graph.clone();
            let pending = pending.into_iter().map(|(name, tc_ref)| async {
                match self.subcontext(name.clone()).await {
                    Ok(cxt) => {
                        cxt.resolve_inner(request, &current_state, tc_ref)
                            .map(|r| (name, r))
                            .await
                    }
                    Err(cause) => (name, Err(cause)),
                }
            });
            let mut pending = FuturesUnordered::from_iter(pending);
            while let Some((name, result)) = pending.next().await {
                if let Err(cause) = &result {
                    debug!("Error resolving {}: {}", name, cause);
                } else {
                    debug!("resolved {}", name);
                }

                let state = result?;
                graph.provided.insert(name, state);
            }
        }

        debug!("Txn::execute complete, returning {}...", capture);
        graph
            .provided
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
    }

    pub async fn resolve(&self, request: &Request, provider: TCRef) -> TCResult<State> {
        let graph = TxnState::new(None);
        self.resolve_inner(request, &graph, provider).await
    }

    async fn resolve_inner(
        &self,
        request: &Request,
        graph: &TxnState,
        provider: TCRef,
    ) -> TCResult<State> {
        validate_id(request, &self.inner.id)?;

        debug!("Txn::resolve {}", provider);

        match provider {
            TCRef::Flow(control) => self.resolve_flow(request, graph, *control).await,

            TCRef::Id(id_ref) => graph.dereference_state_owned(id_ref.id()),

            TCRef::Method(method) => match method {
                Method::Get((id_ref, path, key)) => {
                    self.resolve_get(request, graph, id_ref, &path[..], key)
                        .await
                }
                Method::Put((id_ref, path, key, value)) => {
                    self.resolve_put(request, graph, id_ref, &path[..], key, value)
                        .map_ok(State::from)
                        .await
                }
                Method::Post((id_ref, path, params)) => {
                    self.resolve_post(request, graph, id_ref, &path[..], params)
                        .await
                }

                Method::Delete((id_ref, path, key)) => {
                    self.resolve_delete(request, graph, id_ref, &path[..], key)
                        .map_ok(State::from)
                        .await
                }
            },

            TCRef::Op(op_ref) => match op_ref {
                OpRef::Get((link, key)) => {
                    let key = graph.dereference_value(key)?;
                    self.inner.gateway.get(request, self, &link, key).await
                }

                OpRef::Put((link, key, value)) => {
                    let key = graph.dereference_value(key)?;
                    let value = graph.dereference(value)?;
                    self.inner
                        .gateway
                        .put(&request, self, &link, key, value)
                        .await?;

                    Ok(().into())
                }

                OpRef::Post((link, data)) => {
                    debug!("Txn::resolve POST {} <- {}", link, data);

                    self.inner
                        .gateway
                        .post(request, self, link, data.into())
                        .map_ok(State::from)
                        .await
                }

                OpRef::Delete((link, key)) => {
                    let key = graph.dereference_value(key)?;
                    self.inner
                        .gateway
                        .delete(request, self, &link, key)
                        .map_ok(State::from)
                        .await
                }
            },
        }
    }

    fn resolve_flow<'a>(
        &'a self,
        request: &'a Request,
        graph: &'a TxnState,
        control: FlowControl,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            match control {
                FlowControl::After((when, then)) => {
                    let when = when
                        .into_iter()
                        .map(|tc_ref| self.resolve_inner(request, graph, tc_ref));

                    try_join_all(when).await?;
                    Ok(State::Scalar(Scalar::Ref(Box::new(then.clone()))))
                }
                FlowControl::If((cond, then, or_else)) => {
                    let cond = self.resolve_inner(request, graph, cond).await?;

                    if let State::Scalar(Scalar::Value(Value::Number(Number::Bool(cond)))) = cond {
                        if cond.into() {
                            Ok(State::Scalar(then.clone()))
                        } else {
                            Ok(State::Scalar(or_else.clone()))
                        }
                    } else {
                        Err(error::bad_request(
                            "Expected a boolean condition but found",
                            cond,
                        ))
                    }
                }
            }
        })
    }

    async fn resolve_get(
        &self,
        request: &Request,
        graph: &TxnState,
        subject: IdRef,
        path: &[PathSegment],
        key: Key,
    ) -> TCResult<State> {
        debug!("Txn::resolve {}::GET {}", subject, key);

        let subject = graph.dereference_state(subject.id())?;
        let key = graph.dereference_value(key)?;

        debug!("Txn::resolve {}::GET {}", subject, key);

        match subject {
            State::Chain(chain) => chain.get(request, self, path, key).await,
            State::Collection(collection) => {
                collection
                    .get(request, self, &path[..], key)
                    .map_ok(State::from)
                    .await
            }
            State::Object(object) => object.get(request, self, path, key).await,
            State::Scalar(scalar) => match scalar {
                Scalar::Op(op_def) if path.is_empty() => {
                    op_def
                        .route(request, Some(subject.clone()))
                        .get(request, self, key)
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(path)),
                other => other.get(request, self, &path[..], key).await,
            },
        }
    }

    async fn resolve_put(
        &self,
        request: &Request,
        graph: &TxnState,
        subject: IdRef,
        path: &[PathSegment],
        key: Key,
        value: Scalar,
    ) -> TCResult<()> {
        let subject = graph.dereference_state(subject.id())?;
        let key = graph.dereference_value(key)?;
        let value = graph.dereference(value)?;

        match subject {
            State::Chain(chain) => chain.put(request, self, &path[..], key, value).await,
            State::Collection(collection) => {
                collection.put(request, self, &path[..], key, value).await
            }
            State::Object(object) => object.put(request, self, path, key, value).await,
            State::Scalar(scalar) => match scalar {
                Scalar::Op(op_def) if path.len() == 0 => {
                    op_def
                        .route(request, Some(subject.clone()))
                        .put(request, self, key, value)
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(path)),
                other => other.put(request, self, path, key, value).await,
            },
        }
    }

    async fn resolve_post(
        &self,
        request: &Request,
        graph: &TxnState,
        subject: IdRef,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        let subject = graph.dereference_state(subject.id())?;

        match subject {
            State::Chain(chain) => chain.post(request, self, path, params).await,
            State::Collection(collection) => collection.post(request, self, path, params).await,
            State::Object(object) => object.post(request, self, path, params).await,
            State::Scalar(scalar) => match scalar {
                Scalar::Op(op_def) if path.is_empty() => {
                    op_def
                        .route(request, Some(subject.clone()))
                        .post(request, self, params)
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(path)),
                other => other.post(request, self, path, params).await,
            },
        }
    }

    async fn resolve_delete(
        &self,
        request: &Request,
        graph: &TxnState,
        subject: IdRef,
        path: &[PathSegment],
        key: Key,
    ) -> TCResult<()> {
        debug!("Txn::resolve {}::GET {}", subject, key);

        let subject = graph.dereference_state(subject.id())?;
        let key = graph.dereference_value(key)?;

        debug!("Txn::resolve {}::GET {}", subject, key);

        match subject {
            State::Chain(chain) => chain.delete(request, self, path, key).await,
            State::Collection(collection) => collection.delete(request, self, &path[..], key).await,
            State::Object(object) => object.delete(request, self, path, key).await,
            State::Scalar(scalar) => match scalar {
                Scalar::Op(op_def) if path.is_empty() => {
                    op_def
                        .route(request, Some(subject.clone()))
                        .delete(request, self, key)
                        .await
                }
                Scalar::Op(_) => Err(error::path_not_found(path)),
                other => other.delete(request, self, path, key).await,
            },
        }
    }

    pub async fn mutate(&self, state: State) {
        let state: Box<dyn Transact> = match state {
            State::Chain(chain) => Box::new(chain),
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

fn requires<'a>(scalar: &'a Scalar, txn_state: &'a TxnState) -> TCResult<HashSet<&'a IdRef>> {
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

fn ref_requires<'a>(tc_ref: &'a TCRef, txn_state: &'a TxnState) -> TCResult<HashSet<&'a IdRef>> {
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
    txn_state: &'a TxnState,
) -> TCResult<HashSet<&'a IdRef>> {
    let mut deps = HashSet::new();

    match control {
        FlowControl::After((when, _)) => {
            for tc_ref in when {
                deps.extend(ref_requires(tc_ref, txn_state)?);
            }
        }
        FlowControl::If((cond, _, _)) => {
            deps.extend(ref_requires(cond, txn_state)?);
        }
    }

    Ok(deps)
}

fn method_requires<'a>(
    method: &'a Method,
    txn_state: &'a TxnState,
) -> TCResult<HashSet<&'a IdRef>> {
    let mut deps = HashSet::new();

    match method {
        Method::Get((subject, _path, key)) => {
            deps.insert(subject);

            if let Key::Ref(tc_ref) = key {
                deps.insert(tc_ref);
            }
        }
        Method::Put((subject, _path, key, value)) => {
            deps.insert(subject);

            if let Key::Ref(key) = key {
                deps.insert(key);
            }

            deps.extend(requires(value, txn_state)?);
        }
        Method::Post((subject, _path, _params)) => {
            deps.insert(subject);
        }
        Method::Delete((subject, _path, key)) => {
            deps.insert(subject);

            if let Key::Ref(tc_ref) = key {
                deps.insert(tc_ref);
            }
        }
    }

    Ok(deps)
}

fn op_requires<'a>(op_ref: &'a OpRef, txn_state: &'a TxnState) -> TCResult<HashSet<&'a IdRef>> {
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
        OpRef::Delete((_path, key)) => {
            if let Key::Ref(tc_ref) = key {
                deps.insert(tc_ref);
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
