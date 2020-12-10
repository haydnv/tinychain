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
use crate::class::{Public, State, TCBoxTryFuture, TCResult};
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
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

        while !is_resolved(dereference_state(&graph, &capture)?) {
            let mut pending = vec![];
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
                            let object = dereference_object(&graph, object)?;
                            graph.insert(name, Scalar::Object(object).into());
                        } else if let Scalar::Tuple(tuple) = scalar {
                            let tuple = dereference_tuple(&graph, tuple)?;
                            graph.insert(name, Scalar::Tuple(tuple).into());
                        }
                    }
                }
            }

            if pending.is_empty() && !is_resolved(dereference_state(&graph, &capture)?) {
                return Err(error::bad_request(
                    "Cannot resolve all dependencies of",
                    capture,
                ));
            }

            let current_state = graph.clone();
            let pending = pending.into_iter().map(|(name, tc_ref)| async {
                match self.subcontext(name.clone()).await {
                    Ok(cxt) => {
                        cxt.resolve(request, &current_state, tc_ref)
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

    pub async fn resolve(
        &self,
        request: &Request,
        provided: &HashMap<Id, State>,
        provider: TCRef,
    ) -> TCResult<State> {
        validate_id(request, &self.inner.id)?;

        debug!("Txn::resolve {}", provider);

        match provider {
            TCRef::Flow(control) => self.resolve_flow(request, provided, *control).await,

            TCRef::Id(id_ref) => dereference_state_owned(&provided, id_ref.id()),

            TCRef::Method(method) => match method {
                Method::Get((id_ref, path, key)) => {
                    self.resolve_get(request, provided, id_ref, &path[..], key)
                        .await
                }
                Method::Put((id_ref, path, key, value)) => {
                    self.resolve_put(request, provided, id_ref, &path[..], key, value)
                        .map_ok(State::from)
                        .await
                }
                Method::Post((id_ref, path, params)) => {
                    self.resolve_post(request, provided, id_ref, &path[..], params)
                        .await
                }

                Method::Delete((id_ref, path, key)) => {
                    self.resolve_delete(request, provided, id_ref, &path[..], key)
                        .map_ok(State::from)
                        .await
                }
            },

            TCRef::Op(op_ref) => match op_ref {
                OpRef::Get((link, key)) => {
                    let key = dereference_value(&provided, key)?;
                    self.inner.gateway.get(request, self, &link, key).await
                }

                OpRef::Put((link, key, value)) => {
                    let key = dereference_value(&provided, key)?;
                    let value = dereference(&provided, &value)?;
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
                    let key = dereference_value(&provided, key)?;
                    self.inner
                        .gateway
                        .delete(request, self, &link, key)
                        .map_ok(State::from)
                        .await
                }
            },
        }
    }

    pub fn resolve_object<'a>(
        &'a self,
        request: &'a Request,
        provided: &'a HashMap<Id, State>,
        object: Object,
    ) -> impl Stream<Item = TCResult<(Id, State)>> + 'a {
        FuturesUnordered::from_iter(object.into_iter().map(|(id, scalar)| {
            let provider: TCBoxTryFuture<State> = match scalar {
                Scalar::Ref(tc_ref) => {
                    Box::pin(async move { self.resolve(request, provided, *tc_ref).await })
                }
                other => Box::pin(async move { Ok(State::Scalar(other)) }),
            };

            provider.map_ok(|state| (id, state))
        }))
    }

    fn resolve_flow<'a>(
        &'a self,
        request: &'a Request,
        provided: &'a HashMap<Id, State>,
        control: FlowControl,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            match control {
                FlowControl::After((when, then)) => {
                    let when = when
                        .into_iter()
                        .map(|tc_ref| self.resolve(request, provided, tc_ref));

                    try_join_all(when).await?;
                    Ok(State::Scalar(Scalar::Ref(Box::new(then))))
                }
                FlowControl::If((cond, then, or_else)) => {
                    let cond = self.resolve(request, provided, cond).await?;

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
            }
        })
    }

    async fn resolve_get(
        &self,
        request: &Request,
        provided: &HashMap<Id, State>,
        subject: IdRef,
        path: &[PathSegment],
        key: Key,
    ) -> TCResult<State> {
        debug!("Txn::resolve {}::GET {}", subject, key);

        let subject = dereference_state(&provided, subject.id())?;
        let key = dereference_value(&provided, key)?;

        debug!("Txn::resolve {}::GET {}", subject, key);

        match subject {
            State::Chain(chain) => chain.get(request, self, path, key).await,
            State::Cluster(cluster) => cluster.get(request, self, path, key).await,
            State::Collection(collection) => {
                collection
                    .get(request, self, &path[..], key)
                    .map_ok(State::from)
                    .await
            }
            State::Object(object) => object.get(request, self, path, key).await,
            State::Scalar(scalar) => match scalar {
                Scalar::Op(op_def) => {
                    if !&path[..].is_empty() {
                        return Err(error::path_not_found(path));
                    }

                    op_def
                        .get(request, self, key, Some(subject.clone().into()))
                        .await
                }
                other => other.get(request, self, &path[..], key).await,
            },
        }
    }

    async fn resolve_put(
        &self,
        request: &Request,
        provided: &HashMap<Id, State>,
        subject: IdRef,
        path: &[PathSegment],
        key: Key,
        value: Scalar,
    ) -> TCResult<()> {
        let subject = dereference_state(&provided, subject.id())?;
        let key = dereference_value(&provided, key)?;
        let value = dereference(&provided, &value)?;

        match subject {
            State::Chain(chain) => chain.put(request, self, &path[..], key, value).await,
            State::Cluster(cluster) => cluster.put(request, self, &path[..], key, value).await,
            State::Collection(collection) => {
                collection.put(request, self, &path[..], key, value).await
            }
            State::Object(object) => object.put(request, self, path, key, value).await,
            State::Scalar(scalar) => match scalar {
                Scalar::Op(op_def) if path.len() == 0 => {
                    op_def
                        .put(request, self, key, value, Some(subject.clone().into()))
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
        provided: &HashMap<Id, State>,
        subject: IdRef,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        let subject = dereference_state(&provided, subject.id())?;

        match subject {
            State::Chain(chain) => chain.post(request, self, path, params).await,
            State::Cluster(cluster) => cluster.post(request, self, path, params).await,
            State::Collection(collection) => collection.post(request, self, path, params).await,
            State::Object(object) => object.post(request, self, path, params).await,
            State::Scalar(scalar) => match scalar {
                Scalar::Op(op_def) if path.is_empty() => {
                    op_def
                        .post(request, self, params, Some(subject.clone().into()))
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
        provided: &HashMap<Id, State>,
        subject: IdRef,
        path: &[PathSegment],
        key: Key,
    ) -> TCResult<()> {
        debug!("Txn::resolve {}::GET {}", subject, key);

        let subject = dereference_state(&provided, subject.id())?;
        let key = dereference_value(&provided, key)?;

        debug!("Txn::resolve {}::GET {}", subject, key);

        match subject {
            State::Chain(chain) => chain.delete(request, self, path, key).await,
            State::Cluster(cluster) => cluster.delete(request, self, path, key).await,
            State::Collection(collection) => collection.delete(request, self, &path[..], key).await,
            State::Object(object) => object.delete(request, self, path, key).await,
            State::Scalar(scalar) => match scalar {
                Scalar::Op(op_def) if path.is_empty() => {
                    op_def
                        .delete(request, self, key, Some(subject.clone().into()))
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
            _ => dereference_scalar(provided, scalar).map(State::Scalar),
        },
        _ => dereference_scalar(provided, scalar).map(State::Scalar),
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
            other => Ok(Scalar::Ref(Box::new(other.clone()))),
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
    txn_state: &'a HashMap<Id, State>,
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
