use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::iter::{self, FromIterator};
use std::slice;
use std::str::FromStr;
use std::sync::Arc;

use futures::future::{self, TryFutureExt};
use futures::stream::{FuturesUnordered, StreamExt};
use futures::try_join;
use log::debug;
use rand::Rng;
use serde::de;
use tokio::sync::mpsc;

use crate::block::{BlockData, Dir, DirEntry, File};
use crate::class::State;
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::lock::RwLock;
use crate::request::Request;
use crate::scalar::*;
use crate::{TCBoxTryFuture, TCResult};

use super::Transact;

pub type Graph = HashMap<Id, State>;

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

    pub fn id(&'_ self) -> &'_ TxnId {
        &self.inner.id
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

    pub async fn execute<I: IntoIterator<Item = (Id, Scalar)>>(
        &self,
        request: &Request,
        mut graph: Graph,
        program: I,
    ) -> TCResult<State> {
        let mut program: VecDeque<(Id, Scalar)> = program.into_iter().collect();
        if program.is_empty() {
            return Ok(().into());
        }

        let (capture, _) = program.back().unwrap().clone();

        let mut required: HashSet<Id> = iter::once(capture.clone()).collect();

        while !graph.contains_key(&capture) {
            let mut resolved = vec![];

            {
                debug!(
                    "program is {}",
                    Value::from_iter(program.iter().map(|(id, _)| id.clone()))
                );

                let mut pending = FuturesUnordered::new();
                let mut reduced = VecDeque::with_capacity(program.len());
                while let Some((id, scalar)) = program.pop_back() {
                    debug!("resolving {}...", id);

                    if graph.contains_key(&id) {
                        return Err(error::bad_request("There is already a value for", id));
                    }

                    if !scalar.is_ref() {
                        debug!("no resolution needed for {}: {}", id, scalar);

                        resolved.push((id, scalar.into()));
                    } else if required.contains(&id) {
                        let mut deps = HashSet::new();
                        scalar.requires(&mut deps);

                        let provided: HashSet<Id> = graph.keys().cloned().collect();
                        if let Some(dep) = deps.difference(&provided).next() {
                            debug!("unmet dependency: {}", dep);

                            required.extend(deps);
                            reduced.push_front((id, scalar));
                        } else {
                            debug!("all dependencies satisfied for {}: {}", id, scalar);

                            let provider = async {
                                let subcontext = self.subcontext(id.clone()).await?;
                                scalar
                                    .resolve(request, &subcontext, &graph)
                                    .map_ok(|state| (id, state))
                                    .await
                            };

                            pending.push(provider);
                        }
                    } else {
                        debug!("{} is not yet needed", id);

                        reduced.push_front((id, scalar));
                    }
                }
                program = reduced;

                if resolved.is_empty() && pending.is_empty() {
                    return Err(error::bad_request(
                        "Could not resolve all dependencies",
                        Value::from_iter(required),
                    ));
                }

                while let Some(result) = pending.next().await {
                    let (id, state) = result?;

                    match state {
                        State::Scalar(scalar) if scalar.is_ref() => {
                            debug!("{} is now {}", id, scalar);
                            program.push_back((id, scalar));
                        }
                        state => {
                            debug!("{} is now {}", id, state);
                            resolved.push((id, state));
                        }
                    }
                }
            }

            graph.extend(resolved);
        }

        graph
            .remove(&capture)
            .ok_or_else(|| error::not_found(capture))
    }

    pub async fn resolve_op(
        &self,
        request: &Request,
        context: &HashMap<Id, State>,
        op_ref: OpRef,
    ) -> TCResult<State> {
        validate_id(request, self.id())?;

        match op_ref {
            OpRef::Get((link, key)) => {
                let key = key.resolve_value(request, self, context).await?;
                self.inner.gateway.get(request, self, &link, key).await
            }
            OpRef::Put((link, key, value)) => {
                let (key, value) = try_join!(
                    key.resolve_value(request, self, context),
                    value.resolve(request, self, context)
                )?;
                self.inner
                    .gateway
                    .put(request, self, &link, key, value)
                    .map_ok(State::from)
                    .await
            }
            OpRef::Post((link, params)) => {
                let params = params.resolve(request, self, context).await?;
                if let State::Scalar(params) = params {
                    self.inner.gateway.post(request, self, link, params).await
                } else {
                    Err(error::not_implemented(format!(
                        "POST with params {}",
                        params
                    )))
                }
            }
            OpRef::Delete((link, key)) => {
                let key = key.resolve_value(request, self, context).await?;
                self.inner
                    .gateway
                    .delete(request, self, &link, key)
                    .map_ok(State::from)
                    .await
            }
        }
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

    pub async fn mutate(&self, state: State) {
        let state: Box<dyn Transact> = match state {
            State::Chain(chain) => Box::new(chain),
            State::Collection(collection) => Box::new(collection),
            other => panic!("{} does not support transactional mutations!", other),
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

fn validate_id(request: &Request, expected: &TxnId) -> TCResult<()> {
    match request.txn_id() {
        Some(txn_id) if txn_id != expected => Err(error::unsupported(format!(
            "Cannot access Transaction {} from {}",
            expected, txn_id
        ))),
        _ => Ok(()),
    }
}
