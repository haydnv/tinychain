use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use futures::{future, Stream};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::auth::Auth;
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::state::{Dir, GetResult, State};
use crate::value::link::*;
use crate::value::op::Subject;
use crate::value::*;

use super::{Transact, TxnContext};

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
    context: Arc<Dir>,
    gateway: Arc<Gateway>,
    mutated: Arc<RwLock<Vec<Arc<dyn Transact>>>>,
}

impl Txn {
    pub async fn new(gateway: Arc<Gateway>, workspace: Arc<Dir>) -> TCResult<Arc<Txn>> {
        let id = TxnId::new(Gateway::time());
        let context: PathSegment = id.clone().try_into()?;
        let context = workspace.create_dir(&id, context.into()).await?;

        Ok(Arc::new(Txn {
            id,
            context,
            gateway,
            mutated: Arc::new(RwLock::new(vec![])),
        }))
    }

    pub fn context(self: &Arc<Self>) -> Arc<Dir> {
        self.context.clone()
    }

    pub async fn subcontext(self: &Arc<Self>, subcontext: ValueId) -> TCResult<Arc<Txn>> {
        let subcontext: Arc<Dir> = self.context.create_dir(&self.id, subcontext.into()).await?;

        Ok(Arc::new(Txn {
            id: self.id.clone(),
            context: subcontext,
            gateway: self.gateway.clone(),
            mutated: self.mutated.clone(),
        }))
    }

    pub fn id(&'_ self) -> &'_ TxnId {
        &self.id
    }

    pub async fn commit(&self) {
        println!("commit!");

        future::join_all(self.mutated.write().unwrap().drain(..).map(|s| async move {
            s.commit(&self.id).await;
        }))
        .await;
    }

    pub async fn rollback(&self) {
        println!("rollback!");

        future::join_all(self.mutated.write().unwrap().drain(..).map(|s| async move {
            s.rollback(&self.id).await;
        }))
        .await;
    }

    fn mutate(self: &Arc<Self>, state: Arc<dyn Transact>) {
        self.mutated.write().unwrap().push(state)
    }

    pub async fn get(
        self: &Arc<Self>,
        subject: Subject,
        selector: Value,
        auth: &Auth,
    ) -> GetResult {
        println!("txn::get {}", subject);

        match subject {
            Subject::Ref(r) => match self.resolve(r) {
                Ok(state) => state.get(self, selector).await,
                Err(cause) => Err(cause),
            },
            Subject::Link(l) => {
                self.gateway
                    .get(&l, selector, auth, Some(self.id.clone()))
                    .await
            }
        }
    }

    pub async fn put<S: Stream<Item = (Value, Value)>>(
        self: &Arc<Self>,
        subject: Subject,
        selector: Value,
        data: State,
        auth: &Auth,
    ) -> TCResult<State> {
        println!("txn::put {}", subject);

        match subject {
            Subject::Ref(r) => match self.resolve(r) {
                Ok(state) => state.put(self, selector, data).await,
                Err(cause) => Err(cause),
            },
            Subject::Link(l) => self.gateway.put(&l, selector, data, auth).await,
        }
    }

    pub async fn post<S: Stream<Item = (ValueId, Value)>>(
        self: &Arc<Self>,
        subject: Subject,
        _op: S,
        _auth: &Auth,
    ) -> TCResult<TxnContext> {
        println!("txn::post {}", subject);

        Err(error::not_implemented())
    }

    fn resolve(&self, _id: TCRef) -> TCResult<State> {
        Err(error::not_implemented())
    }
}
