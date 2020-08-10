use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use futures::future;
use futures::stream::Stream;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::block::dir::Dir;
use crate::class::{TCBoxTryFuture, TCResult};
use crate::collection::graph::Graph;
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::value::link::*;
use crate::value::*;

use super::Transact;

const GRAPH_NAME: &str = ".graph";

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
        let context = workspace.create_dir(&id, &context.into()).await?;

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

    pub async fn subcontext(self: Arc<Self>, subcontext: ValueId) -> TCResult<Arc<Txn>> {
        if subcontext.as_str() == GRAPH_NAME {
            return Err(error::bad_request("This name is reserved", subcontext));
        }

        let subcontext: Arc<Dir> = self
            .context
            .create_dir(&self.id, &subcontext.into())
            .await?;

        Ok(Arc::new(Txn {
            id: self.id.clone(),
            context: subcontext,
            gateway: self.gateway.clone(),
            mutated: self.mutated.clone(),
        }))
    }

    pub fn subcontext_tmp<'a>(self: Arc<Self>) -> TCBoxTryFuture<'a, Arc<Txn>> {
        Box::pin(async move {
            let id = self.context.unique_id(self.id()).await?;
            let subcontext: Arc<Dir> = self.context.create_dir(&self.id, &id.into()).await?;

            Ok(Arc::new(Txn {
                id: self.id.clone(),
                context: subcontext,
                gateway: self.gateway.clone(),
                mutated: self.mutated.clone(),
            }))
        })
    }

    pub fn id(&'_ self) -> &'_ TxnId {
        &self.id
    }

    pub async fn execute<S: Stream<Item = (ValueId, Value)>>(
        &self,
        _parameters: S,
    ) -> TCResult<Graph> {
        // instantiate a graph with two node classes: providers (Values) and provided (States)
        // instantiate a new FuturesUnordered to execute the transaction's sub-tasks
        // for each parameter:
        //   if it's already resolved, add it to the graph
        //   if all its dependencies are resolved, add a resolver future to the collection
        //   otherwise (if it depends on an unresolved parameter) add it to the graph

        // while there are unresolved futures in the collection:
        // for each resolved parameter:
        //   update its status in the graph
        //   get a list of its unresolved dependents
        //   for each of them:
        //     if all its dependencies are resolved, add a resolver future to the collection

        // if there are still unresolved parameters in the graph, return an error
        // otherwise, return the graph

        Err(error::not_implemented())
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
}
