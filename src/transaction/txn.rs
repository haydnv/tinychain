use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

use futures::future;
use futures::stream::{FuturesUnordered, Stream, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::block::dir::Dir;
use crate::class::{State, TCBoxTryFuture, TCResult};
use crate::collection::graph::Graph;
use crate::collection::table;
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::value::link::PathSegment;
use crate::value::string::{StringType, TCString};
use crate::value::{label, Label, Op, Value, ValueId, ValueType};

use super::Transact;

const DEFAULT_MAX_VALUE_SIZE: usize = 1_000_000;
const GRAPH_NAME: Label = label(".graph");
const NAME: Label = label("name");
const PROVIDED: Label = label("provided");
const PROVIDER: Label = label("provider");
const VALUE: Label = label("value");

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
        if subcontext == GRAPH_NAME {
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

    pub async fn execute<S: Stream<Item = (ValueId, Value)> + Unpin>(
        self: Arc<Self>,
        mut parameters: S,
    ) -> TCResult<Graph> {
        let graph = create_graph(self.clone()).await?;

        let mut pending = FuturesUnordered::new();

        while let Some(param) = parameters.next().await {
            let (name, param) = param;
            if let Value::TCString(TCString::Ref(id)) = param {
                return Err(error::bad_request(
                    "Found reference to nonexistent value",
                    id,
                ));
            } else if let Value::Op(op) = param {
                graph
                    .add_node(
                        self.id.clone(),
                        PROVIDER.into(),
                        vec![name.clone().into()],
                        vec![Value::Op(op.clone())],
                    )
                    .await?;

                pending.push(self.clone().resolve(name, *op));
            } else {
                graph
                    .add_node(
                        self.id.clone(),
                        PROVIDED.into(),
                        vec![name.into()],
                        vec![param],
                    )
                    .await?;
            }
        }

        while let Some(result) = pending.next().await {
            let (name, state) = result?;

            //   update its status in the graph
            graph
                .remove_node(self.clone(), PROVIDER.into(), vec![name.clone().into()])
                .await?;

            match state {
                State::Value(value) => {
                    graph
                        .add_node(
                            self.id.clone(),
                            PROVIDED.into(),
                            vec![name.into()],
                            vec![value],
                        )
                        .await?;
                }
                _other => {
                    unimplemented!();
                }
            }

            //   get a list of its unresolved dependents
            //   for each of them:
            //     if all its dependencies are resolved, add a resolver future to the collection
        }

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

    async fn resolve(self: Arc<Self>, _name: ValueId, _op: Op) -> TCResult<(ValueId, State)> {
        Err(error::not_implemented())
    }

    fn mutate(self: &Arc<Self>, state: Arc<dyn Transact>) {
        self.mutated.write().unwrap().push(state)
    }
}

async fn create_graph(txn: Arc<Txn>) -> TCResult<Graph> {
    let graph_schema: HashMap<ValueId, table::Schema> = vec![
        (
            PROVIDED.into(),
            table::Schema::from((
                vec![(
                    NAME.into(),
                    ValueType::TCString(StringType::Id),
                    DEFAULT_MAX_VALUE_SIZE,
                )
                    .into()],
                vec![(VALUE.into(), ValueType::Value, DEFAULT_MAX_VALUE_SIZE).into()],
            )),
        ),
        (
            PROVIDER.into(),
            table::Schema::from((
                vec![(
                    NAME.into(),
                    ValueType::TCString(StringType::Id),
                    DEFAULT_MAX_VALUE_SIZE,
                )
                    .into()],
                vec![(VALUE.into(), ValueType::Value, DEFAULT_MAX_VALUE_SIZE).into()],
            )),
        ),
    ]
    .into_iter()
    .collect();

    Graph::create(txn, graph_schema).await
}
