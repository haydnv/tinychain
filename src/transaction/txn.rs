use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::hash::Hash;
use std::iter;
use std::sync::{Arc, RwLock};

use futures::future::{self, try_join_all, TryFutureExt};
use futures::stream::{FuturesUnordered, Stream, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::block::dir::Dir;
use crate::class::{State, TCBoxTryFuture, TCResult};
use crate::collection::graph::Graph;
use crate::collection::schema::{GraphSchema, IndexSchema, RowSchema, TableSchema};
use crate::collection::table::{ColumnBound, Selection};
use crate::error;
use crate::gateway::{Gateway, NetworkTime};
use crate::value::class::{NumberType, StringType, ValueType};
use crate::value::link::PathSegment;
use crate::value::{label, Label, Number, Value, ValueId};

use super::Transact;

const ERR_CORRUPT: &str = "Transaction corrupted! Please file a bug report.";
const ERR_INVALID_REF: &str = "Found reference to nonexistent value";

const DEFAULT_MAX_VALUE_SIZE: usize = 100_000;
const GRAPH_NAME: Label = label(".graph");

const DEPENDS: Label = label("depends");
const NAME: Label = label("name");
const PRECEDES: Label = label("precedes");
const PROVIDER: Label = label("provider");
const REQUIRES: Label = label("requires");
const RESOLVED: Label = label("resolved");
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

    pub async fn subcontext(&self, subcontext: ValueId) -> TCResult<Arc<Txn>> {
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

    pub fn subcontext_tmp<'a>(&'a self) -> TCBoxTryFuture<'a, Arc<Txn>> {
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
    ) -> TCResult<HashMap<ValueId, State>> {
        let graph = create_graph(self.clone()).await?;
        // TODO: replace this with a Cluster when Cluster is implemented
        let mut resolved = HashMap::new();

        // while there are any unresolved parameters
        let mut done = false;
        while !done {
            done = true;

            let providers = graph
                .clone()
                .nodes(&PROVIDER.into())
                .ok_or_else(|| error::internal(ERR_CORRUPT))?
                .clone();

            // get a stream of parameters remaining to resolve
            let mut unresolved = providers
                .slice(
                    iter::once((
                        RESOLVED.into(),
                        ColumnBound::Is(Number::Bool(false.into()).into()),
                    ))
                    .collect(),
                )?
                .select(vec![NAME.into(), VALUE.into()])?
                .stream(self.id.clone())
                .await?;

            // for each one, check if all its dependencies are resolved
            let mut pending = FuturesUnordered::new();
            while let Some(mut to_resolve) = unresolved.next().await {
                let (name, value) = if to_resolve.len() == 2 {
                    (to_resolve.pop().unwrap(), to_resolve.pop().unwrap())
                } else {
                    return Err(error::internal(ERR_CORRUPT));
                };

                done = false;
                let mut ready = true;
                let mut requires = graph
                    .clone()
                    .bft(
                        self.clone(),
                        (PROVIDER.into(), vec![name.clone()]),
                        REQUIRES.into(),
                        1,
                    )
                    .await?;

                while let Some(provider) = requires.next().await {
                    let (_, mut name) = provider?;
                    let name = name.pop().ok_or_else(|| error::internal(ERR_CORRUPT))?;
                    let name: ValueId = name.try_into()?;
                    if !resolved.contains_key(&name) {
                        ready = false;
                        break;
                    }
                }

                // if so, resolve it
                if ready {
                    pending.push(
                        self.clone()
                            .resolve(resolved.clone(), value)
                            .map_ok(|state| (name, state)),
                    );
                }
            }

            // update the graph with the newly resolved states
            let mut updates = Vec::with_capacity(pending.len());
            while let Some(state) = pending.next().await {
                let (name, state) = state?;
                let update = providers
                    .slice(
                        iter::once((
                            RESOLVED.into(),
                            ColumnBound::Is(Number::Bool(false.into()).into()),
                        ))
                        .collect(),
                    )?
                    .update(
                        self.clone(),
                        iter::once((RESOLVED.into(), Number::Bool(true.into()).into())).collect(),
                    );
                updates.push(update);

                let name: ValueId = name.try_into()?;
                resolved.insert(name, state);
            }

            try_join_all(updates).await?;

            // if there are still parameters coming in, repeat the process
            if let Some(param) = parameters.next().await {
                let (name, value) = param;
                graph.add_node(
                    self.id.clone(),
                    PROVIDER.into(),
                    vec![name.into()],
                    vec![Number::Bool(false.into()).into(), value],
                ).await?;
                done = false;
            }
        }

        Ok(resolved)
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

    async fn resolve(
        self: Arc<Self>,
        _provided: HashMap<ValueId, State>,
        _value: Value,
    ) -> TCResult<State> {
        Err(error::not_implemented())
    }

    fn mutate(self: &Arc<Self>, state: Arc<dyn Transact>) {
        self.mutated.write().unwrap().push(state)
    }
}

async fn create_graph(txn: Arc<Txn>) -> TCResult<Arc<Graph>> {
    let key: RowSchema = vec![(
        NAME.into(),
        ValueType::TCString(StringType::Id),
        DEFAULT_MAX_VALUE_SIZE,
    )
        .into()];
    let values: RowSchema = vec![
        (RESOLVED.into(), ValueType::Number(NumberType::Bool)).into(),
        (VALUE.into(), ValueType::Value, DEFAULT_MAX_VALUE_SIZE).into(),
    ];
    let node_schema = IndexSchema::from((key, values));
    let node_schema = TableSchema::from((
        node_schema,
        iter::once((RESOLVED.into(), vec![RESOLVED.into()])),
    ));
    let node_schema: HashMap<ValueId, TableSchema> =
        iter::once((PROVIDER.into(), node_schema)).collect();

    let edge_schema: HashMap<ValueId, NumberType> = vec![
        (PRECEDES.into(), NumberType::Bool),
        (REQUIRES.into(), NumberType::Bool),
    ]
    .into_iter()
    .collect();

    let schema = GraphSchema::from((node_schema, edge_schema));
    Graph::create(txn, schema).await.map(Arc::new)
}
