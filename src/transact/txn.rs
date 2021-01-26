use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
use std::str::FromStr;

use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use rand::Rng;
use serde::de;

use error::*;
use generic::{Id, NetworkTime, PathSegment};

use super::{Refer, State};

const INVALID_ID: &str = "Invalid transaction ID";

pub const MIN_ID: TxnId = TxnId {
    timestamp: 0,
    nonce: 0,
};

#[async_trait]
pub trait Transact {
    async fn commit(&self, txn_id: &TxnId);

    async fn finalize(&self, txn_id: &TxnId);
}

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
}

impl FromStr for TxnId {
    type Err = TCError;

    fn from_str(s: &str) -> TCResult<TxnId> {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() == 2 {
            let timestamp = parts[0]
                .parse()
                .map_err(|e| TCError::bad_request(INVALID_ID, e))?;

            let nonce = parts[1]
                .parse()
                .map_err(|e| TCError::bad_request(INVALID_ID, e))?;

            Ok(TxnId { timestamp, nonce })
        } else {
            Err(TCError::bad_request(INVALID_ID, s))
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

impl From<&'_ TxnId> for PathSegment {
    fn from(txn_id: &'_ TxnId) -> Self {
        txn_id.to_string().parse().unwrap()
    }
}

impl fmt::Display for TxnId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.nonce)
    }
}

#[derive(Clone)]
pub struct Txn<S> {
    id: TxnId,
    state: HashMap<Id, S>,
}

impl<S: Refer + State> Txn<S> {
    pub fn new<I: IntoIterator<Item = (Id, S)>>(data: I, id: TxnId) -> Self {
        let state = data.into_iter().collect();
        Self { id, state }
    }

    pub async fn execute(&mut self, capture: Id) -> TCResult<S> {
        while self.resolve_id(&capture)?.is_ref() {
            let mut pending = Vec::with_capacity(self.state.len());
            let mut unvisited = Vec::with_capacity(self.state.len());
            unvisited.push(capture.clone());

            while let Some(id) = unvisited.pop() {
                let state = self.resolve_id(&capture)?;

                if state.is_ref() {
                    let mut deps = HashSet::new();
                    state.requires(&mut deps);

                    let mut ready = true;
                    for dep_id in deps.into_iter() {
                        if self.resolve_id(&dep_id)?.is_ref() {
                            ready = false;
                            unvisited.push(dep_id);
                        }
                    }

                    if ready {
                        pending.push(id);
                    }
                }
            }

            if pending.is_empty() && self.resolve_id(&capture)?.is_ref() {
                return Err(TCError::bad_request(
                    "Cannot resolve all dependencies of",
                    capture,
                ));
            }

            let mut providers = FuturesUnordered::from_iter(
                pending
                    .into_iter()
                    .map(|id| async { (id, Err(TCError::not_implemented("State::resolve"))) }),
            );

            while let Some((id, r)) = providers.next().await {
                match r {
                    Ok(state) => {
                        self.state.insert(id, state);
                    }
                    Err(cause) => return Err(cause.consume(format!("Error resolving {}", id))),
                }
            }
        }

        self.state
            .remove(&capture)
            .ok_or_else(|| TCError::not_found(capture))
    }

    pub fn resolve_id(&'_ self, id: &Id) -> TCResult<&'_ S> {
        self.state.get(id).ok_or_else(|| TCError::not_found(id))
    }
}
