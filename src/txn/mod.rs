use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use rand::Rng;
use serde::de;

use error::*;
use generic::{Id, NetworkTime, PathSegment};

use crate::state::State;

pub mod lock;
mod request;

pub use request::Request;

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
pub struct Txn {
    id: TxnId,
    state: HashMap<Id, State>,
}

impl Txn {
    pub fn new<I: IntoIterator<Item = (Id, State)>>(data: I, id: TxnId) -> Self {
        let state = data.into_iter().collect();
        Self { id, state }
    }

    pub async fn execute(&mut self, capture: Id) -> TCResult<State> {
        self.state
            .remove(&capture)
            .ok_or_else(|| TCError::not_found(capture))
    }
}
