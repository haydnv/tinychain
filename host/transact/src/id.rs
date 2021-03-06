//! [`TxnId`]

use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use rand::Rng;

use destream::IntoStream;
use tc_error::*;
use tcgeneric::{Id, NetworkTime};

const INVALID_ID: &str = "Invalid transaction ID";

pub const MIN_ID: TxnId = TxnId {
    timestamp: 0,
    nonce: 0,
};

/// The unique ID of a transaction, used for identity and ordering.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TxnId {
    timestamp: u64, // nanoseconds since Unix epoch
    nonce: u16,
}

impl TxnId {
    /// Construct a new `TxnId`.
    pub fn new(time: NetworkTime) -> TxnId {
        TxnId {
            timestamp: time.as_nanos(),
            nonce: rand::thread_rng().gen(),
        }
    }

    /// Return the timestamp of this `TxnId`.
    pub fn time(&self) -> NetworkTime {
        NetworkTime::from_nanos(self.timestamp)
    }

    /// Convert this `TxnId` into an [`Id`].
    pub fn to_id(&self) -> Id {
        self.to_string().parse().unwrap()
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

#[async_trait]
impl destream::de::FromStream for TxnId {
    type Context = ();

    async fn from_stream<D: destream::de::Decoder>(
        context: (),
        d: &mut D,
    ) -> Result<Self, D::Error> {
        let s = <String as destream::de::FromStream>::from_stream(context, d).await?;
        Self::from_str(&s).map_err(destream::de::Error::custom)
    }
}

impl<'en> destream::en::IntoStream<'en> for TxnId {
    fn into_stream<E: destream::en::Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        self.to_string().into_stream(e)
    }
}

impl<'en> destream::en::ToStream<'en> for TxnId {
    fn to_stream<E: destream::en::Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        self.to_string().into_stream(e)
    }
}

impl fmt::Display for TxnId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.nonce)
    }
}
