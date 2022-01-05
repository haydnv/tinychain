//! A transaction ID

use std::fmt;
use std::str::FromStr;

use async_hash::Hash;
use async_trait::async_trait;
use rand::Rng;

use destream::IntoStream;
use sha2::digest::{Digest, Output};
use tc_error::*;
use tcgeneric::{Id, NetworkTime};

const INVALID_ID: &str = "Invalid transaction ID";

/// A zero-values [`TxnId`].
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
    pub fn new(time: NetworkTime) -> Self {
        let mut rng = rand::thread_rng();
        let nonce = loop {
            let nonce = rng.gen();
            if nonce > 0 && nonce < (u16::MAX - 1) {
                break nonce;
            }
        };

        Self {
            timestamp: time.as_nanos(),
            nonce,
        }
    }

    /// Return the last valid TxnId before this one (e.g. to construct a range).
    pub fn prev(&self) -> Self {
        Self {
            timestamp: self.timestamp,
            nonce: self.nonce - 1,
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

impl<D: Digest> Hash<D> for TxnId {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self)
    }
}

impl<'a, D: Digest> Hash<D> for &'a TxnId {
    fn hash(self) -> Output<D> {
        let mut bytes = [0u8; 10];
        bytes[..8].copy_from_slice(&self.timestamp.to_be_bytes());
        bytes[8..].copy_from_slice(&self.nonce.to_be_bytes());
        D::digest(&bytes)
    }
}

impl fmt::Display for TxnId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}-{}", self.timestamp, self.nonce)
    }
}
