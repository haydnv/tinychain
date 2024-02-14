//! Time utilities for TinyChain

use std::convert::{TryFrom, TryInto};
use std::ops;
use std::time;

use tc_error::*;

/// The current time of the TinyChain network, used to generate transaction IDs.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct NetworkTime {
    nanos: u64,
}

/// A TinyChain timestamp, used for absolute ordering of transactions.
impl NetworkTime {
    // TODO: replace system time with an explicit network time synchronization system.
    /// The current time.
    pub fn now() -> NetworkTime {
        NetworkTime::from_nanos(
            time::SystemTime::now()
                .duration_since(time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        )
    }

    /// This timestamp in nanoseconds since the Unix epoch.
    pub fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Constructs a new timestamp from a duration in nanoseconds since the Unix epoch.
    pub fn from_nanos(nanos: u64) -> NetworkTime {
        NetworkTime { nanos }
    }
}

impl ops::Add<time::Duration> for NetworkTime {
    type Output = Self;

    fn add(self, other: time::Duration) -> Self {
        NetworkTime {
            nanos: self.nanos + other.as_nanos() as u64,
        }
    }
}

impl ops::Add<NetworkTime> for time::Duration {
    type Output = NetworkTime;

    fn add(self, other: NetworkTime) -> NetworkTime {
        NetworkTime {
            nanos: self.as_nanos() as u64 + other.nanos,
        }
    }
}

impl ops::Sub<NetworkTime> for NetworkTime {
    type Output = time::Duration;

    fn sub(self, other: NetworkTime) -> time::Duration {
        time::Duration::from_nanos(self.nanos - other.nanos)
    }
}

impl ops::Sub<time::Duration> for NetworkTime {
    type Output = Self;

    fn sub(self, other: time::Duration) -> Self {
        NetworkTime {
            nanos: self.nanos - other.as_nanos() as u64,
        }
    }
}

impl ops::Sub<NetworkTime> for time::Duration {
    type Output = NetworkTime;

    fn sub(self, other: NetworkTime) -> NetworkTime {
        NetworkTime {
            nanos: self.as_nanos() as u64 - other.nanos,
        }
    }
}

impl TryFrom<time::SystemTime> for NetworkTime {
    type Error = TCError;

    fn try_from(st: time::SystemTime) -> TCResult<NetworkTime> {
        let st = st
            .duration_since(time::UNIX_EPOCH)
            .map_err(|cause| bad_request!("invalid timestamp: {:?}", st).consume(cause))?;

        let nanos: u64 = st.as_nanos().try_into().map_err(|cause| {
            bad_request!("expected a 64-bit timestamp but found {:?}", st).consume(cause)
        })?;

        Ok(NetworkTime::from_nanos(nanos))
    }
}

impl From<NetworkTime> for time::SystemTime {
    fn from(nt: NetworkTime) -> Self {
        time::UNIX_EPOCH + time::Duration::from_nanos(nt.nanos)
    }
}
