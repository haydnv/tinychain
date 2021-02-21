//! Time utilities for Tinychain. UNSTABLE.

use std::convert::{TryFrom, TryInto};
use std::ops;
use std::time;
use std::time::Duration;

use tc_error::*;

#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct NetworkTime {
    nanos: u64,
}

/// A Tinychain timestamp, used for absolute ordering of transactions.
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

impl TryFrom<time::SystemTime> for NetworkTime {
    type Error = TCError;

    fn try_from(st: time::SystemTime) -> TCResult<NetworkTime> {
        let st = st
            .duration_since(time::UNIX_EPOCH)
            .map_err(|e| TCError::bad_request("invalid timestamp", e))?;
        let nanos: u64 = st
            .as_nanos()
            .try_into()
            .map_err(|e| TCError::bad_request("invalid timestamp", e))?;
        Ok(NetworkTime::from_nanos(nanos))
    }
}

impl From<NetworkTime> for time::SystemTime {
    fn from(nt: NetworkTime) -> Self {
        time::UNIX_EPOCH + Duration::from_nanos(nt.nanos as u64)
    }
}
