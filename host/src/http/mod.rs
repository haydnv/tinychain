//! The HTTP interface for `Gateway`.

use std::fmt;
use std::str::FromStr;

use tc_error::TCError;

mod client;
mod server;

pub use client::*;
pub use server::*;

enum Encoding {
    Json,
}

impl Default for Encoding {
    fn default() -> Self {
        Self::Json
    }
}

impl FromStr for Encoding {
    type Err = TCError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "application/json" => Ok(Self::Json),
            _ => Err(TCError::bad_request("invalid encoding specifier", s)),
        }
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Json => f.write_str("application/json"),
        }
    }
}
