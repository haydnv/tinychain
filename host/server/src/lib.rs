//! State replication management

use std::time::Duration;

mod builder;
mod cluster;
mod gateway;
mod kernel;
mod server;
mod txn;

pub use builder::{Aes256Key, ServerBuilder};

pub const DEFAULT_TTL: Duration = Duration::from_secs(3);
pub const SERVICE_TYPE: &'static str = "_tinychain._tcp.local.";
