//! State replication management

use std::time::Duration;

pub use builder::{Aes256Key, ServerBuilder};

mod builder;
mod cluster;
mod kernel;
mod server;
mod txn;

pub const DEFAULT_TTL: Duration = Duration::from_secs(3);
pub const SERVICE_TYPE: &'static str = "_tinychain._tcp.local.";
