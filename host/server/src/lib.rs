//! State replication management

use std::time::Duration;

use tc_value::{Link, Value};

pub use builder::{Aes256Key, ServerBuilder};

mod builder;
mod claim;
mod cluster;
mod gateway;
mod kernel;
mod server;
mod txn;

pub const DEFAULT_TTL: Duration = Duration::from_secs(3);
pub const SERVICE_TYPE: &'static str = "_tinychain._tcp.local.";

type Actor = rjwt::Actor<Value>;
type SignedToken = rjwt::SignedToken<Link, Value, claim::Claim>;
