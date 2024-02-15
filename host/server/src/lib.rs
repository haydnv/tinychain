//! State replication management

use std::time::Duration;

use async_trait::async_trait;

use tc_value::{Link, ToUrl, Value};

use tc_error::TCResult;
use tcgeneric::Map;

pub use builder::ServerBuilder;
pub use server::Server;
pub use txn::Txn;

mod builder;
mod claim;
mod cluster;
mod kernel;
mod server;
mod txn;

pub mod aes256 {
    pub use aes_gcm_siv::aead::OsRng;
    pub use aes_gcm_siv::{Aes256GcmSiv, KeyInit};

    pub type Key = aes_gcm_siv::Key<Aes256GcmSiv>;
}

pub const DEFAULT_TTL: Duration = Duration::from_secs(3);
pub const SERVICE_TYPE: &'static str = "_tinychain._tcp.local.";

type Actor = rjwt::Actor<Value>;
type SignedToken = rjwt::SignedToken<Link, Value, claim::Claim>;

#[async_trait]
pub trait RPCClient<State> {
    async fn get(&self, link: ToUrl<'_>, key: Value) -> TCResult<State>;

    async fn put(&self, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()>;

    async fn post(&self, link: ToUrl<'_>, params: Map<State>) -> TCResult<State>;

    async fn delete(&self, link: ToUrl<'_>, key: Value) -> TCResult<State>;
}
