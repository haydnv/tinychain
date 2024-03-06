//! State replication management

use std::time::Duration;

use umask::Mode;

use tc_value::{Address, Host, Link, Protocol, Value};

pub use rjwt::VerifyingKey;

pub use tc_state::CacheBlock;

pub use builder::{Broadcast, Builder, Replicator};
pub use claim::Claim;
pub use client::RPCClient;
pub use kernel::Endpoint;
pub use server::Server;
pub use txn::Txn;

mod builder;
mod claim;
mod client;
mod cluster;
mod kernel;
mod server;
mod txn;

pub mod aes256 {
    pub use aes_gcm_siv::aead::OsRng;
    pub use aes_gcm_siv::{Aes256GcmSiv, KeyInit};

    pub type Key = aes_gcm_siv::Key<Aes256GcmSiv>;
}

pub const DEFAULT_MAX_RETRIES: u8 = 3;
pub const DEFAULT_PORT: u16 = 8702;
pub const DEFAULT_TTL: Duration = Duration::from_secs(3);
pub const SERVICE_TYPE: &'static str = "_tinychain._tcp.local.";

pub type Actor = rjwt::Actor<Value>;
pub type SignedToken = rjwt::SignedToken<Link, Value, Claim>;
pub type Token = rjwt::Token<Link, Value, Claim>;

#[cfg(feature = "service")]
pub type Chain<T> = tc_chain::Chain<State, Txn, CacheBlock, T>;
#[cfg(feature = "service")]
pub type Collection = tc_collection::CollectionBase<State, Txn>;
pub type State = tc_state::State<Txn>;

pub trait Authorize {
    fn has_any<const N: usize>(&self, modes: [Mode; N]) -> bool;

    fn is_none(self) -> bool;

    fn may_read(self) -> bool;

    fn may_write(self) -> bool;

    fn may_execute(self) -> bool;
}

impl Authorize for Mode {
    fn has_any<const N: usize>(&self, modes: [Mode; N]) -> bool {
        modes.into_iter().any(|mode| self.has(mode))
    }

    #[inline]
    fn is_none(self) -> bool {
        self == Mode::new()
    }

    #[inline]
    fn may_read(self) -> bool {
        const ALLOW: [Mode; 3] = [umask::USER_READ, umask::GROUP_READ, umask::OTHERS_READ];
        self.has_any(ALLOW)
    }

    #[inline]
    fn may_write(self) -> bool {
        const ALLOW: [Mode; 3] = [umask::USER_WRITE, umask::GROUP_WRITE, umask::OTHERS_WRITE];
        self.has_any(ALLOW)
    }

    #[inline]
    fn may_execute(self) -> bool {
        const ALLOW: [Mode; 3] = [umask::USER_EXEC, umask::GROUP_EXEC, umask::OTHERS_EXEC];
        self.has_any(ALLOW)
    }
}

pub fn default_host() -> Host {
    Host::from((Protocol::HTTP, Address::LOCALHOST, DEFAULT_PORT))
}
