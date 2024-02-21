//! State replication management

use std::time::Duration;

use async_trait::async_trait;
use umask::Mode;

use tc_value::{Link, ToUrl, Value};

use tc_error::*;
use tcgeneric::Map;

pub use builder::Builder;
pub use kernel::Endpoint;
pub use server::Server;
pub use tc_state::CacheBlock;
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

pub const DEFAULT_MAX_RETRIES: u8 = 3;
pub const DEFAULT_PORT: u16 = 8702;
pub const DEFAULT_TTL: Duration = Duration::from_secs(3);
pub const SERVICE_TYPE: &'static str = "_tinychain._tcp.local.";

pub type Actor = rjwt::Actor<Value>;
pub type SignedToken = rjwt::SignedToken<Link, Value, claim::Claim>;
pub type Token = rjwt::Token<Link, Value, claim::Claim>;
pub type State = tc_state::State<Txn>;

pub trait Authorize {
    fn has_any<const N: usize>(&self, modes: [Mode; N]) -> bool;

    fn is_none(self) -> bool;

    fn may_read(self) -> bool;

    fn may_write(self) -> bool;

    fn may_execute(self) -> bool;
}

impl Authorize for Mode {
    #[inline]
    fn has_any<const N: usize>(&self, modes: [Mode; N]) -> bool {
        for mode in modes {
            if self.has(mode) {
                return true;
            }
        }

        false
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

#[async_trait]
pub trait RPCClient: Send + Sync {
    async fn get(&self, link: ToUrl<'_>, key: Value, token: Option<SignedToken>)
        -> TCResult<State>;

    async fn put(
        &self,
        link: ToUrl<'_>,
        key: Value,
        value: State,
        token: Option<SignedToken>,
    ) -> TCResult<()>;

    async fn post(
        &self,
        link: ToUrl<'_>,
        params: Map<State>,
        token: Option<SignedToken>,
    ) -> TCResult<State>;

    async fn delete(&self, link: ToUrl<'_>, key: Value, token: Option<SignedToken>)
        -> TCResult<()>;

    async fn verify(&self, token: String) -> TCResult<SignedToken>;
}
