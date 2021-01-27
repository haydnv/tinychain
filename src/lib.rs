use std::net::SocketAddr;

mod route;

pub mod block;
pub mod kernel;
pub mod scalar;
pub mod state;

#[cfg(feature = "http")]
pub mod http;

pub use auth;
pub use error;
pub use generic;
pub use kernel::*;
pub use transact::{lock as txn_lock, TxnId};
pub use value;

pub type Txn = transact::Txn<state::State>;

#[async_trait::async_trait]
pub trait Server {
    type Error: std::error::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error>;
}
