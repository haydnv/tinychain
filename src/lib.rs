use std::net::SocketAddr;

pub mod block;
pub mod kernel;
pub mod state;
pub mod txn;

#[cfg(feature = "http")]
pub mod http;

pub use kernel::*;

#[async_trait::async_trait]
pub trait Server {
    type Error: std::error::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error>;
}
