use std::net::SocketAddr;

pub mod block;
pub mod kernel;
pub mod scalar;
pub mod state;
pub mod transact;

#[cfg(feature = "http")]
pub mod http;

pub use auth;
pub use error;
pub use generic;
pub use kernel::*;
pub use value;

#[async_trait::async_trait]
pub trait Server {
    type Error: std::error::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error>;
}
