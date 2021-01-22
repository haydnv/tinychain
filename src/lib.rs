pub mod block;

#[cfg(feature = "http")]
pub mod http;

pub mod state;
pub mod txn;

#[async_trait::async_trait]
pub trait Server {
    type Error: std::error::Error;

    async fn listen(self) -> Result<(), Self::Error>;
}
