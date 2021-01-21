use std::net::IpAddr;

use async_trait::async_trait;
use futures::join;

pub mod http;
pub mod state;
pub mod txn;

const DEFAULT_ADDR: &'static str = "127.0.0.1";
const DEFAULT_PORT: u16 = 8702;

#[async_trait]
pub trait Server {
    type Error;

    async fn listen(self) -> Result<(), Self::Error>;
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let address: IpAddr = DEFAULT_ADDR.parse().unwrap();
    let server = http::HTTPServer::new((address, DEFAULT_PORT).into());

    if let (Err(cause),) = join!(server.listen()) {
        log::error!("Server error: {}", cause);
    }

    Ok(())
}
