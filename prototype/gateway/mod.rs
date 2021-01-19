use std::sync::Arc;

use async_trait::async_trait;

#[allow(clippy::module_inception)]
mod gateway;
mod hosted;
mod http;
mod time;

pub type Gateway = gateway::Gateway;
pub type Hosted = hosted::Hosted;
pub type HttpServer = http::Server;
pub type NetworkTime = time::NetworkTime;

#[async_trait]
pub trait Server {
    type Error;

    async fn listen(self: Arc<Self>, gateway: Arc<Gateway>) -> Result<(), Self::Error>;
}
