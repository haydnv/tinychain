use std::sync::Arc;

use async_trait::async_trait;

use crate::internal::Dir;

#[allow(dead_code, clippy::module_inception)]
mod gateway;
mod hosted;

#[allow(dead_code)]
pub mod op;
mod time;

pub type Gateway = gateway::Gateway;
pub type Hosted = hosted::Hosted;
pub type NetworkTime = time::NetworkTime;

#[async_trait]
pub trait Protocol {
    type Config;
    type Error;

    fn new(config: Self::Config, gateway: Arc<Gateway>, workspace: Arc<Dir>) -> Arc<Self>;

    async fn listen(self: Arc<Self>) -> Result<(), Self::Error>;
}
