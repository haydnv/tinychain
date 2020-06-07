use std::sync::Arc;

use async_trait::async_trait;

#[allow(clippy::module_inception)]
mod gateway;
mod hosted;
mod time;

pub type Gateway = gateway::Gateway;
pub type Hosted = hosted::Hosted;
pub type NetworkTime = time::NetworkTime;

#[async_trait]
pub trait Protocol {
    type Error;

    async fn listen(self: Arc<Self>) -> Result<(), Self::Error>;
}
