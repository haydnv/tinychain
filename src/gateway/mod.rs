use std::sync::Arc;

use async_trait::async_trait;

#[allow(clippy::module_inception)]
mod gateway;
mod hosted;
pub mod op;

pub type Gateway = gateway::Gateway;
pub type Hosted = hosted::Hosted;

#[async_trait]
pub trait Protocol {
    type Config;
    type Error;

    fn new(gateway: Arc<Gateway>, config: Self::Config) -> Self;

    async fn listen(&self) -> Result<(), Self::Error>;
}
