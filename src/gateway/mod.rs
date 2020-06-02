#[allow(clippy::module_inception)]
mod gateway;
mod hosted;
pub mod op;

pub type Gateway = gateway::Gateway;
pub type Hosted = hosted::Hosted;
