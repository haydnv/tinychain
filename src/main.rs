use std::collections::HashMap;
use std::net::IpAddr;

use structopt::StructOpt;

use tinychain::gateway::Gateway;

#[derive(Clone, StructOpt)]
struct Config {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[cfg(feature = "http")]
    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "log_level", default_value = "warn")]
    pub log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Config::from_args();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(config.log_level))
        .init();

    #[allow(unused_mut)]
    let mut gateway_config = HashMap::new();

    #[cfg(feature = "http")]
    gateway_config.insert(value::LinkProtocol::HTTP, config.http_port);

    let gateway = Gateway::new(tinychain::Kernel, config.address, gateway_config);

    if let Err(cause) = gateway.listen().await {
        log::error!("Server error: {}", cause);
    }

    Ok(())
}
