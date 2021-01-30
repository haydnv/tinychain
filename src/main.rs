use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;

use structopt::StructOpt;

#[derive(Clone, StructOpt)]
struct Config {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[cfg(feature = "http")]
    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "log_level", default_value = "warn")]
    pub log_level: String,

    #[structopt(long = "workspace", default_value = "/tmp/tc/tmp")]
    pub workspace: PathBuf,
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

    let txn_server = tinychain::txn::TxnServer::new(config.workspace).await;
    let kernel = tinychain::Kernel::new(txn_server);
    let gateway = tinychain::gateway::Gateway::new(kernel, config.address, gateway_config);

    if let Err(cause) = gateway.listen().await {
        log::error!("Server error: {}", cause);
    }

    Ok(())
}
