use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;

use structopt::StructOpt;

use error::TCError;
use transact::fs;

use tinychain::*;

#[derive(Clone, StructOpt)]
struct Config {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[structopt(long = "log_level", default_value = "warn")]
    pub log_level: String,

    #[structopt(long = "data_dir")]
    pub data_dir: Option<PathBuf>,

    #[structopt(long = "cluster")]
    pub clusters: Vec<generic::TCPathBuf>,

    #[structopt(long = "workspace", default_value = "/tmp/tc/tmp")]
    pub workspace: PathBuf,

    #[cfg(feature = "http")]
    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Config::from_args();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(config.log_level))
        .init();

    let mut clusters = Vec::with_capacity(config.clusters.len());
    if !config.clusters.is_empty() {
        let data_dir = config
            .data_dir
            .ok_or_else(|| TCError::internal("missing required option: --data_dir"))?;

        let data_dir = fs::mount(data_dir).await;

        for path in config.clusters {
            let dir_lock = data_dir.read().await;
            let cluster = cluster::Cluster::load(dir_lock, path.into()).await?;
            clusters.push(cluster);
        }
    }

    #[allow(unused_mut)]
    let mut gateway_config = HashMap::new();

    #[cfg(feature = "http")]
    gateway_config.insert(value::LinkProtocol::HTTP, config.http_port);

    let txn_server = tinychain::txn::TxnServer::new(config.workspace).await;
    let kernel = tinychain::Kernel::new(clusters);
    let gateway =
        tinychain::gateway::Gateway::new(kernel, txn_server, config.address, gateway_config);

    if let Err(cause) = gateway.listen().await {
        log::error!("Server error: {}", cause);
    }

    Ok(())
}
