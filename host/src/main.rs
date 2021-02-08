use std::collections::HashMap;
use std::net::IpAddr;
use std::path::PathBuf;

use bytes::Bytes;
use futures::TryFutureExt;
use structopt::StructOpt;

use error::TCError;

use generic::PathSegment;
use tinychain::*;

fn data_size(flag: &str) -> error::TCResult<usize> {
    if flag.is_empty() {
        return Err(error::TCError::bad_request("Invalid size specified", flag));
    }

    let msg = "Unable to parse value";
    let size = usize::from_str_radix(&flag[0..flag.len() - 1], 10)
        .map_err(|_| error::TCError::bad_request(msg, flag))?;

    if flag.ends_with('K') {
        Ok(size * 1000)
    } else if flag.ends_with('M') {
        Ok(size * 1_000_000)
    } else {
        Err(error::TCError::bad_request(
            "Unable to parse request_limit",
            flag,
        ))
    }
}

#[derive(Clone, StructOpt)]
struct Config {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[structopt(long = "log_level", default_value = "warn")]
    pub log_level: String,

    #[structopt(long = "cache_size", default_value = "10M", parse(try_from_str = data_size))]
    pub cache_size: usize,

    #[structopt(long = "config", default_value = "config")]
    pub config: PathBuf,

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

    let cache = fs::Cache::new(config.cache_size);

    let mut clusters = Vec::with_capacity(config.clusters.len());
    if !config.clusters.is_empty() {
        let data_dir = config
            .data_dir
            .ok_or_else(|| TCError::internal("missing required option: --data_dir"))?;

        let data_dir = fs::load(cache.clone(), data_dir)
            .map_ok(fs::Dir::new)
            .await?;

        for path in config.clusters {
            let config = get_config(&config.config, &path).await?;
            let cluster = cluster::Cluster::load(data_dir.clone(), path, config).await?;

            clusters.push(cluster);
        }
    }

    #[allow(unused_mut)]
    let mut gateway_config = HashMap::new();

    #[cfg(feature = "http")]
    gateway_config.insert(value::LinkProtocol::HTTP, config.http_port);

    let workspace = fs::load(cache, config.workspace).await?;
    let txn_server = tinychain::txn::TxnServer::new(workspace).await;
    let kernel = tinychain::Kernel::new(clusters);
    let gateway =
        tinychain::gateway::Gateway::new(kernel, txn_server, config.address, gateway_config);

    if let Err(cause) = gateway.listen().await {
        log::error!("Server error: {}", cause);
    }

    Ok(())
}

async fn get_config<'a>(config_dir: &PathBuf, path: &'a [PathSegment]) -> error::TCResult<Bytes> {
    let mut fs_path = config_dir.clone();
    for id in path {
        fs_path.push(id.to_string());
    }

    match tokio::fs::read(&fs_path).await {
        Ok(config) => Ok(Bytes::from(config)),
        Err(cause) => Err(error::TCError::internal(format!(
            "could not read config file at {:?}: {}",
            fs_path, cause
        ))),
    }
}
