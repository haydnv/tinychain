use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;

use structopt::StructOpt;

use error::TCError;

use generic::PathSegment;
use tinychain::gateway::Gateway;
use tinychain::*;
use tokio::time::Duration;
use transact::TxnId;

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

fn duration(flag: &str) -> error::TCResult<Duration> {
    u64::from_str(flag)
        .map(Duration::from_secs)
        .map_err(|_| error::TCError::bad_request("Invalid duration", flag))
}

#[derive(Clone, StructOpt)]
struct Config {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[structopt(long = "log_level", default_value = "warn")]
    pub log_level: String,

    #[structopt(long = "workspace", default_value = "/tmp/tc/tmp")]
    pub workspace: PathBuf,

    #[structopt(long = "cache_size", default_value = "10M", parse(try_from_str = data_size))]
    pub cache_size: usize,

    #[structopt(long = "config", default_value = "config")]
    pub config: PathBuf,

    #[structopt(long = "data_dir")]
    pub data_dir: Option<PathBuf>,

    #[structopt(long = "cluster")]
    pub clusters: Vec<generic::TCPathBuf>,

    #[structopt(long = "request_ttl", default_value = "30", parse(try_from_str = duration))]
    pub request_ttl: Duration,

    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Config::from_args();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(config.log_level))
        .init();

    let cache = fs::Cache::new(config.cache_size);

    let workspace = fs::load(cache.clone(), config.workspace).await?;
    let txn_server = tinychain::txn::TxnServer::new(workspace).await;

    let kernel = tinychain::Kernel::new(vec![]);
    let gateway = tinychain::gateway::Gateway::new(
        kernel,
        txn_server.clone(),
        config.address,
        config.http_port,
        config.request_ttl,
    );

    let token = gateway.issue_token();
    let mut clusters = Vec::with_capacity(config.clusters.len());
    if !config.clusters.is_empty() {
        let data_dir = config
            .data_dir
            .ok_or_else(|| TCError::internal("missing required option: --data_dir"))?;

        let data_dir = fs::load(cache.clone(), data_dir).await?;

        for path in config.clusters {
            let txn_id = TxnId::new(Gateway::time());
            let request = txn::Request::new(token.clone(), txn_id);
            let txn = gateway.new_txn(request).await?;
            let config = get_config(&config.config, &path).await?;
            let cluster = cluster::Cluster::load(data_dir.clone(), txn, path, config).await?;

            clusters.push(cluster);
        }
    }

    let kernel = tinychain::Kernel::new(vec![]);
    let gateway = tinychain::gateway::Gateway::new(
        kernel,
        txn_server,
        config.address,
        config.http_port,
        config.request_ttl,
    );

    if let Err(cause) = gateway.listen().await {
        log::error!("Server error: {}", cause);
    }

    Ok(())
}

async fn get_config(config_dir: &PathBuf, path: &[PathSegment]) -> error::TCResult<Vec<u8>> {
    let mut fs_path = config_dir.clone();
    for id in path {
        fs_path.push(id.to_string());
    }

    match tokio::fs::read(&fs_path).await {
        Ok(config) => Ok(config),
        Err(cause) => Err(error::TCError::internal(format!(
            "could not read config file at {:?}: {}",
            fs_path, cause
        ))),
    }
}
