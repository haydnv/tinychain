use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

use arrayfire as af;
use structopt::StructOpt;

mod auth;
mod block;
mod chain;
mod class;
mod cluster;
mod collection;
mod error;
mod gateway;
mod kernel;
mod lock;
mod transaction;
mod value;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn data_size(flag: &str) -> error::TCResult<usize> {
    if flag.is_empty() {
        return Err(error::bad_request("Invalid size specified", flag));
    }

    let msg = "Unable to parse value";
    let size = usize::from_str_radix(&flag[0..flag.len() - 1], 10)
        .map_err(|_| error::bad_request(msg, flag))?;
    if flag.ends_with('K') {
        Ok(size * 1000)
    } else if flag.ends_with('M') {
        Ok(size * 1_000_000)
    } else {
        Err(error::bad_request(msg, flag))
    }
}

#[derive(Clone, StructOpt)]
struct Config {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[structopt(long = "data_dir", default_value = "/tmp/tc/data")]
    pub data_dir: PathBuf,

    #[structopt(long = "workspace", default_value = "/tmp/tc/tmp")]
    pub workspace: PathBuf,

    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "ext")]
    pub adapters: Vec<value::link::Link>,

    #[structopt(long = "host")]
    pub hosted: Vec<value::link::TCPath>,

    #[structopt(long = "peer")]
    pub peers: Vec<value::link::LinkHost>,

    #[structopt(long = "request_limit", default_value = "10M", parse(try_from_str = data_size))]
    pub request_limit: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Config::from_args();

    println!("Tinychain version {}", VERSION);
    println!("Data directory: {}", &config.data_dir.to_str().unwrap());
    println!("Working directory: {}", &config.workspace.to_str().unwrap());
    println!();

    af::info();
    println!();

    let txn_id = transaction::TxnId::new(gateway::Gateway::time());
    let fs_cache_persistent = block::hostfs::mount(config.data_dir);
    let data_dir = block::Dir::create(fs_cache_persistent, false);
    let fs_cache_temporary = block::hostfs::mount(config.workspace);
    let workspace = block::Dir::create(fs_cache_temporary, true);

    use transaction::Transact;
    data_dir.commit(&txn_id).await;
    workspace.commit(&txn_id).await;

    let hosted = configure(config.hosted, data_dir.clone(), workspace.clone()).await?;
    let gateway = gateway::Gateway::new(
        config.peers,
        config.adapters,
        hosted,
        workspace.clone(),
        config.request_limit,
    )
    .map_err(Box::new)?;

    Arc::new(gateway)
        .http_listen(config.address, config.http_port)
        .await
        .map_err(|e| e.into())
}

async fn configure(
    clusters: Vec<value::link::TCPath>,
    data_dir: Arc<block::Dir>,
    workspace: Arc<block::Dir>,
) -> error::TCResult<gateway::Hosted> {
    let mut hosted = gateway::Hosted::new();
    for path in clusters {
        let cluster = cluster::Cluster::create(path.clone(), data_dir.clone(), workspace.clone())?;
        hosted.push(path, cluster);
    }

    Ok(hosted)
}
