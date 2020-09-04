use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

use arrayfire as af;
use structopt::StructOpt;

mod auth;
mod block;
mod class;
mod collection;
mod error;
mod gateway;
mod kernel;
mod lock;
mod transaction;
mod value;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn block_size(flag: &str) -> class::TCResult<usize> {
    let msg = "Unable to parse value of block_size";
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

    #[structopt(long = "host")]
    pub hosted: Vec<value::link::TCPath>,

    #[structopt(long = "peer")]
    pub peers: Vec<value::link::LinkHost>,
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
    let data_dir = block::dir::Dir::create(fs_cache_persistent, false);
    let fs_cache_temporary = block::hostfs::mount(config.workspace);
    let workspace = block::dir::Dir::create(fs_cache_temporary, true);

    use transaction::Transact;
    data_dir.commit(&txn_id).await;
    workspace.commit(&txn_id).await;

    let hosted = configure(config.hosted, data_dir.clone(), workspace.clone()).await?;
    let gateway = gateway::Gateway::new(config.peers, hosted, workspace.clone());
    Arc::new(gateway)
        .http_listen(config.address, config.http_port)
        .await
        .map_err(|e| e.into())
}

async fn configure(
    clusters: Vec<value::link::TCPath>,
    _data_dir: Arc<block::dir::Dir>,
    _workspace: Arc<block::dir::Dir>,
) -> class::TCResult<gateway::Hosted> {
    let hosted = gateway::Hosted::new();
    if clusters.is_empty() {
        return Ok(hosted);
    }

    unimplemented!(); // TODO: load hosted cluster data from disk
}
