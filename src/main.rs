use std::net::IpAddr;
use std::path::PathBuf;
use std::sync::Arc;

use structopt::StructOpt;

#[allow(dead_code)]
mod auth;

mod cluster;
mod error;
mod gateway;
mod http;
mod internal;
mod kernel;
mod state;
mod transaction;
mod value;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const RESERVED: [&str; 1] = ["/sbin"];

fn block_size(flag: &str) -> value::TCResult<usize> {
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

    #[structopt(long = "block_size", default_value = "1K", parse(try_from_str = block_size))]
    pub block_size: usize,

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

    let workspace = internal::Dir::new_tmp(config.workspace.clone());
    let hosted = configure(config, workspace.clone()).await?;
    let _gateway = gateway::Gateway::new(hosted, workspace);

    Ok(())
}

async fn configure(
    config: Config,
    workspace: Arc<internal::Dir>,
) -> value::TCResult<gateway::Hosted> {
    let data_dir = internal::Dir::new(config.data_dir);

    let txn = gateway::Gateway::new(gateway::Hosted::new(), workspace)
        .transaction()
        .await?;
    let txn_id = txn.id();
    let mut hosted = gateway::Hosted::new();
    for path in config.hosted {
        for reserved in RESERVED.iter() {
            if path.to_string().starts_with(reserved) {
                return Err(error::bad_request("Attempt to host a reserved path", path));
            }

            if hosted.get(&path).is_some() {
                return Err(error::bad_request(
                    "Cannot host a subdirectory of a hosted directory",
                    path,
                ));
            }

            let hosted_cluster = if let Some(_dir) = data_dir.get_dir(txn_id, &path).await? {
                //                use internal::file::File;
                //                cluster::Cluster::from_dir(txn_id, dir).await
                panic!("NOT IMPLEMENTED")
            } else {
                cluster::Cluster::new(txn_id, data_dir.create_dir(&txn_id, path.clone()).await?)
            };

            hosted.push(path.clone(), hosted_cluster);
        }
    }

    Ok(hosted)
}
