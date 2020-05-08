use std::path::PathBuf;

use structopt::StructOpt;

mod error;
mod host;
mod http;
mod internal;
mod object;
mod state;
mod transaction;
mod value;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn block_size(flag: &str) -> value::TCResult<usize> {
    let msg = "Unable to parse value of block_size";

    let size = usize::from_str_radix(&flag[0..flag.len() - 1], 10)
        .map_err(|_| error::bad_request(msg, flag))?;

    if flag.ends_with('K') {
        Ok(size * 1000)
    } else if flag.ends_with('M') {
        Ok(size * 1_000_000)
    } else if flag.ends_with('G') {
        Ok(size * 1_000_000_000)
    } else {
        Err(error::bad_request(msg, flag))
    }
}

#[derive(Clone, StructOpt)]
pub struct HostConfig {
    #[structopt(long = "block_size", default_value = "1K", parse(try_from_str = block_size))]
    pub block_size: usize,

    #[structopt(long = "data_dir", default_value = "/tmp/tc/data")]
    pub data_dir: PathBuf,

    #[structopt(long = "workspace", default_value = "/tmp/tc/tmp")]
    pub workspace: PathBuf,

    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "host")]
    pub hosted: Vec<value::TCPath>,

    #[structopt(long = "peer")]
    pub peers: Vec<value::Link>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = HostConfig::from_args();

    println!("Tinychain version {}", VERSION);
    println!("Data directory: {}", &config.data_dir.to_str().unwrap());
    println!("Working directory: {}", &config.workspace.to_str().unwrap());
    println!();

    let data_dir = internal::block::Store::new(config.data_dir, config.block_size, None);
    let workspace = internal::block::Store::new_tmp(config.workspace, config.block_size, None);

    let host = host::Host::new(data_dir, workspace, config.hosted).await?;

    http::listen(host, config.http_port).await?;
    Ok(())
}
