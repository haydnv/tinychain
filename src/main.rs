use std::convert::TryInto;
use std::path::PathBuf;

use structopt::StructOpt;

mod error;
mod host;
mod http;
mod internal;
mod state;
mod transaction;
mod value;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, StructOpt)]
pub struct HostConfig {
    #[structopt(long = "data_dir", default_value = "/tmp/tc/data")]
    pub data_dir: PathBuf,

    #[structopt(long = "workspace", default_value = "/tmp/tc/tmp")]
    pub workspace: PathBuf,

    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "host")]
    pub host: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = HostConfig::from_args();

    for path in &config.host {
        if !path.starts_with('/') {
            eprintln!("Hosted path must start with a '/': {}", path);
            panic!();
        }
    }

    println!("Tinychain version {}", VERSION);
    println!("Data directory: {}", &config.data_dir.to_str().unwrap());
    println!("Working directory: {}", &config.workspace.to_str().unwrap());
    println!();

    let data_dir = internal::block::Store::new(config.data_dir, None);
    let workspace = internal::block::Store::new_tmp(config.workspace, None);

    let hosted = config
        .host
        .iter()
        .map(|d| d.as_str().try_into())
        .collect::<value::TCResult<Vec<value::TCPath>>>()?;

    let host = host::Host::new(data_dir, workspace, hosted).await?;

    http::listen(host, config.http_port).await?;
    Ok(())
}
