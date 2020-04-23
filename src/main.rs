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

    println!("Tinychain version {}", VERSION);
    println!("Working directory: {}", &config.workspace.to_str().unwrap());
    println!();

    let data_dir = internal::FsDir::new(value::Link::to("/")?, config.data_dir);
    let workspace = internal::FsDir::new_tmp(value::Link::to("/")?, config.workspace);

    let hosted = config
        .host
        .iter()
        .map(|d| value::Link::to(d))
        .collect::<value::TCResult<Vec<value::Link>>>()?;

    let host = host::Host::new(data_dir, workspace)?;

    for path in hosted {
        host.claim(path).await?;
    }

    http::listen(host, config.http_port).await?;
    Ok(())
}
