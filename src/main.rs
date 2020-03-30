use std::path::PathBuf;
use std::sync::Arc;

use structopt::StructOpt;

mod cache;
mod context;
mod drive;
mod error;
mod host;
mod http;
mod table;
mod transaction;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, StructOpt)]
pub struct HostConfig {
    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "workspace", default_value = "/tmp/tc")]
    pub workspace: PathBuf,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = HostConfig::from_args();

    println!("Tinychain version {}", VERSION);
    println!("Working directory: {}", &config.workspace.to_str().unwrap());

    let workspace = drive::Drive::new(config.workspace);
    let host = Arc::new(host::HostContext::new(workspace));
    http::listen(host, config.http_port).await?;
    Ok(())
}
