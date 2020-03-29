use std::path::PathBuf;

use structopt::StructOpt;

mod drive;
mod http;

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

    let _workspace = drive::Drive::new(config.workspace);
    http::listen(config.http_port).await?;
    Ok(())
}
