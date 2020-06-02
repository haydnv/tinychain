use structopt::StructOpt;

#[allow(dead_code)]
mod auth;

mod error;
mod gateway;
mod host;
mod http;
mod i18n;
mod internal;
mod state;
mod transaction;
mod value;

const DEFAULT_LOCALE: &str = "en_US";
const VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = host::HostConfig::from_args();

    println!("Tinychain version {}", VERSION);
    println!("Data directory: {}", &config.data_dir.to_str().unwrap());
    println!("Working directory: {}", &config.workspace.to_str().unwrap());
    println!();

    let host = host::Host::new(config).await?;
    host.http_listen().await?;

    Ok(())
}
