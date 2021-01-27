use std::net::IpAddr;

use futures::future::{try_join_all, Future};
use structopt::StructOpt;

use tinychain::gateway::Gateway;

#[derive(Clone, StructOpt)]
struct Config {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "log_level", default_value = "warn")]
    pub log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Config::from_args();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(config.log_level))
        .init();

    #[allow(unused_variables)]
    let gateway = Gateway::new();

    #[allow(unused_mut)]
    let mut servers = Vec::<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error>>> + Unpin>>::with_capacity(1);

    #[cfg(feature = "http")]
    {
        servers.push(Box::new(
            gateway.http_listen(config.address, config.http_port),
        ));
    }

    if let Err(cause) = try_join_all(servers).await {
        log::error!("Server error: {}", cause);
    }

    Ok(())
}
