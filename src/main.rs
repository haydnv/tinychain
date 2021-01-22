use std::net::IpAddr;

use futures::future::{try_join_all, Future};
use structopt::StructOpt;

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

    #[allow(unused_mut)]
    let mut servers = Vec::<Box<dyn Future<Output = Result<(), Box<dyn std::error::Error>>> + Unpin>>::with_capacity(1);

    #[cfg(feature = "http")]
    {
        use futures::TryFutureExt;
        use tinychain::{http, Server};

        let http_addr = (config.address, config.http_port).into();
        let server = http::HTTPServer::new(http_addr);
        servers.push(Box::new(server.listen().map_err(|e| {
            let e: Box<dyn std::error::Error> = Box::new(e);
            e
        })));
    }

    if let Err(cause) = try_join_all(servers).await {
        log::error!("Server error: {}", cause);
    }

    Ok(())
}
