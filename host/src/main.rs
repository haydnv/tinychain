use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
use destream::de::Error;
use tokio::time::Duration;

use tc_error::*;
use tc_value::LinkHost;

use tinychain::*;

fn data_size(flag: &str) -> TCResult<usize> {
    if flag.is_empty() || flag == "0" {
        return Ok(0);
    }

    let size = usize::from_str_radix(&flag[0..flag.len() - 1], 10)
        .map_err(|_| TCError::invalid_value(flag, "a data size"))?;

    if flag.ends_with('K') {
        Ok(size * 1000)
    } else if flag.ends_with('M') {
        Ok(size * 1_000_000)
    } else if flag.ends_with('G') {
        Ok(size * 1_000_000_000)
    } else {
        Err(TCError::invalid_value(
            &flag[flag.len() - 1..],
            "a data size suffix",
        ))
    }
}

fn duration(flag: &str) -> TCResult<Duration> {
    u64::from_str(flag)
        .map(Duration::from_secs)
        .map_err(|_| TCError::invalid_value(flag, "a duration"))
}

#[derive(Clone, Parser)]
struct Config {
    #[arg(
        long,
        default_value = "127.0.0.1",
        help = "the IP address of this host"
    )]
    pub address: IpAddr,

    #[arg(
        long = "cache_size",
        value_parser = data_size,
        default_value = "1G",
        help = "the maximum size of the in-memory transactional filesystem cache (in bytes)",
    )]
    pub cache_size: usize,

    #[arg(
        long = "data_dir",
        help = "the directory to use for persistent data storage"
    )]
    pub data_dir: PathBuf,

    #[arg(
        long = "http_port",
        default_value = "8702",
        help = "the port for the HTTP server to bind"
    )]
    pub http_port: u16,

    #[arg(
        long = "log_level",
        default_value = "info",
        value_parser = ["trace", "debug", "info", "warn", "error"],
        help = "the log message level to write",
    )]
    pub log_level: String,

    #[arg(
        long = "public_key",
        help = "a hexadecimal string representation of this host's clusters' public key"
    )]
    pub public_key: Option<String>,

    #[arg(long, help = "a link to the cluster to replicate from on startup")]
    pub replicate: Option<LinkHost>,

    #[arg(
        long = "request_ttl",
        value_parser = duration,
        default_value = "30",
        help = "maximum allowed request duration",
    )]
    pub request_ttl: Duration,

    #[arg(
        long = "stack_size",
        value_parser = data_size,
        default_value = "4M",
        help = "the size of the stack of each worker thread (in bytes)",
    )]
    pub stack_size: usize,

    #[arg(
        long,
        default_value = "/tmp/tc/tmp",
        help = "the directory to use as a temporary workspace in case of a cache overflow"
    )]
    pub workspace: PathBuf,
}

impl Config {
    fn gateway(&self) -> gateway::Config {
        gateway::Config {
            addr: self.address,
            http_port: self.http_port,
            request_ttl: self.request_ttl,
        }
    }
}

fn main() {
    let config = Config::parse();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(&config.log_level))
        .init();

    let gateway_config = config.gateway();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .thread_stack_size(config.stack_size)
        .build()
        .expect("tokio runtime");

    let builder = runtime
        .block_on(Builder::load(
            config.cache_size,
            config.data_dir,
            config.workspace,
        ))
        .with_public_key(config.public_key)
        .with_gateway(gateway_config)
        .with_lead(config.replicate);

    match runtime.block_on(builder.replicate_and_serve()) {
        Ok(_) => {}
        Err(cause) => panic!("HTTP server failed: {}", cause),
    }
}
