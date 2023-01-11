use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
use tokio::time::Duration;

use tc_error::*;
use tc_value::LinkHost;

use tinychain::*;

fn data_size(flag: &str) -> TCResult<usize> {
    const ERR: &str = "unable to parse data size";

    if flag.is_empty() || flag == "0" {
        return Ok(0);
    }

    let size = usize::from_str_radix(&flag[0..flag.len() - 1], 10)
        .map_err(|_| TCError::bad_request(ERR, flag))?;

    if flag.ends_with('K') {
        Ok(size * 1000)
    } else if flag.ends_with('M') {
        Ok(size * 1_000_000)
    } else if flag.ends_with('G') {
        Ok(size * 1_000_000_000)
    } else {
        Err(TCError::bad_request(ERR, flag))
    }
}

fn duration(flag: &str) -> TCResult<Duration> {
    u64::from_str(flag)
        .map(Duration::from_secs)
        .map_err(|_| TCError::bad_request("invalid duration", flag))
}

#[derive(Clone, Parser)]
struct Config {
    #[arg(long, default_value = "127.0.0.1", help = "the IP address to bind")]
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

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .thread_stack_size(config.stack_size)
        .build()
        .expect("tokio runtime");

    let gateway_config = config.gateway();
    let mut builder = Builder::new(config.data_dir, config.workspace)
        .with_cache_size(config.cache_size)
        .with_gateway(gateway_config);

    if let Some(lead) = config.replicate {
        builder = builder.with_lead(lead);
    }

    match runtime.block_on(builder.replicate_and_serve()) {
        Ok(_) => {}
        Err(cause) => panic!("HTTP server failed: {}", cause),
    }
}
