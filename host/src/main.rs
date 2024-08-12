use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use clap::Parser;
use freqfs::Cache;
use log::info;
use tokio::time::Duration;

use tc_error::*;
use tc_server::aes256::Key;
use tc_server::{Broadcast, Builder, Replicator};
use tc_value::{Host, Protocol};

use tinychain::{http, MIN_CACHE_SIZE};

fn data_size(flag: &str) -> TCResult<usize> {
    if flag.is_empty() || flag == "0" {
        return Ok(0);
    }

    let size = usize::from_str_radix(&flag[0..flag.len() - 1], 10)
        .map_err(|_| TCError::unexpected(flag, "a data size"))?;

    if flag.ends_with('K') {
        Ok(size * 1000)
    } else if flag.ends_with('M') {
        Ok(size * 1_000_000)
    } else if flag.ends_with('G') {
        Ok(size * 1_000_000_000)
    } else {
        Err(TCError::unexpected(
            &flag[flag.len() - 1..],
            "a data size suffix",
        ))
    }
}

fn duration(flag: &str) -> TCResult<Duration> {
    u64::from_str(flag)
        .map(Duration::from_secs)
        .map_err(|_| TCError::unexpected(flag, "a duration"))
}

#[derive(Clone, Parser)]
struct Config {
    #[arg(long, help = "the IP address of this host")]
    pub address: Option<IpAddr>,

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
        long = "insecure",
        help = "disable security checks when installing new services"
    )]
    pub insecure: bool,

    #[arg(
        long = "log_level",
        default_value = "info",
        value_parser = ["trace", "debug", "info", "warn", "error"],
        help = "the log message level to write",
    )]
    pub log_level: String,

    #[arg(
        long = "symmetric_key",
        help = "a hexadecimal string representations of amn AES256 key used for replication at startup"
    )]
    pub keys: Vec<String>,

    #[arg(
        long = "peer",
        help = "the address of one or more peers to replicate from"
    )]
    pub peers: Vec<Host>,

    #[cfg(debug_assertions)]
    #[arg(
        long = "request_ttl",
        value_parser = duration,
        default_value = "10",
        help = "maximum allowed request duration (in seconds)",
    )]
    pub request_ttl: Duration,

    #[cfg(not(debug_assertions))]
    #[arg(
        long = "request_ttl",
        value_parser = duration,
        default_value = "3",
        help = "maximum allowed request duration (in seconds)",
    )]
    pub request_ttl: Duration,

    #[arg(
        long = "stack_size",
        value_parser = data_size,
        default_value = "48M",
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

fn main() {
    let mut config = Config::parse();

    let keys = config
        .keys
        .into_iter()
        .map(|hex_key| {
            let key = hex::decode(&hex_key).expect("AES256 key");

            assert_eq!(
                key.len(),
                32,
                "invalid AES256 key: {hex_key} ({} bytes but should be 32)",
                key.len()
            );

            Key::from_slice(key.as_slice()).clone()
        })
        .collect::<Vec<Key>>();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(&config.log_level))
        .init();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .thread_stack_size(config.stack_size)
        .build()
        .expect("tokio runtime");

    if !config.workspace.exists() {
        std::fs::create_dir_all(&config.workspace).expect("create workspace");
    }

    if config.cache_size < MIN_CACHE_SIZE {
        panic!(
            "cache size {} is below the minimum of {}",
            config.cache_size, MIN_CACHE_SIZE
        );
    }

    let (data_dir, workspace) = rt
        .block_on(async move {
            let cache = Cache::new(config.cache_size, None);
            let data_dir = cache.clone().load(config.data_dir)?;
            let workspace = cache.clone().load(config.workspace)?;
            TCResult::Ok((data_dir, workspace))
        })
        .expect("cache load");

    let rpc_client = Arc::new(http::Client::new());

    let builder = Builder::load(data_dir, workspace, rpc_client)
        .detect_address()
        .set_port(config.http_port)
        .with_keys(keys)
        .set_secure(false);

    let mut broadcast = Broadcast::new();

    let peers = if config.peers.is_empty() {
        rt.block_on(broadcast.discover())
            .expect("mDNS peer discovery");

        broadcast.peers(Protocol::default())
    } else {
        config.peers.drain(..).collect()
    };

    let app_server = rt.block_on(builder.build());
    let address = app_server.address().clone();

    let http_io_task = {
        let replicator = Replicator::from(&app_server);

        let http_server = http::Server::new(app_server, config.request_ttl);
        let http_io_task = rt.spawn(http_server.listen(config.http_port));

        if !peers.is_empty() {
            info!("attempting to replicate from peers: {:?}", peers);

            assert!(
                rt.block_on(replicator.with_peers(peers).replicate_and_join()),
                "replication failed"
            );
        };

        http_io_task
    };

    rt.block_on(broadcast.make_discoverable(&address))
        .expect("mDNS daemon");

    rt.block_on(http_io_task)
        .expect("server shutdown")
        .expect("server");
}
