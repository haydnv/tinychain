use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

use bytes::Bytes;
use clap::Parser;
use destream::de::FromStream;
use futures::future::{self, TryFutureExt};
use futures::stream;
use futures::try_join;
use tokio::time::Duration;

use tc_error::*;
use tc_transact::{Transact, TxnId};
use tc_value::{LinkHost, LinkProtocol};

use tinychain::gateway::Gateway;
use tinychain::object::InstanceClass;
use tinychain::*;

type TokioError = Box<dyn std::error::Error + Send + Sync + 'static>;

const MIN_CACHE_SIZE: usize = 5000;

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
    #[arg(long, default_value = "0.0.0", help = "the IP address to bind")]
    pub address: IpAddr,

    #[arg(
        long = "cache_size",
        value_parser = data_size,
        default_value = "1G",
        help = "the maximum size of the in-memory transactional filesystem cache (in bytes)",
    )]
    pub cache_size: usize,

    // TODO: delete
    #[arg(
        long = "cluster",
        help = "(DEPRECATED) path(s) to cluster configuration files (this flag can be repeated)"
    )]
    pub clusters: Vec<PathBuf>,

    #[arg(
        long = "data_dir",
        help = "the directory to use for persistent data storage"
    )]
    pub data_dir: Option<PathBuf>,

    #[arg(
        long = "http_port",
        default_value = "8702",
        help = "the port for the HTTP server to bind"
    )]
    pub http_port: u16,

    #[arg(
        long = "log_level",
        default_value = "warn",
        value_parser = ["trace", "debug", "warn", "error"],
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

// TODO: split into startup, run, and shutdown functions
async fn load_and_serve(config: Config) -> Result<(), TokioError> {
    let gateway_config = config.gateway();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(config.log_level))
        .init();

    if !config.workspace.exists() {
        log::info!(
            "workspace directory {:?} does not exist, attempting to create it...",
            config.workspace
        );

        std::fs::create_dir_all(&config.workspace)?;
    }

    if config.cache_size < MIN_CACHE_SIZE {
        return Err(TCError::bad_request("the minimum cache size is", MIN_CACHE_SIZE).into());
    }

    let cache = freqfs::Cache::new(config.cache_size.into(), Duration::from_secs(1), None);
    let workspace = cache.clone().load(config.workspace).await?;
    let txn_id = TxnId::new(Gateway::time());

    let data_dir = if let Some(data_dir) = config.data_dir {
        if !data_dir.exists() {
            panic!("{:?} does not exist--create it or provide a different path for the --data_dir flag", data_dir);
        }

        let data_dir = cache.load(data_dir).await?;
        tinychain::fs::Dir::load(data_dir, txn_id).await
    } else {
        Err(TCError::internal("the --data-dir option is required"))
    }?;

    #[cfg(feature = "tensor")]
    {
        tc_tensor::print_af_info();
        println!();
    }

    let txn_server = tinychain::txn::TxnServer::new(workspace).await;

    let kernel = tinychain::Kernel::bootstrap();
    let gateway = Gateway::new(gateway_config.clone(), kernel, txn_server.clone());
    let token = gateway.new_token(&txn_id)?;
    let txn = txn_server.new_txn(gateway, txn_id, token).await?;

    // TODO: delete
    let mut clusters = Vec::with_capacity(config.clusters.len());
    if !config.clusters.is_empty() {
        let host = LinkHost::from((
            LinkProtocol::HTTP,
            config.address.clone(),
            Some(config.http_port),
        ));

        for path in config.clusters {
            let config = tokio::fs::read(&path)
                .await
                .expect(&format!("read from {:?}", &path));

            let mut decoder = destream_json::de::Decoder::from_stream(stream::once(future::ready(
                Ok(Bytes::from(config)),
            )));

            let cluster = match InstanceClass::from_stream((), &mut decoder).await {
                Ok(class) => {
                    cluster::instantiate(&txn, host.clone(), class, data_dir.clone()).await?
                }
                Err(cause) => panic!("error parsing cluster config {:?}: {}", path, cause),
            };

            clusters.push(cluster);
        }
    }

    let library: kernel::Library = {
        let link = if let Some(host) = config.replicate {
            (host, kernel::LIB.into()).into()
        } else {
            kernel::LIB.into()
        };

        let store = data_dir
            .get_or_create_store(txn_id, tcgeneric::label(kernel::LIB[0]).into())
            .await?;

        tc_transact::fs::Persist::load(&txn, link, store).await?
    };

    data_dir.commit(&txn_id).await;

    let kernel = tinychain::Kernel::with_userspace(library.clone(), clusters);
    let gateway = tinychain::gateway::Gateway::new(gateway_config, kernel, txn_server);

    log::info!(
        "starting server, stack size is {}, cache size is {}",
        config.stack_size,
        config.cache_size
    );

    try_join!(
        gateway.clone().listen().map_err(TokioError::from),
        replicate(gateway, library).map_err(TokioError::from)
    )?;

    Ok(())
}

async fn replicate(gateway: Arc<Gateway>, library: Library) -> TCResult<()> {
    let txn = gateway.new_txn(TxnId::new(Gateway::time()), None).await?;
    let txn = library.claim(&txn).await?;

    library
        .add_replica(&txn, txn.link(library.link().path().clone()))
        .await?;

    library.distribute_commit(&txn).await
}

fn main() {
    let config = Config::parse();

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .thread_stack_size(config.stack_size)
        .build()
        .expect("tokio runtime");

    match runtime.block_on(load_and_serve(config)) {
        Ok(_) => {}
        Err(cause) => panic!("HTTP server failed: {}", cause),
    }
}
