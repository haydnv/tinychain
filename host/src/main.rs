use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;

use destream::de::FromStream;
use futures::{future, stream};
use structopt::StructOpt;
use tokio::time::Duration;

use tc_error::*;
use tc_transact::{Transact, TxnId};

use tinychain::gateway::Gateway;
use tinychain::object::InstanceClass;
use tinychain::*;

fn data_size(flag: &str) -> TCResult<u64> {
    if flag.is_empty() {
        return Err(TCError::bad_request("Invalid size specified", flag));
    }

    let msg = "Unable to parse value";
    let size = u64::from_str_radix(&flag[0..flag.len() - 1], 10)
        .map_err(|_| TCError::bad_request(msg, flag))?;

    if flag.ends_with('K') {
        Ok(size * 1000)
    } else if flag.ends_with('M') {
        Ok(size * 1_000_000)
    } else {
        Err(TCError::bad_request("Unable to parse request_limit", flag))
    }
}

fn duration(flag: &str) -> TCResult<Duration> {
    u64::from_str(flag)
        .map(Duration::from_secs)
        .map_err(|_| TCError::bad_request("Invalid duration", flag))
}

#[derive(Clone, StructOpt)]
struct Config {
    #[structopt(long = "address", default_value = "127.0.0.1")]
    pub address: IpAddr,

    #[structopt(long = "log_level", default_value = "warn")]
    pub log_level: String,

    #[structopt(long = "workspace", default_value = "/tmp/tc/tmp")]
    pub workspace: PathBuf,

    #[structopt(long = "cache_size", default_value = "10M", parse(try_from_str = data_size))]
    pub cache_size: u64,

    #[structopt(long = "data_dir")]
    pub data_dir: Option<PathBuf>,

    #[structopt(long = "cluster")]
    pub clusters: Vec<PathBuf>,

    #[structopt(long = "request_ttl", default_value = "30", parse(try_from_str = duration))]
    pub request_ttl: Duration,

    #[structopt(long = "http_port", default_value = "8702")]
    pub http_port: u16,

    #[structopt(long = "peer")]
    pub peers: Vec<IpAddr>,
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Config::from_args();
    let gateway_config = config.gateway();

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(config.log_level))
        .init();

    let (workspace, data_dir) =
        mount(config.workspace.clone(), config.data_dir, config.cache_size).await?;

    let txn_server = tinychain::txn::TxnServer::new(workspace).await;

    let mut clusters = Vec::with_capacity(config.clusters.len());
    if !config.clusters.is_empty() {
        let txn_id = TxnId::new(Gateway::time());

        let data_dir = data_dir.ok_or_else(|| {
            TCError::internal("the --data_dir option is required to host a Cluster")
        })?;

        for path in config.clusters {
            let config = tokio::fs::read(&path)
                .await
                .expect(&format!("read from {:?}", &path));

            let mut decoder =
                destream_json::de::Decoder::from_stream(stream::once(future::ready(Ok(config))));

            let cluster = match InstanceClass::from_stream((), &mut decoder).await {
                Ok(class) => cluster::instantiate(class, data_dir.clone(), txn_id).await?,
                Err(cause) => panic!("error parsing cluster config {:?}: {}", path, cause),
            };

            clusters.push(cluster);
        }

        data_dir.commit(&txn_id).await;
    }

    let kernel = tinychain::Kernel::new(clusters);
    let gateway = tinychain::gateway::Gateway::new(gateway_config, kernel, txn_server);

    gateway.replicate(config.peers).await?;

    if let Err(cause) = gateway.listen().await {
        log::error!("server error: {}", cause);
    }

    if config.workspace.exists() {
        use futures::TryFutureExt;
        tokio::fs::remove_dir_all(config.workspace)
            .map_err(|e| {
                let err: Box<dyn std::error::Error + Send + Sync + 'static> = Box::new(e);
                err
            })
            .await?
    }

    Ok(())
}
