//! TinyChain is a distributed state machine with an HTTP + JSON API designed to provide
//! cross-service transactions across an ensemble of microservices which implement the
//! TinyChain protocol. TinyChain itself is also a Turing-complete application platform.
//!
//! TinyChain currently supports `BlockChain`, `BTree`, `Table`, and `Tensor` collection types,
//! with more planned for the future.
//!
//! TinyChain is intended to be used as an executable binary (i.e., with `cargo install`) via its
//! HTTP API. For usage instructions and more details, visit the repository page at
//! [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use futures::future::TryFutureExt;

pub use tc_btree as btree;
pub use tc_error::*;
pub use tc_table as table;
#[cfg(feature = "tensor")]
pub use tc_tensor as tensor;
pub use tc_transact as transact;
pub use tc_value as value;
pub use tcgeneric as generic;

pub mod chain;
pub mod closure;
pub mod cluster;
pub mod collection;
pub mod fs;
pub mod gateway;
pub mod kernel;
pub mod object;
pub mod scalar;
pub mod state;
pub mod stream;
pub mod txn;

mod http;
mod route;

/// The minimum size of the transactional filesystem cache, in bytes
pub const MIN_CACHE_SIZE: usize = 5000;

type TokioError = Box<dyn std::error::Error + Send + Sync + 'static>;
type UserSpace = (kernel::Class, kernel::Library, kernel::Service);

pub struct Builder {
    cache_size: usize,
    cache: Option<Arc<freqfs::Cache<fs::CacheBlock>>>,
    data_dir: PathBuf,
    gateway: Option<gateway::Config>,
    lead: Option<value::LinkHost>,
    workspace: PathBuf,
}

impl Builder {
    pub fn new(data_dir: PathBuf, workspace: PathBuf) -> Self {
        assert!(
            data_dir.exists(),
            "data directory not found: {}",
            data_dir.display()
        );

        Self::maybe_create(&workspace);

        Self {
            cache_size: 1_000_000_000, // 1GB
            cache: None,
            data_dir,
            gateway: None,
            lead: None,
            workspace,
        }
    }

    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        if cache_size < MIN_CACHE_SIZE {
            panic!(
                "{} is less than the minimum cache size of {}",
                cache_size, MIN_CACHE_SIZE
            );
        }

        self.cache_size = cache_size;
        self
    }

    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    pub fn data_dir(&self) -> &PathBuf {
        &self.data_dir
    }

    pub fn with_gateway(mut self, gateway: gateway::Config) -> Self {
        self.gateway = Some(gateway);
        self
    }

    pub fn gateway(&self) -> Option<&gateway::Config> {
        self.gateway.as_ref()
    }

    pub fn with_lead(mut self, lead: value::LinkHost) -> Self {
        self.lead = Some(lead);
        self
    }

    pub fn lead(&self) -> Option<&value::LinkHost> {
        self.lead.as_ref()
    }

    fn cache(&mut self) -> Arc<freqfs::Cache<fs::CacheBlock>> {
        if self.cache.is_none() {
            let cache = freqfs::Cache::<fs::CacheBlock>::new(
                self.cache_size.into(),
                Duration::from_secs(1),
                None,
            );

            self.cache = Some(cache);
        }

        self.cache.as_ref().expect("cache").clone()
    }

    fn maybe_create(path: &PathBuf) {
        if !path.exists() {
            log::info!(
                "directory {} does not exist, attempting to create it...",
                path.display()
            );

            std::fs::create_dir_all(path).expect("create directory hierarchy");
        }
    }

    async fn load_dir(&mut self, path: PathBuf, txn_id: transact::TxnId) -> fs::Dir {
        Self::maybe_create(&path);

        self.cache()
            .load(path)
            .map_err(fs::io_err)
            .and_then(|cache| fs::Dir::load(cache, txn_id))
            .await
            .expect("store")
    }

    async fn workspace(&mut self) -> freqfs::DirLock<fs::CacheBlock> {
        self.cache()
            .load(self.workspace.clone())
            .await
            .expect("workspace")
    }

    async fn load_or_create<T>(
        &mut self,
        txn: &txn::Txn,
        path_label: tcgeneric::PathLabel,
    ) -> cluster::Cluster<T>
    where
        cluster::Cluster<T>: transact::fs::Persist<fs::Dir, Schema = (value::Link, value::Link), Txn = txn::Txn>
            + Send
            + Sync,
    {
        use transact::fs::Persist;
        use transact::Transaction;

        let txn_id = *txn.id();

        let dir = {
            let mut path = self.data_dir.clone();
            path.extend(&path_label[..]);

            self.load_dir(path, txn_id).await
        };

        let cluster_link: value::Link = if let Some(host) = &self.lead {
            (host.clone(), path_label.into()).into()
        } else {
            path_label.into()
        };

        let self_link = txn.link(cluster_link.path().clone());

        let schema = (self_link, cluster_link);
        cluster::Cluster::<T>::load_or_create(txn_id, schema, dir.into()).expect("cluster")
    }

    async fn load_userspace(
        &mut self,
        txn_server: txn::TxnServer,
        gateway: Arc<gateway::Gateway>,
    ) -> UserSpace {
        use chain::Recover;
        use transact::Transact;

        let txn_id = transact::TxnId::new(gateway::Gateway::time());
        let token = gateway.new_token(&txn_id).expect("token");
        let txn = txn_server
            .new_txn(gateway, txn_id, token)
            .await
            .expect("transaction");

        // no need to claim ownership of this txn since there's no way to make outbound requests
        // because they would be impossible to authorize since userspace is not yet loaded
        // i.e. there is no way for other hosts to check any of these Clusters' public keys

        let class: kernel::Class = self.load_or_create(&txn, kernel::CLASS).await;
        let library: kernel::Library = self.load_or_create(&txn, kernel::LIB).await;
        let service: kernel::Service = self.load_or_create(&txn, kernel::SERVICE).await;

        futures::try_join!(
            class.recover(&txn),
            library.recover(&txn),
            service.recover(&txn),
        )
        .expect("recover userspace");

        futures::join!(
            class.commit(txn_id),
            library.commit(txn_id),
            service.commit(txn_id),
        );

        (class, library, service)
    }

    async fn bootstrap(mut self) -> (Arc<gateway::Gateway>, UserSpace) {
        let gateway_config = self.gateway().cloned().expect("gateway config");
        let workspace = self.workspace().await;

        let kernel = kernel::Kernel::bootstrap();
        let txn_server = txn::TxnServer::new(workspace).await;
        let gateway = gateway::Gateway::new(gateway_config.clone(), kernel, txn_server.clone());

        let (class, library, service) = self
            .load_userspace(txn_server.clone(), Arc::new(gateway))
            .await;

        let kernel =
            kernel::Kernel::with_userspace(class.clone(), library.clone(), service.clone());

        let gateway = Arc::new(gateway::Gateway::new(
            gateway_config,
            kernel,
            txn_server.clone(),
        ));

        (gateway, (class, library, service))
    }

    async fn replicate(gateway: Arc<gateway::Gateway>, userspace: UserSpace) -> TCResult<()> {
        let txn = gateway
            .new_txn(transact::TxnId::new(gateway::Gateway::time()), None)
            .await?;

        async fn replicate_cluster<T>(txn: &txn::Txn, cluster: cluster::Cluster<T>) -> TCResult<()>
        where
            T: cluster::Replica + transact::Transact + Send + Sync,
        {
            let txn = cluster.claim(&txn).await?;

            cluster
                .add_replica(&txn, txn.link(cluster.link().path().clone()))
                .await?;

            cluster.distribute_commit(&txn).await
        }

        futures::try_join!(
            replicate_cluster(&txn, userspace.0),
            replicate_cluster(&txn, userspace.1),
            replicate_cluster(&txn, userspace.2),
        )?;

        Ok(())
    }

    pub async fn replicate_and_serve(self) -> Result<(), TokioError> {
        let (gateway, userspace) = self.bootstrap().await;

        futures::try_join!(
            gateway.clone().listen().map_err(TokioError::from),
            Self::replicate(gateway, userspace).map_err(TokioError::from)
        )?;

        Ok(())
    }
}
