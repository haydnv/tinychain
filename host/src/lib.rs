//! TinyChain is a distributed state machine with an HTTP + JSON API designed to provide
//! cross-service transactions across an ensemble of microservices which implement the
//! TinyChain protocol.
//!
//! TinyChain currently supports `BlockChain`, `BTree`, `Table`, and `Tensor` collection types,
//! with more planned for the future.
//!
//! TinyChain is intended to be used as an executable binary (i.e., with `cargo install`) via its
//! HTTP API. For usage instructions and more details, visit the repository page at
//! [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain).

use std::path::PathBuf;
use std::sync::Arc;

use futures::future::TryFutureExt;

pub use tc_error::*;

pub mod chain;
pub mod closure;
pub mod cluster;
pub mod collection;
pub mod fs;
pub mod gateway;
pub mod kernel;
pub mod object;
pub mod state;
// pub mod stream;
pub mod txn;

mod http;
mod route;

/// The minimum size of the transactional filesystem cache, in bytes
pub const MIN_CACHE_SIZE: usize = 5000;

type TokioError = Box<dyn std::error::Error + Send + Sync + 'static>;
// type UserSpace = (kernel::Class, kernel::Library, kernel::Service);

/// Build a new host.
pub struct Builder {
    cache: Arc<freqfs::Cache<fs::CacheBlock>>,
    data_dir: PathBuf,
    gateway: Option<gateway::Config>,
    lead: Option<tc_value::Host>,
    public_key: Option<bytes::Bytes>,
    workspace: freqfs::DirLock<fs::CacheBlock>,
}

impl Builder {
    /// Load the transactional filesystem cache.
    pub async fn load(cache_size: usize, data_dir: PathBuf, workspace: PathBuf) -> Self {
        assert!(
            data_dir.exists(),
            "data directory not found: {}",
            data_dir.display()
        );

        Self::maybe_create(&workspace);

        let cache = freqfs::Cache::<fs::CacheBlock>::new(cache_size.into(), None);

        let workspace = cache.clone().load(workspace).expect("workspace");

        Self {
            cache,
            data_dir,
            gateway: None,
            lead: None,
            public_key: None,
            workspace,
        }
    }

    /// Specify the [`gateway::Config`] of this host.
    pub fn with_gateway(mut self, gateway: gateway::Config) -> Self {
        self.gateway = Some(gateway);
        self
    }

    /// Specify the host to replicate from (if any).
    pub fn with_lead(mut self, lead: Option<tc_value::Host>) -> Self {
        self.lead = lead;
        self
    }

    /// Specify the public key of the cluster to join (if any).
    pub fn with_public_key(mut self, public_key: Option<String>) -> Self {
        if let Some(public_key) = public_key {
            let public_key = hex::decode(public_key).expect("public key");

            let len = public_key.len();
            assert_eq!(len, 32, "an Ed25519 public key has 32 bytes, not {}", len);

            self.public_key = Some(public_key.into())
        }

        self
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

    async fn load_dir(&self, path: PathBuf, txn_id: tc_transact::TxnId) -> fs::Dir {
        Self::maybe_create(&path);

        log::debug!("load {} into cache", path.display());
        let cache = self.cache.clone().load(path).expect("cache dir");

        log::debug!("load {:?} into the transactional filesystem", cache);
        fs::Dir::load(txn_id, cache).await.expect("store")
    }

    // async fn load_or_create<T>(
    //     &self,
    //     txn: &txn::Txn,
    //     path_label: tcgeneric::PathLabel,
    // ) -> cluster::Cluster<T>
    // where
    //     cluster::Cluster<T>: tc_transact::fs::Persist<fs::CacheBlock, Schema = cluster::Schema, Txn = txn::Txn>
    //         + Send
    //         + Sync,
    // {
    //     use tc_transact::fs::Persist;
    //     use tc_transact::Transaction;
    //
    //     let txn_id = *txn.id();
    //     let host = self.gateway.as_ref().expect("gateway config").host();
    //
    //     let dir = {
    //         let mut path = self.data_dir.clone();
    //         path.extend(&path_label[..]);
    //
    //         self.load_dir(path, txn_id).await
    //     };
    //
    //     log::debug!("loaded {:?}", dir);
    //
    //     let actor_id = tcgeneric::TCPathBuf::default().into();
    //     let actor = if let Some(public_key) = &self.public_key {
    //         txn::Actor::with_public_key(actor_id, public_key)
    //             .map(Arc::new)
    //             .expect("actor")
    //     } else {
    //         Arc::new(txn::Actor::new(actor_id))
    //     };
    //
    //     let schema = cluster::Schema::new(host, path_label.into(), self.lead.clone(), actor);
    //     cluster::Cluster::<T>::load_or_create(txn_id, schema, dir.into())
    //         .await
    //         .expect("cluster")
    // }

    // async fn load_userspace(
    //     &self,
    //     txn_server: txn::TxnServer,
    //     gateway: Arc<gateway::Gateway>,
    // ) -> UserSpace {
    //     use chain::Recover;
    //     use tc_transact::Transact;
    //
    //     let txn_id = tc_transact::TxnId::new(gateway::Gateway::time());
    //     let token = gateway.new_token(&txn_id).expect("token");
    //     let txn = txn_server
    //         .new_txn(gateway, txn_id, token)
    //         .await
    //         .expect("transaction");
    //
    //     // no need to claim ownership of this txn since there's no way to make outbound requests
    //     // because they would be impossible to authorize since userspace is not yet loaded
    //     // i.e. there is no way for other hosts to check any of these Clusters' public keys
    //
    //     let class: kernel::Class = self.load_or_create(&txn, kernel::CLASS).await;
    //     let library: kernel::Library = self.load_or_create(&txn, kernel::LIB).await;
    //     let service: kernel::Service = self.load_or_create(&txn, kernel::SERVICE).await;
    //
    //     futures::try_join!(
    //         class.recover(&txn),
    //         library.recover(&txn),
    //         service.recover(&txn),
    //     )
    //     .expect("recover userspace");
    //
    //     futures::join!(
    //         class.commit(txn_id),
    //         library.commit(txn_id),
    //         service.commit(txn_id),
    //     );
    //
    //     (class, library, service)
    // }

    // async fn bootstrap(self) -> (Arc<gateway::Gateway>, UserSpace) {
    //     let gateway_config = self.gateway.clone().expect("gateway config");
    //
    //     let kernel = kernel::Kernel::bootstrap();
    //     let txn_server = txn::TxnServer::new(self.workspace.clone()).await;
    //     let gateway = gateway::Gateway::new(gateway_config.clone(), kernel, txn_server.clone());
    //
    //     let (class, library, service) = self.load_userspace(txn_server.clone(), gateway).await;
    //
    //     let kernel =
    //         kernel::Kernel::with_userspace(class.clone(), library.clone(), service.clone());
    //
    //     let gateway = gateway::Gateway::new(gateway_config, kernel, txn_server.clone());
    //
    //     (gateway, (class, library, service))
    // }

    // async fn replicate(gateway: Arc<gateway::Gateway>, userspace: UserSpace) -> TCResult<()> {
    //     let txn = gateway
    //         .new_txn(tc_transact::TxnId::new(gateway::Gateway::time()), None)
    //         .await?;
    //
    //     async fn replicate_cluster<T>(txn: &txn::Txn, cluster: cluster::Cluster<T>) -> TCResult<()>
    //     where
    //         T: cluster::Replica + tc_transact::Transact + Send + Sync,
    //     {
    //         let txn = cluster.claim(&txn).await?;
    //
    //         cluster.add_replica(&txn, txn.host().clone()).await?;
    //
    //         cluster.distribute_commit(&txn).await
    //     }
    //
    //     futures::try_join!(
    //         replicate_cluster(&txn, userspace.0),
    //         replicate_cluster(&txn, userspace.1),
    //         replicate_cluster(&txn, userspace.2),
    //     )?;
    //
    //     Ok(())
    // }

    //    /// Start a server and replicate its state from the lead replica, if any.
    // pub async fn replicate_and_serve(self) -> Result<(), TokioError> {
    //     let (gateway, userspace) = self.bootstrap().await;
    //
    //     futures::try_join!(
    //         gateway.clone().listen().map_err(TokioError::from),
    //         Self::replicate(gateway, userspace).map_err(TokioError::from)
    //     )?;
    //
    //     Ok(())
    // }
}
