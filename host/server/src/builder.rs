use std::collections::{BTreeSet, HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;

use freqfs::DirLock;
use log::{debug, info, warn};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

use tc_state::CacheBlock;
use tc_transact::fs;
use tc_transact::Transaction;
use tc_value::{Address, Host, Link, Protocol};
use tcgeneric::NetworkTime;

use crate::aes256::Key as Aes256Key;
use crate::kernel::{Kernel, Schema};
use crate::server::Server;
use crate::txn::{Txn, TxnServer};
use crate::{RPCClient, DEFAULT_MAX_RETRIES, DEFAULT_PORT, DEFAULT_TTL, SERVICE_TYPE};

/// A builder struct for a [`Server`].
///
/// The expected sequence of events to bootstrap a server is:
///  1. Initialize a [`freqfs::Cache`]
///  2. Load the data directory and transactional workspace into the cache
///  3. Initialize an implementation of [`RPCClient`]
///  4. Create a new [`Builder`] with the data directory, workspace, and RPC client
///  5. Register one or more [`crate::aes256::Key`]s to use for symmetric encryption
///  6. For a secure server, provide links to the user and group authorized to administer the server
///  7. Load the kernel
///  8. Discover peers via mDNS
///  9. Start the public interface for the server (e.g. HTTP or HTTPS)
///  10. Replicate from the first peer to respond using one of the provided encryption keys
///    10.1 For each dir, replicate the entries in the dir (not their state, i.e. not recursively)
///    10.2 Repeat step 10.1 until all directory entries are replicated
///    10.3 Replicate each chain in each service
///  11. Send requests to join the replica set authenticated using the hash of the present replica state, all in a single transaction
///  12. Repeat steps 10-11 until successful
///  13. Mark the server ready to receive requests from a load balancer
///  14. Broadcast the server's availability via mDNS
///
/// See the `examples` dir for usage examples.
pub struct Builder {
    protocol: Protocol,
    address: IpAddr,
    port: u16,
    request_ttl: Duration,
    rpc_client: Arc<dyn RPCClient>,
    data_dir: DirLock<CacheBlock>,
    workspace: DirLock<CacheBlock>,
    max_retries: u8,
    owner: Option<Link>,
    group: Option<Link>,
    keys: HashSet<Aes256Key>,
    secure: bool,
}

impl Builder {
    pub fn load(
        data_dir: DirLock<CacheBlock>,
        workspace: DirLock<CacheBlock>,
        rpc_client: Arc<dyn RPCClient>,
    ) -> Self {
        Self {
            protocol: Protocol::default(),
            address: Ipv4Addr::LOCALHOST.into(),
            port: DEFAULT_PORT,
            request_ttl: DEFAULT_TTL,
            rpc_client,
            data_dir,
            workspace,
            max_retries: DEFAULT_MAX_RETRIES,
            owner: None,
            group: None,
            keys: HashSet::new(),
            secure: true,
        }
    }

    async fn build(mut self) -> Server {
        let host = self.host();

        if self.secure {
            if self.owner.is_none() {
                panic!("a server without an owner cannot be secure--specify an owner or disable security");
            } else if self.group.is_none() {
                self.group = self.owner.clone();
            }
        }

        let txn_server = TxnServer::create(self.workspace, self.rpc_client, self.request_ttl);
        let txn: Txn = txn_server.new_txn(NetworkTime::now(), None);
        let txn_id = *txn.id();

        let data_dir = fs::Dir::load(txn_id, self.data_dir)
            .await
            .expect("data dir");

        let schema = Schema::new(host, self.owner, self.group, self.keys);

        let kernel: Kernel = fs::Persist::load_or_create(txn_id, schema, data_dir)
            .await
            .expect("kernel");

        kernel.commit(txn_id).await;

        Server::new(kernel.into(), txn_server)
    }

    pub async fn start(self) -> Replicator {
        let max_retries = self.max_retries;
        let address = self.address;
        let port = self.port;

        let server = self.build().await;

        Replicator {
            address,
            port,
            max_retries,
            server,
            peers: HashMap::new(),
        }
    }

    pub fn host(&mut self) -> Host {
        Host::from((self.protocol, self.address.into()))
    }

    pub fn set_max_retries(mut self, max_retries: u8) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn set_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn set_group(mut self, group: Link) -> Self {
        self.group = Some(group);
        self
    }

    pub fn set_owner(mut self, owner: Link) -> Self {
        self.owner = Some(owner);
        self
    }

    pub fn set_secure(mut self, secure: bool) -> Self {
        self.secure = secure;
        self
    }

    pub fn with_keys<Keys: IntoIterator<Item = Aes256Key>>(mut self, keys: Keys) -> Self {
        self.keys.extend(keys);
        self
    }
}

pub struct Replicator {
    address: IpAddr,
    port: u16,
    peers: HashMap<String, HashSet<IpAddr>>,
    server: Server,
    max_retries: u8,
}

impl Replicator {
    pub async fn discover(mut self) -> Self {
        let mdns = ServiceDaemon::new().expect("Failed to create daemon");
        let receiver = mdns.browse(SERVICE_TYPE).expect("browse mDNS peers");
        let mut search_complete = false;

        loop {
            match receiver.recv_async().await {
                Ok(event) => match event {
                    ServiceEvent::SearchStarted(_params) if search_complete => break,
                    ServiceEvent::SearchStarted(params) => {
                        info!("searching for peers of {params}");
                        search_complete = true;
                    }
                    ServiceEvent::ServiceFound(name, addr) => {
                        info!("discovered peer of {name} at {addr}")
                    }
                    ServiceEvent::ServiceResolved(info) => {
                        let full_name = info.get_fullname();

                        self.peers
                            .insert(full_name.to_string(), info.get_addresses().clone());

                        info!("resolved peer: {full_name}")
                    }
                    other => debug!("ignoring mDNS event: {:?}", other),
                },
                Err(cause) => warn!("mDNS error: {cause}"),
            }
        }

        self
    }

    pub async fn replicate_and_join(self, protocol: Protocol) -> Self {
        let peers: BTreeSet<Host> = self
            .peers
            .values()
            .filter_map(|peers| {
                if peers.is_empty() {
                    None
                } else {
                    peers
                        .into_iter()
                        .next()
                        .copied()
                        .map(Address::from)
                        .map(|addr| (protocol, addr))
                        .map(Host::from)
                }
            })
            .collect();

        let mut i = 0;
        let joined = loop {
            match self.server.replicate_and_join(peers.clone()).await {
                Ok(()) => {
                    break true;
                }
                Err(progress) => {
                    if progress {
                        i = 0
                    } else if i < self.max_retries {
                        i += 1
                    } else {
                        break false;
                    }
                }
            }
        };

        if joined {
            self
        } else {
            panic!("failed to join replica set")
        }
    }

    pub async fn make_discoverable(&self) -> mdns_sd::Result<()> {
        let hostname = gethostname::gethostname().into_string().expect("hostname");

        let mdns = ServiceDaemon::new()?;

        let my_service = ServiceInfo::new(
            SERVICE_TYPE,
            &hostname,
            &hostname,
            &self.address,
            self.port,
            HashMap::<String, String>::default(),
        )?;

        mdns.register(my_service)
    }

    pub fn ready(self) -> Server {
        self.server
    }
}
