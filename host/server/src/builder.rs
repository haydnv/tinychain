use std::collections::{BTreeSet, HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;

use freqfs::DirLock;
use futures::TryFutureExt;
use gethostname::gethostname;
use log::{debug, info, warn};
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

#[cfg(feature = "service")]
use tc_state::chain::Recover;
use tc_state::CacheBlock;
use tc_transact::{fs, TxnId};
use tc_value::{Host, Link, Protocol};
use tcgeneric::NetworkTime;

use crate::aes256::Key as Aes256Key;
use crate::client::Client;
use crate::kernel::{Kernel, Schema};
use crate::server::Server;
use crate::txn::TxnServer;
use crate::{RPCClient, DEFAULT_MAX_RETRIES, DEFAULT_PORT, DEFAULT_TTL, SERVICE_TYPE};

pub struct Broadcast {
    hostname: Option<String>,
    daemon: ServiceDaemon,
    peers: HashMap<String, (HashSet<IpAddr>, u16)>,
}

impl Broadcast {
    pub fn new() -> Self {
        Self {
            hostname: None,
            daemon: ServiceDaemon::new().expect("mDNS daemon"),
            peers: HashMap::new(),
        }
    }

    pub fn hostname(&mut self) -> &str {
        if self.hostname.is_none() {
            self.hostname = gethostname().into_string().expect("hostname").into();
        }

        self.hostname.as_ref().expect("hostname")
    }

    pub fn set_hostname(mut self, hostname: String) -> Self {
        self.hostname = Some(hostname);
        self
    }

    pub fn peers(&self, protocol: Protocol) -> BTreeSet<Host> {
        let mut peers = BTreeSet::new();

        for (name, (ip_addrs, port)) in &self.peers {
            if ip_addrs.len() > 1 {
                info!("host {name} provided multiple IP addresses");
            }

            if let Some(ip_addr) = ip_addrs.into_iter().next() {
                peers.insert(Host::from((protocol, (*ip_addr).into(), *port)));
            } else {
                warn!("host {name} provided an empty address list");
            }
        }

        peers
    }

    pub async fn discover(&mut self) -> mdns_sd::Result<()> {
        let receiver = self.daemon.browse(SERVICE_TYPE)?;

        let mut search_started = false;

        loop {
            match receiver.recv_async().await {
                Ok(event) => match event {
                    ServiceEvent::SearchStarted(_params) if search_started => {
                        info!("mDNS discovered {} peers", self.peers.len());
                        break Ok(());
                    }
                    ServiceEvent::SearchStarted(params) => {
                        info!("searching for peers of {params}");
                        search_started = true;
                    }
                    ServiceEvent::ServiceFound(name, addr) => {
                        info!("discovered peer of {name} at {addr}")
                    }
                    ServiceEvent::ServiceResolved(info) => {
                        let full_name = info.get_fullname();
                        let addresses = info.get_addresses().clone();
                        let port = info.get_port();

                        self.peers.insert(full_name.to_string(), (addresses, port));

                        info!("resolved peer: {full_name}")
                    }
                    other => debug!("ignoring mDNS event: {:?}", other),
                },
                Err(cause) => warn!("mDNS error: {cause}"),
            }
        }
    }

    pub async fn make_discoverable(&mut self, host: &Host) -> mdns_sd::Result<()> {
        let hostname = self.hostname();
        let address = host.address().as_ip().expect("IP address");

        let my_service = ServiceInfo::new(
            SERVICE_TYPE,
            "one",
            &hostname,
            &address,
            host.port().unwrap_or(DEFAULT_PORT),
            HashMap::<String, String>::default(),
        )?;

        info!("registering mDNS service at {}", host);

        self.daemon.register(my_service)?;

        Ok(())
    }
}

pub struct Replicator {
    kernel: Arc<Kernel>,
    txn_server: TxnServer,
    peers: BTreeSet<Host>,
    max_retries: u8,
}

impl Replicator {
    pub fn max_retries(&self) -> u8 {
        self.max_retries
    }

    pub fn set_max_retries(mut self, max_retries: u8) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn peers(&self) -> &BTreeSet<Host> {
        &self.peers
    }

    pub fn with_peers(mut self, peers: impl IntoIterator<Item = Host>) -> Self {
        self.peers.extend(peers);
        self
    }

    pub async fn replicate_and_join(&self) -> bool {
        let mut i = 1;
        let joined = loop {
            match self
                .kernel
                .replicate_and_join(&self.txn_server, &self.peers)
                .await
            {
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

        joined
    }
}

impl<'a> From<&'a Server> for Replicator {
    fn from(server: &'a Server) -> Self {
        Self {
            kernel: server.kernel(),
            txn_server: server.txn_server(),
            peers: BTreeSet::new(),
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }
}

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
///  8. Start the public interface for the server (e.g. HTTP or HTTPS)
///  9. Discover peers via mDNS
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
    lead: Option<Host>,
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
            lead: None,
            owner: None,
            group: None,
            keys: HashSet::new(),
            secure: true,
        }
    }

    pub async fn build(mut self) -> Server {
        let host = self.host();
        let lead = self.lead.unwrap_or_else(|| host.clone());

        if self.secure {
            if self.owner.is_none() {
                panic!("a server without an owner cannot be secure--specify an owner or disable security");
            } else if self.group.is_none() {
                self.group = self.owner.clone();
            }
        }

        let txn_id = TxnId::new(NetworkTime::now());

        let data_dir = fs::Dir::load(txn_id, self.data_dir)
            .await
            .expect("data dir");

        let schema = Schema::new(lead, host.clone(), self.owner, self.group, self.keys);

        let kernel: Arc<Kernel> = fs::Persist::load_or_create(txn_id, schema, data_dir)
            .map_ok(Arc::new)
            .await
            .expect("kernel");

        kernel.commit(txn_id).await;
        kernel.finalize(&txn_id).await;

        info!("committed kernel");

        let client = Client::new(host, kernel.clone(), self.rpc_client);
        let txn_server = TxnServer::create(client, self.workspace, self.request_ttl);

        #[cfg(feature = "service")]
        {
            let txn = txn_server
                .create_txn(NetworkTime::now())
                .expect("transaction context");

            kernel.recover(&txn).await.expect("recover service state");
        }

        Server::new(kernel, txn_server).expect("server")
    }

    pub fn detect_address(mut self) -> Self {
        self.address = local_ip_address::local_ip().expect("local IP address");
        self
    }

    pub fn host(&mut self) -> Host {
        Host::from((self.protocol, self.address.into(), self.port))
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
