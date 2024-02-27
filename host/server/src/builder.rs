use std::collections::{BTreeSet, HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;
use std::time::Duration;

use freqfs::DirLock;
use futures::TryFutureExt;
use log::{debug, info, warn};
use mdns_sd::{ServiceEvent, ServiceInfo};

use tc_state::CacheBlock;
use tc_transact::{fs, TxnId};
use tc_value::{Address, Host, Link, Protocol};
use tcgeneric::NetworkTime;

use crate::aes256::Key as Aes256Key;
use crate::client::Client;
use crate::kernel::{Kernel, Schema};
use crate::server::Server;
use crate::txn::TxnServer;
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
            max_retries: DEFAULT_MAX_RETRIES,
            lead: None,
            owner: None,
            group: None,
            keys: HashSet::new(),
            secure: true,
        }
    }

    async fn build(mut self) -> Server {
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
        kernel.finalize(txn_id).await;

        let client = Client::new(host, kernel.clone(), self.rpc_client);
        let txn_server = TxnServer::create(client, self.workspace, self.request_ttl);

        Server::new(kernel, txn_server).expect("server")
    }

    pub fn detect_address(mut self) -> Self {
        self.address = local_ip_address::local_ip().expect("local IP address");
        self
    }

    pub async fn start(self) -> Replicator {
        let max_retries = self.max_retries;
        let host = Host::from((self.protocol, self.address.into(), self.port));

        let server = self.build().await;

        Replicator {
            host,
            max_retries,
            server,
            peers: HashMap::new(),
        }
    }

    pub fn host(&mut self) -> Host {
        Host::from((self.protocol, self.address.into(), self.port))
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
    host: Host,
    peers: HashMap<String, (HashSet<IpAddr>, u16)>,
    server: Server,
    max_retries: u8,
}

impl Replicator {
    pub fn host(&self) -> &Host {
        &self.host
    }

    pub fn peers(&self) -> &HashMap<String, (HashSet<IpAddr>, u16)> {
        &self.peers
    }

    pub async fn discover(&mut self) -> mdns_sd::Result<()> {
        let receiver = self.server.mdns().browse(SERVICE_TYPE)?;
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

    pub async fn replicate_and_join(&self, protocol: Protocol) -> bool {
        let peers: BTreeSet<Host> = self
            .peers
            .values()
            .filter_map(|(peers, port)| {
                if peers.is_empty() {
                    None
                } else {
                    peers
                        .into_iter()
                        .next()
                        .copied()
                        .map(Address::from)
                        .map(|addr| (protocol, addr, *port))
                        .map(Host::from)
                }
            })
            .collect();

        let mut i = 1;
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

        joined
    }

    pub async fn make_discoverable(&mut self) -> mdns_sd::Result<()> {
        let hostname = gethostname::gethostname().into_string().expect("hostname");
        let address = self.host.address().as_ip().expect("IP address");

        let my_service = ServiceInfo::new(
            SERVICE_TYPE,
            "one",
            &hostname,
            &address,
            self.host.port().expect("port"),
            HashMap::<String, String>::default(),
        )?;

        info!("registering mDNS service at {}", self.host);

        self.server.mdns().register(my_service)?;

        Ok(())
    }

    pub fn ready(self) -> Server {
        self.server
    }
}
