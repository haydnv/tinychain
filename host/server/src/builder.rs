use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use freqfs::DirLock;
use log::{debug, info, warn};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use rjwt::VerifyingKey;
use tc_state::CacheBlock;

use tc_transact::fs;
use tc_transact::Transaction;
use tc_value::Link;
use tcgeneric::NetworkTime;

use crate::aes256::Key as Aes256Key;
use crate::kernel::Kernel;
use crate::server::Server;
use crate::txn::{Txn, TxnServer};
use crate::{RPCClient, State, DEFAULT_TTL, SERVICE_TYPE};

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
///    10.2 Repeat step 9.1 until all directory entries are replicated
///    10.3 Replicate each chain in each service
///  11. Send requests to join the replica set authenticated using the hash of the present replica state, all in a single transaction
///  12. Repeat steps 10-11 until successful
///  13. Mark the server ready to receive requests from a load balancer
///  14. Broadcast the server's availability via mDNS
///
/// See the `examples` dir for usage examples.
pub struct Builder {
    peers: HashMap<VerifyingKey, HashSet<IpAddr>>,
    request_ttl: Duration,
    rpc_client: Arc<dyn RPCClient<State>>,
    keys: Vec<Aes256Key>,
    data_dir: DirLock<CacheBlock>,
    workspace: DirLock<CacheBlock>,
    owner: Option<Link>,
    group: Option<Link>,
    secure: bool,
}

impl Builder {
    pub fn load(
        data_dir: DirLock<CacheBlock>,
        workspace: DirLock<CacheBlock>,
        rpc_client: Arc<dyn RPCClient<State>>,
    ) -> Self {
        Self {
            peers: HashMap::new(),
            keys: vec![],
            request_ttl: DEFAULT_TTL,
            rpc_client,
            data_dir,
            workspace,
            owner: None,
            group: None,
            secure: true,
        }
    }

    pub async fn build(mut self) -> Server {
        if self.secure {
            if self.owner.is_none() {
                panic!("a server without an owner cannot be secure--specify an owner or disable security");
            } else if self.group.is_none() {
                self.group = self.owner.clone();
            }
        }

        let txn_server = TxnServer::create(self.workspace, self.request_ttl);
        let txn: Txn = txn_server.new_txn(NetworkTime::now());
        let txn_id = *txn.id();

        let data_dir = fs::Dir::load(txn_id, self.data_dir)
            .await
            .expect("data dir");

        let kernel: Kernel =
            fs::Persist::load_or_create(txn_id, (self.owner, self.group), data_dir)
                .await
                .expect("kernel");

        kernel.commit(txn_id).await;

        Server::new(kernel.into(), txn_server)
    }
}

impl Builder {
    // pub fn address(&mut self) -> IpAddr {
    //     if self.address.is_none() {
    //         let ifaces = local_ip_address::list_afinet_netifas().expect("network interface list");
    //
    //         self.address = ifaces
    //             .into_iter()
    //             .inspect(|(name, address)| {
    //                 assert!(
    //                     !address.is_unspecified() && !address.is_multicast(),
    //                     "invalid network interface {name}: {address}"
    //                 );
    //             })
    //             .filter_map(|(_name, address)| {
    //                 if address.is_loopback() {
    //                     None
    //                 } else {
    //                     Some(address)
    //                 }
    //             })
    //             .next();
    //     }
    //
    //     self.address.expect("IP address")
    // }

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
                        let peer_key = &full_name[..full_name.len() - SERVICE_TYPE.len() - 1];
                        let peer_key = STANDARD_NO_PAD.decode(peer_key).expect("key");
                        let peer_key = VerifyingKey::try_from(&peer_key[..]).expect("key");
                        self.peers.insert(peer_key, info.get_addresses().clone());
                        info!("resolved peer: {full_name}")
                    }
                    other => debug!("ignoring mDNS event: {:?}", other),
                },
                Err(cause) => warn!("mDNS error: {cause}"),
            }
        }

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
