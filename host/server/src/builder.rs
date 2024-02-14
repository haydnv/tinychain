use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;
use std::time::Duration;

use aes_gcm_siv::{Aes256GcmSiv, Key};
use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use freqfs::{Cache, DirLock, FileSave};
use log::{debug, info, warn};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use rjwt::VerifyingKey;

use tc_transact::fs;
use tc_transact::public::StateInstance;
use tc_transact::Transaction;
use tc_value::Link;
use tcgeneric::{NetworkTime, ThreadSafe};

use crate::kernel::Kernel;
use crate::server::Server;
use crate::txn::{Txn, TxnServer};
use crate::{DEFAULT_TTL, SERVICE_TYPE};

pub type Aes256Key = Key<Aes256GcmSiv>;

pub struct ServerBuilder<FE> {
    peers: HashMap<VerifyingKey, HashSet<IpAddr>>,
    address: IpAddr,
    port: u16,
    request_ttl: Duration,
    keys: Vec<Aes256Key>,
    data_dir: DirLock<FE>,
    workspace: DirLock<FE>,
    owner: Option<Link>,
    group: Option<Link>,
    secure: bool,
}

impl<FE> ServerBuilder<FE> {
    pub fn load(cache_size: usize, data_dir: PathBuf, workspace: PathBuf) -> Self
    where
        FE: for<'a> FileSave<'a>,
    {
        if !data_dir.exists() {
            panic!("there is no directory at {data_dir:?}");
        }

        if !workspace.exists() {
            std::fs::create_dir_all(&workspace).expect("workspace");
        }

        let cache = Cache::new(cache_size.into(), None);

        let data_dir = cache.clone().load(data_dir).expect("data directory");
        let workspace = cache.clone().load(workspace).expect("workspace");

        Self {
            address: Ipv4Addr::UNSPECIFIED.into(),
            port: 0,
            peers: HashMap::new(),
            keys: vec![],
            request_ttl: DEFAULT_TTL,
            data_dir,
            workspace,
            owner: None,
            group: None,
            secure: true,
        }
    }

    pub async fn build<State>(mut self) -> Server<State, FE>
    where
        State: StateInstance<FE = FE, Txn = Txn<State, FE>>,
        FE: ThreadSafe + Clone + for<'a> FileSave<'a>,
    {
        if self.secure {
            if self.owner.is_none() {
                panic!("a server without an owner cannot be secure--specify an owner or disable security");
            } else if self.group.is_none() {
                self.group = self.owner.clone();
            }
        }

        let txn_server = TxnServer::create(self.workspace, self.request_ttl);
        let txn: Txn<State, FE> = txn_server.new_txn(NetworkTime::now());
        let txn_id = *txn.id();

        let data_dir = fs::Dir::load(txn_id, self.data_dir)
            .await
            .expect("data dir");

        let kernel: Kernel<State, FE> =
            fs::Persist::load_or_create(txn_id, (self.owner, self.group), data_dir)
                .await
                .expect("kernel");

        kernel.commit(txn_id).await;

        Server::new(kernel.into(), txn_server)
    }
}

impl<FE> ServerBuilder<FE> {
    pub fn bind_address(mut self, address: IpAddr) -> Self {
        self.address = address;
        self
    }

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
