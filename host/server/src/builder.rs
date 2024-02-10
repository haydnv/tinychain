use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr};
use std::path::PathBuf;

use aes_gcm_siv::{Aes256GcmSiv, Key};
use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use log::{debug, info, warn};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use rjwt::VerifyingKey;

pub type Aes256Key = Key<Aes256GcmSiv>;

pub const SERVICE_TYPE: &'static str = "_tinychain._tcp.local.";

pub struct ServerBuilder {
    peers: HashMap<VerifyingKey, HashSet<IpAddr>>,
    address: IpAddr,
    port: u16,
    keys: Vec<Aes256Key>,
    data_dir: PathBuf,
}

impl ServerBuilder {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            address: Ipv4Addr::UNSPECIFIED.into(),
            port: 0,
            peers: HashMap::new(),
            keys: vec![],
            data_dir,
        }
    }

    pub fn bind_address(mut self, address: IpAddr) -> Self {
        self.address = address;
        self
    }

    pub fn set_port(mut self, port: u16) -> Self {
        self.port = port;
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

    pub fn with_keys<Keys: IntoIterator<Item = Aes256Key>>(mut self, keys: Keys) -> Self {
        self.keys.extend(keys);
        self
    }

    pub fn load_kernel(self) -> Self {
        todo!("load cluster state from the filesystem")
    }
}
