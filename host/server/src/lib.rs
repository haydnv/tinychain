//! State replication management

use std::collections::{HashMap, HashSet};
use std::net::IpAddr;
use std::path::PathBuf;

use aes_gcm_siv::{Aes256GcmSiv, Key};
use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use log::{debug, info, warn};
use mdns_sd::{ServiceDaemon, ServiceEvent};
use rjwt::VerifyingKey;

use tc_value::{Link, Value};

type Actor = rjwt::Actor<Value>;
type Aes256Key = Key<Aes256GcmSiv>;

pub const SERVICE_TYPE: &'static str = "_tinychain._tcp.local.";

pub struct ServerBuilder {
    actor: Actor,
    peers: HashMap<VerifyingKey, HashSet<IpAddr>>,
    address: IpAddr,
    port: u16,
    keys: Vec<Aes256Key>,
    data_dir: PathBuf,
}

impl ServerBuilder {
    pub fn new(
        data_dir: PathBuf,
        address: IpAddr,
        port: u16,
        symmetric_keys: Vec<Aes256Key>,
    ) -> Self {
        Self {
            actor: Actor::new(Link::default().into()),
            address,
            port,
            peers: HashMap::new(),
            keys: symmetric_keys,
            data_dir,
        }
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

    pub fn load(self) -> Self {
        todo!("load cluster state from the filesystem")
    }
}
