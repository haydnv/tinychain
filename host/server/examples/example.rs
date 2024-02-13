// use cases to consider:
//
//  1.
//  The first host in a cluster starts.
//  It leaves a TCP server open on a random port for later hosts on the local network to discover,
//  only if a shared symmetric key is configured.
//
//  2.
//  A host stays running for months without restarting. It maintains an accurate list of peers.
//
//  3.
//  A transaction causes a random, irregularly distributed set of 51% of all hosts
//  in a cluster to fail.
//
//  4.
//  A single cluster manages xx TBs of data spread across many shards. All the shards
//  operated by one public cloud provider go offline simultaneously but users experience
//  no interruption in service and no data is lost.
//
//  5.
//  One host is running on a public cloud provider.
//  A second host joins from an on-premises datacenter.
//  The list of peers and replicas is updated such that both hosts receive every write operation.

use std::collections::HashMap;
use std::path::PathBuf;

use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use destream::en;
use mdns_sd::{ServiceDaemon, ServiceInfo};
use rjwt::Actor;

use tc_value::{Link, Value};

use tc_server::{ServerBuilder, SERVICE_TYPE};

const CACHE_SIZE: usize = 1_000_000;
const DATA_DIR: &'static str = "/tmp/tc/example/";
const WORKSPACE: &'static str = "/tmp/tc/example/_workspace";

enum CacheBlock {}

impl<'en> en::ToStream<'en> for CacheBlock {
    fn to_stream<En: en::Encoder<'en>>(&self, encoder: En) -> Result<En::Ok, En::Error> {
        en::IntoStream::into_stream((), encoder)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // generate a keypair
    let actor = Actor::<Value>::new(Link::default().into());
    let public_key_encoded = STANDARD_NO_PAD.encode(actor.public_key());

    // characterize local network interfaces
    let hostname = gethostname::gethostname().into_string().expect("hostname");
    println!("hostname: {hostname}");

    let ifaces = local_ip_address::list_afinet_netifas().expect("network interface list");
    let mut ip_addrs = Vec::with_capacity(ifaces.len());

    let mdns = ServiceDaemon::new().expect("Failed to create daemon");

    let port = 80;

    for (name, ip) in ifaces {
        assert!(
            !ip.is_unspecified() && !ip.is_multicast(),
            "invalid network interface {name}: {ip}"
        );

        if ip.is_loopback() {
            println!("not advertising local network interface {name}: {ip}");
        } else {
            println!("will advertise network interface {name}: {ip}");
            ip_addrs.push(ip);
        }
    }

    let my_service = ServiceInfo::new(
        SERVICE_TYPE,
        &public_key_encoded,
        &hostname,
        &ip_addrs[..],
        port,
        HashMap::<String, String>::default(),
    )
    .expect("mDNS service definition");

    mdns.register(my_service).expect("register mDNS service");

    let data_dir: PathBuf = DATA_DIR.parse()?;
    std::fs::create_dir_all(&data_dir)?;

    let builder = ServerBuilder::<CacheBlock>::load(
        CACHE_SIZE,
        data_dir.clone(),
        WORKSPACE.parse().expect("workspace"),
    )
    .set_secure(false);

    builder.discover().await;

    std::fs::remove_dir_all(data_dir)?;

    Ok(())
}
