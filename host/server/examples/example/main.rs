// use cases to consider:
//
//  1.
//  A malicious user sends a token signed with their private key which grants group write access
//  to a cluster. The cluster rejects the request as unauthorized.
//
//  2.
//  The first host in a cluster starts.
//  It leaves a TCP server open on a random port for later hosts on the local network to discover,
//  only if a shared symmetric key is configured.
//
//  3.
//  A host stays running for months without restarting. It maintains an accurate list of peers.
//
//  4.
//  A transaction causes a random, irregularly distributed set of 51% of all hosts
//  in a cluster to fail.
//
//  5.
//  A single cluster manages xx TBs of data spread across many shards. All the shards
//  operated by one public cloud provider go offline simultaneously but users experience
//  no interruption in service and no data is lost.
//
//  6.
//  One host is running on a public cloud provider.
//  A second host joins from an on-premises datacenter.
//  The list of peers and replicas is updated such that both hosts receive every write operation.

use std::collections::HashMap;
use std::sync::Arc;

use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use freqfs::Cache;
use mdns_sd::{ServiceDaemon, ServiceInfo};
use rjwt::Actor;

use tc_value::{Link, Value};

use tc_server::aes256::{Aes256GcmSiv, Key, KeyInit, OsRng};
use tc_server::{Authorize, Server, ServerBuilder, SERVICE_TYPE};
use tcgeneric::{path_label, PathLabel, TCPathBuf};

use crate::mock::{CacheBlock, Client, State};

mod mock;

const CACHE_SIZE: usize = 1_000_000;
const DATA_DIR: &'static str = "/tmp/tc/example_server/data";
const WORKSPACE: &'static str = "/tmp/tc/example_server/workspace";
const HYPOTHETICAL: PathLabel = path_label(&["txn", "hypothetical"]);

async fn create_server(name: String, key: Key) -> Server<State, CacheBlock> {
    let data_dir = format!("{DATA_DIR}/{name}").parse().unwrap();
    let workspace = format!("{WORKSPACE}/{name}").parse().unwrap();

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&workspace).unwrap();

    let cache = Cache::new(CACHE_SIZE, None);
    let data_dir = cache.clone().load(data_dir).unwrap();
    let workspace = cache.clone().load(workspace).unwrap();

    let rpc_client = Arc::new(Client::default());

    let builder = ServerBuilder::load(data_dir, workspace, rpc_client)
        .with_keys(vec![key])
        .set_secure(false);

    let builder = builder.discover().await;

    builder.build().await
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

    let key = Aes256GcmSiv::generate_key(&mut OsRng);
    let server = create_server("one".to_string(), key).await;

    let path = TCPathBuf::from(HYPOTHETICAL);
    let txn = server.get_txn(None)?;
    let (txn, endpoint) = server.authorize_claim_and_route(&path, txn)?;

    assert!(
        endpoint.umask().may_execute(),
        "insufficient permissions: {}",
        endpoint.umask()
    );

    Ok(())
}
