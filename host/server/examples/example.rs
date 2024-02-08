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

use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};
use rjwt::{Actor, VerifyingKey};

use tc_value::{Link, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // generate a keypair
    let actor = Actor::<Value>::new(Link::default().into());
    let public_key_encoded = STANDARD_NO_PAD.encode(actor.public_key());

    // characterize local network interfaces
    let hostname = gethostname::gethostname().into_string().expect("hostname");
    println!("hostname: {hostname}");

    let ifaces = local_ip_address::list_afinet_netifas().expect("network interface list");

    for (name, ip) in &ifaces {
        println!("{}:\t{:?}", name, ip);
    }

    let mdns = ServiceDaemon::new().expect("Failed to create daemon");

    let service_type = "_tinychain._tcp.local.";
    let port = 80;
    let properties: [(&str, &str); 0] = [];

    for (name, ip) in ifaces {
        assert!(
            !ip.is_unspecified() && !ip.is_multicast(),
            "invalid network interface {name}: {ip}"
        );

        if ip.is_loopback() {
            println!("not advertising local network interface {name}: {ip}");
            continue;
        } else {
            println!("advertising network interface {name}: {ip}");
        }

        let my_service = ServiceInfo::new(
            service_type,
            &public_key_encoded,
            &hostname,
            ip,
            port,
            &properties[..],
        )
        .expect("mDNS service definition");

        mdns.register(my_service).expect("register mDNS service");
    }

    let receiver = mdns.browse(service_type).expect("browse mDNS peers");
    let mut search_complete = false;

    loop {
        match receiver.recv_async().await {
            Ok(event) => match event {
                ServiceEvent::SearchStarted(_params) if search_complete => break,
                ServiceEvent::SearchStarted(params) => {
                    println!("searching for peers of {params}");
                    search_complete = true;
                },
                ServiceEvent::ServiceFound(name, addr) => println!("discovered peer of {name} at {addr}"),
                ServiceEvent::ServiceResolved(info) => {
                    let full_name = info.get_fullname();
                    let peer_key = &full_name[..full_name.len() - service_type.len() - 1];
                    let peer_key = STANDARD_NO_PAD.decode(peer_key).expect("key");
                    let peer_key = VerifyingKey::try_from(&peer_key[..]).expect("key");
                    println!("resolved peer: {full_name}")
                },
                other => println!("ignoring mDNS event: {:?}", other),
            },
            Err(cause) => println!("mDNS error: {cause}"),
        }
    }

    Ok(())
}
