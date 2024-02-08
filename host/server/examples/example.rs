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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let hostname = gethostname::gethostname().into_string().expect("hostname");
    println!("hostname: {hostname}");

    let ifaces = local_ip_address::list_afinet_netifas().expect("network interface list");

    for (name, ip) in ifaces {
        println!("{}:\t{:?}", name, ip);
    }

    Ok(())
}
