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
//
//  7.
//  A malicious transaction participant sends a commit message downstream.
//  The downstream participant rejects the commit.
//
//  8.
//  A malicious transaction leader sends commit messages to some dependencies but not others.
//  The messages are rejected.
//
//  9.
//  A malicious transaction leader sends commit messages when it receives a rollback message,
//  and vice versa.
//  The messages are rejected.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::Cache;
use futures::{FutureExt, TryFutureExt};
use rjwt::{Actor, Error, Resolve, VerifyingKey};
use tokio::sync::RwLock;

use tc_error::*;
use tc_scalar::{OpDef, Scalar};
use tc_server::aes256::{Aes256GcmSiv, Key, KeyInit, OsRng};
use tc_server::{Builder, Claim, RPCClient, Server, State, Txn, DEFAULT_PORT};
use tc_transact::Transaction;
use tc_value::{Host, Link, Protocol, ToUrl, Value};
use tcgeneric::{label, path_label, Id, Map, PathLabel, TCPath, TCPathBuf};

const CACHE_SIZE: usize = 1_000_000;
const DATA_DIR: &'static str = "/tmp/tc/example_server/data";
const WORKSPACE: &'static str = "/tmp/tc/example_server/workspace";
const HYPOTHETICAL: PathLabel = path_label(&["txn", "hypothetical"]);

#[derive(Default)]
struct Client {
    servers: RwLock<HashMap<Host, Server>>,
}

impl Client {
    fn add(&self, host: Host, server: Server) {
        let mut servers = self.servers.try_write().unwrap();
        assert!(servers.insert(host, server).is_none());
    }

    fn get_txn(&self, host: &Host) -> TCResult<Txn> {
        let servers = self.servers.try_read().expect("servers");
        let server = servers.get(host).expect("server");
        server.create_txn()
    }

    fn get_server_for<'a>(
        servers: &'a HashMap<Host, Server>,
        link: &ToUrl<'a>,
    ) -> TCResult<&'a Server> {
        let host = link.host().ok_or_else(|| {
            bad_request!("RPC to {} is missing a host", TCPath::from(link.path()))
        })?;

        let port = host.port().ok_or_else(|| {
            bad_request!("RPC to {} is missing a port", TCPath::from(link.path()))
        })?;

        servers
            .get(host)
            .ok_or_else(|| not_found!("server at {host}:{port}"))
    }
}

#[async_trait]
impl Resolve for Client {
    type HostId = Link;
    type ActorId = Value;
    type Claims = Claim;

    async fn resolve(
        &self,
        host: &Self::HostId,
        actor_id: &Self::ActorId,
    ) -> Result<Actor<Self::ActorId>, Error> {
        let link = ToUrl::from(host);
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link).map_err(Error::fetch)?;
        let txn = server.create_txn().map_err(Error::fetch)?;

        let public_key = self
            .get(&txn, link, actor_id.clone())
            .map(|result| {
                result
                    .and_then(Value::try_from)
                    .and_then(Arc::<[u8]>::try_from)
            })
            .map_err(|cause| Error::auth(format!("{cause} (for {actor_id} at {host}")))
            .await?;

        VerifyingKey::try_from(&*public_key)
            .map(|public_key| Actor::with_public_key(actor_id.clone(), public_key))
            .map_err(|cause| Error::auth(format!("{cause} (for {actor_id} at {host})")))
    }
}

#[async_trait]
impl RPCClient for Client {
    async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;

        let txn_id = *txn.id();
        let txn = server.get_txn(Some(txn_id), self.extract_jwt(txn)).await?;

        let endpoint = server.authorize_claim_and_route(link.path(), &txn)?;
        let handler = endpoint.get(key)?;
        handler.await
    }

    async fn put(&self, txn: &Txn, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;

        let txn_id = *txn.id();
        let txn = server.get_txn(Some(txn_id), self.extract_jwt(txn)).await?;

        let endpoint = server.authorize_claim_and_route(link.path(), &txn)?;
        let handler = endpoint.put(key, value)?;
        handler.await
    }

    async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: Map<State>) -> TCResult<State> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;

        let txn_id = *txn.id();
        let txn = server.get_txn(Some(txn_id), self.extract_jwt(txn)).await?;

        let endpoint = server.authorize_claim_and_route(link.path(), &txn)?;
        let handler = endpoint.post(params)?;
        handler.await
    }

    async fn delete(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<()> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;

        let txn_id = *txn.id();
        let txn = server.get_txn(Some(txn_id), self.extract_jwt(txn)).await?;

        let endpoint = server.authorize_claim_and_route(link.path(), &txn)?;
        let handler = endpoint.delete(key)?;
        handler.await
    }
}

fn builder(rpc_client: Arc<Client>, name: String, key: Key) -> Builder {
    let data_dir = format!("{DATA_DIR}/{name}").parse().unwrap();
    let workspace = format!("{WORKSPACE}/{name}").parse().unwrap();

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&workspace).unwrap();

    let cache = Cache::new(CACHE_SIZE, None);
    let data_dir = cache.clone().load(data_dir).unwrap();
    let workspace = cache.clone().load(workspace).unwrap();

    Builder::load(data_dir, workspace, rpc_client)
        .detect_address()
        .with_keys([key])
        .set_secure(false)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // create an RPC client
    let client = Arc::new(Client::default());

    // generate a shared symmetric encryption key
    let key = Aes256GcmSiv::generate_key(&mut OsRng);

    // start the first server
    let mut server1 = builder(client.clone(), "one".to_string(), key)
        .set_port(DEFAULT_PORT)
        .start()
        .await;

    server1.make_discoverable().await?;

    let host1 = server1.host().clone();
    client.add(host1.clone(), server1.ready());

    // check that it's working
    let link = Link::new(host1.clone(), TCPathBuf::from(HYPOTHETICAL));

    let hello_world = Scalar::Value("Hello, World!".to_string().into());

    let op_def = [(Id::from(label("_return")), hello_world.clone())]
        .into_iter()
        .collect();

    let params = [(
        label("op").into(),
        State::Scalar(OpDef::Post(op_def).into()),
    )]
    .into_iter()
    .collect();

    let txn = client.get_txn(&host1)?;
    let response = client.post(&txn, link.into(), params).await?;

    assert_eq!(Scalar::try_from(response).unwrap(), hello_world);

    // try creating a cluster directory
    let dir_name: Id = "test".parse().unwrap();
    let link = Link::new(host1.clone(), [label("class").into()].into());

    let txn = client.get_txn(&host1)?;
    client
        .put(
            &txn,
            link.into(),
            dir_name.into(),
            Map::<State>::new().into(),
        )
        .await?;

    // start a second server and replicate the state of the first
    let mut server2 = builder(client.clone(), "two".to_string(), key)
        .start()
        .await;

    server2.discover().await?;
    assert!(!server2.peers().is_empty());

    assert!(server2.replicate_and_join(Protocol::HTTP).await);
    server2.make_discoverable().await?;

    let host2 = server2.host().clone();
    client.add(host2.clone(), server2.ready());

    Ok(())
}
