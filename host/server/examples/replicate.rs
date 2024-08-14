use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::Cache;
use futures::FutureExt;
use log::info;
use tokio::sync::RwLock;

use tc_error::*;
use tc_scalar::Scalar;
use tc_server::aes256::{Aes256GcmSiv, Key, KeyInit, OsRng};
use tc_server::*;
use tc_state::object::InstanceClass;
use tc_transact::{Transaction, TxnId};
use tc_value::{Host, Link, Protocol, ToUrl, Value, Version as VersionNumber};
use tcgeneric::{label, path_label, Id, Label, Map, PathLabel, TCPath, TCPathBuf};

const CACHE_SIZE: usize = 1_000_000;
const DATA_DIR: &'static str = "/tmp/tc/example_server/data";
const WORKSPACE: &'static str = "/tmp/tc/example_server/workspace";
const HYPOTHETICAL: PathLabel = path_label(&["transact", "hypothetical"]);

const CLASS: Label = label("class");
const TEST: Label = label("test");

#[derive(Default)]
struct Client {
    servers: RwLock<HashMap<Host, Server>>,
}

impl Client {
    fn add(&self, host: Host, server: Server) {
        let mut servers = self.servers.try_write().unwrap();
        assert!(
            servers.insert(host.clone(), server).is_none(),
            "{host} is already known"
        );
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

        host.port().ok_or_else(|| {
            bad_request!("RPC to {} is missing a port", TCPath::from(link.path()))
        })?;

        servers
            .get(host)
            .ok_or_else(|| not_found!("server at {host}"))
    }
}

#[async_trait]
impl RPCClient for Client {
    async fn fetch(&self, txn_id: TxnId, link: ToUrl<'_>, actor_id: Value) -> TCResult<Actor> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;
        let txn = server.get_txn(txn_id, None).await?;

        let public_key = self
            .get(&txn, link, actor_id.clone())
            .map(|result| {
                result
                    .and_then(Value::try_from)
                    .and_then(Arc::<[u8]>::try_from)
            })
            .await?;

        VerifyingKey::try_from(&*public_key)
            .map(|public_key| Actor::with_public_key(actor_id, public_key))
            .map_err(|cause| bad_request!("invalid public key: {cause}"))
    }

    async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;

        let txn_id = *txn.id();
        let txn = server.get_txn(txn_id, self.extract_jwt(txn)).await?;

        let endpoint = server.authorize_claim_and_route(link.path(), &txn)?;
        let handler = endpoint.get(key)?;
        handler.await
    }

    async fn put(&self, txn: &Txn, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;

        let txn_id = *txn.id();
        let txn = server.get_txn(txn_id, self.extract_jwt(txn)).await?;

        let endpoint = server.authorize_claim_and_route(link.path(), &txn)?;
        let handler = endpoint.put(key, value)?;
        handler.await
    }

    async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: Map<State>) -> TCResult<State> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;

        let txn_id = *txn.id();
        let txn = server.get_txn(txn_id, self.extract_jwt(txn)).await?;

        let endpoint = server.authorize_claim_and_route(link.path(), &txn)?;
        let handler = endpoint.post(params)?;
        handler.await
    }

    async fn delete(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<()> {
        let servers = self.servers.read().await;
        let server = Self::get_server_for(&*servers, &link)?;

        let txn_id = *txn.id();
        let txn = server.get_txn(txn_id, self.extract_jwt(txn)).await?;

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

    let data_dir = DATA_DIR.parse::<PathBuf>().unwrap();
    if data_dir.exists() {
        std::fs::remove_dir_all(&data_dir).unwrap();
    }

    let workspace = WORKSPACE.parse::<PathBuf>().unwrap();
    if workspace.exists() {
        std::fs::remove_dir_all(&workspace).unwrap();
    }

    // create an RPC client
    let client = Arc::new(Client::default());

    // generate a shared symmetric encryption key
    let key = Aes256GcmSiv::generate_key(&mut OsRng);

    // start the first server
    let server1 = builder(client.clone(), "one".to_string(), key)
        .set_port(DEFAULT_PORT)
        .build()
        .await;

    let host1 = server1.address().clone();
    client.add(host1.clone(), server1);

    let mut broadcast1 = Broadcast::new();
    broadcast1.make_discoverable(&host1).await?;

    // check that it's working
    let link = Link::new(host1.clone(), TCPathBuf::from(HYPOTHETICAL));

    let hello_world = Scalar::Value("Hello, World!".to_string().into());

    let op_def = [(Id::from(label("_return")), hello_world.clone())]
        .into_iter()
        .map(|(id, state)| State::Tuple(vec![id.into(), state.into()].into()))
        .collect();

    let params: Map<State> = [(label("op").into(), State::Tuple(op_def))]
        .into_iter()
        .collect();

    let txn = client.get_txn(&host1)?;
    let response = client.post(&txn, link.into(), params).await?;

    assert_eq!(Scalar::try_from(response).unwrap(), hello_world);

    // try creating a cluster directory entry
    let link = Link::new(host1.clone(), [CLASS.into()].into());

    let txn = client.get_txn(&host1)?;

    client
        .put(&txn, link.into(), Value::Id(TEST.into()), false.into())
        .await?;

    // make sure the new dir entry is present and committed
    let txn = client.get_txn(&host1)?;
    let link = Link::new(host1.clone(), [CLASS.into(), TEST.into()].into());
    client.get(&txn, link.into(), Value::default()).await?;

    // start a second server and replicate the state of the first
    let server2 = builder(client.clone(), "two".to_string(), key)
        .set_port(DEFAULT_PORT + 1)
        .build()
        .await;

    let host2 = server2.address().clone();

    let mut broadcast2 = Broadcast::new();
    broadcast2.discover().await?;

    let peers = broadcast2.peers(Protocol::default());
    assert!(!peers.is_empty());

    let replicator = Replicator::from(&server2).with_peers(peers);
    client.add(host2.clone(), server2);

    assert!(replicator.replicate_and_join().await);

    broadcast2.make_discoverable(&host2).await?;

    // make sure the new dir entry is present and committed on both hosts
    let txn = client.get_txn(&host1)?;
    let link = Link::new(host1.clone(), [CLASS.into(), TEST.into()].into());
    client.get(&txn, link.into(), Value::default()).await?;

    let txn = client.get_txn(&host2)?;
    let link = Link::new(host2.clone(), [CLASS.into(), TEST.into()].into());
    client.get(&txn, link.into(), Value::default()).await?;

    // create a new directory item and make sure it's correctly replicated
    let version_number = VersionNumber::default();
    let classname: Id = label("classname").into();
    let class = InstanceClass::new(Map::default());

    let link = Link::new(host1.clone(), [CLASS.into(), TEST.into()].into());
    let txn = client.get_txn(&host1)?;
    client
        .put(
            &txn,
            link.into(),
            version_number.into(),
            State::Map([(classname, State::from(class))].into_iter().collect()),
        )
        .await?;

    info!("replication test succeeded");

    Ok(())
}
