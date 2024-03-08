use async_trait::async_trait;
use log::trace;
use std::sync::Arc;

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{Host, ToUrl, Value};
use tcgeneric::Map;

use crate::kernel::Kernel;
use crate::{Actor, State, Txn};

pub trait Egress: Send + Sync {
    fn is_authorized(&self, link: &ToUrl<'_>, write: bool) -> bool;
}

#[async_trait]
pub trait RPCClient: Send + Sync {
    fn extract_jwt(&self, txn: &Txn) -> Option<String> {
        txn.token().map(|token| token.jwt().to_string())
    }

    async fn fetch(&self, txn_id: TxnId, link: ToUrl<'_>, actor_id: Value) -> TCResult<Actor>;

    async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State>;

    async fn put(&self, txn: &Txn, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()>;

    async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: Map<State>) -> TCResult<State>;

    async fn delete(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<()>;
}

#[derive(Clone)]
struct ClientInner {
    port: u16,
    kernel: Arc<Kernel>,
    client: Arc<dyn RPCClient>,
}

impl ClientInner {
    fn new(port: u16, kernel: Arc<Kernel>, client: Arc<dyn RPCClient>) -> Self {
        Self {
            port,
            kernel,
            client,
        }
    }

    #[inline]
    fn is_loopback(&self, link: &ToUrl) -> bool {
        // todo: check if the port matches the default port for the protocol (e.g. 80 for HTTP)
        link.host()
            .map(|host| host.is_localhost() && host.port() == Some(self.port))
            .unwrap_or(true)
    }
}

#[async_trait]
impl RPCClient for ClientInner {
    async fn fetch(&self, txn_id: TxnId, link: ToUrl<'_>, actor_id: Value) -> TCResult<Actor> {
        trace!("fetch actor {actor_id:?} at {link}");

        if self.is_loopback(&link) {
            let public_key = self
                .kernel
                .public_key(txn_id, link.path())
                .map_err(rjwt::Error::fetch)?;

            Ok(Actor::with_public_key(actor_id.clone(), public_key))
        } else {
            self.client.fetch(txn_id, link, actor_id).await
        }
    }

    async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State> {
        if self.is_loopback(&link) {
            let endpoint = self.kernel.route(link.path(), txn)?;
            let handler = endpoint.get(key)?;
            handler.await
        } else {
            self.client.get(txn, link, key).await
        }
    }

    async fn put(&self, txn: &Txn, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()> {
        if self.is_loopback(&link) {
            let endpoint = self.kernel.route(link.path(), txn)?;
            let handler = endpoint.put(key, value)?;
            handler.await
        } else {
            self.client.put(txn, link, key, value).await
        }
    }

    async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: Map<State>) -> TCResult<State> {
        if self.is_loopback(&link) {
            let endpoint = self.kernel.route(link.path(), txn)?;
            let handler = endpoint.post(params)?;
            handler.await
        } else {
            self.client.post(txn, link, params).await
        }
    }

    async fn delete(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<()> {
        if self.is_loopback(&link) {
            let endpoint = self.kernel.route(link.path(), txn)?;
            let handler = endpoint.delete(key)?;
            handler.await
        } else {
            self.client.delete(txn, link, key).await
        }
    }
}

// use an additional struct to provide indirection when calling RPCClient methods
// in order to minimize build time
#[derive(Clone)]
pub(crate) struct Client {
    host: Host,
    client: Arc<dyn RPCClient>,
    egress: Option<Arc<dyn Egress>>,
}

impl Client {
    pub fn new(host: Host, kernel: Arc<Kernel>, client: Arc<dyn RPCClient>) -> Self {
        let client = ClientInner::new(host.port().expect("port"), kernel, client);

        Self {
            host,
            client: Arc::new(client),
            egress: None,
        }
    }

    pub fn with_egress(self, egress: Arc<dyn Egress>) -> Self {
        Self {
            host: self.host,
            client: self.client,
            egress: Some(egress),
        }
    }

    pub fn host(&self) -> &Host {
        &self.host
    }

    #[inline]
    fn authorize(&self, link: &ToUrl<'_>, write: bool) -> TCResult<()> {
        let egress = self
            .egress
            .as_ref()
            .ok_or_else(|| unauthorized!("egress (attempted RPC to {link})"))?;

        if egress.is_authorized(&link, write) {
            Ok(())
        } else {
            Err(unauthorized!("egress to {link}"))
        }
    }
}

#[async_trait]
impl RPCClient for Client {
    async fn fetch(&self, txn_id: TxnId, link: ToUrl<'_>, actor_id: Value) -> TCResult<Actor> {
        self.client.fetch(txn_id, link, actor_id).await
    }

    async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State> {
        self.authorize(&link, false)?;
        self.client.get(txn, link, key).await
    }

    async fn put(&self, txn: &Txn, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()> {
        self.authorize(&link, true)?;
        self.client.put(txn, link, key, value).await
    }

    async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: Map<State>) -> TCResult<State> {
        self.authorize(&link, true)?;
        self.client.post(txn, link, params).await
    }

    async fn delete(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<()> {
        self.authorize(&link, true)?;
        self.client.delete(txn, link, key).await
    }
}
