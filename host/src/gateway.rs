//! [`Gateway`] handles network traffic.

use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{Future, TryFutureExt};
use futures::try_join;
use log::debug;
use serde::de::DeserializeOwned;

use tc_error::*;
use tcgeneric::{Map, NetworkTime, TCPathBuf};

use crate::http;
use crate::kernel::Kernel;
use crate::route::Route;
use crate::scalar::{Link, LinkHost, LinkProtocol, Value};
use crate::state::State;
use crate::txn::*;

/// Configuration for [`Gateway`].
pub struct Config {
    pub addr: IpAddr,
    pub http_port: u16,
    pub request_ttl: Duration,
}

/// A client used by [`Gateway`]
#[async_trait]
pub trait Client {
    /// Read a simple value.
    async fn fetch<T: DeserializeOwned>(
        &self,
        txn_id: &TxnId,
        link: &Link,
        key: &Value,
    ) -> TCResult<T>;

    /// Read a [`State`].
    async fn get(&self, txn: Txn, link: Link, key: Value) -> TCResult<State>;

    /// Set `key` = `value` within the state referred to by `link`.
    async fn put(&self, txn_id: Txn, link: Link, key: Value, value: State) -> TCResult<()>;

    /// Execute a remote POST op.
    async fn post(&self, txn: Txn, link: Link, params: State) -> TCResult<State>;

    /// Delete `key` from the state referred to by `link`.
    async fn delete(&self, txn_id: &Txn, link: Link, key: Value) -> TCResult<()>;
}

/// A server used by [`Gateway`].
#[async_trait]
pub trait Server {
    type Error: std::error::Error;

    /// Handle incoming requests.
    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error>;
}

/// Responsible for handling inbound and outbound traffic over the network.
pub struct Gateway {
    config: Config,
    kernel: Kernel,
    txn_server: TxnServer,
    root: LinkHost,
    client: http::Client,
    actor: Actor,
}

impl Gateway {
    /// Return the current timestamp.
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    /// Initialize a new `Gateway`
    pub fn new(config: Config, kernel: Kernel, txn_server: TxnServer) -> Arc<Self> {
        let root = LinkHost::from((
            LinkProtocol::HTTP,
            config.addr.clone(),
            Some(config.http_port),
        ));

        Arc::new(Self {
            config,
            kernel,
            txn_server,
            root,
            client: http::Client::new(),
            actor: Actor::new(Link::default().into()),
        })
    }

    /// Return the network address of this `Gateway`
    pub fn root(&self) -> &LinkHost {
        &self.root
    }

    /// Return a [`Link`] to the given path at this host.
    pub fn link(&self, path: TCPathBuf) -> Link {
        Link::from((self.root.clone(), path))
    }

    /// Authorize a transaction to execute on this host.
    pub async fn new_txn(self: &Arc<Self>, txn_id: TxnId, token: Option<String>) -> TCResult<Txn> {
        let token = if let Some(token) = token {
            use rjwt::Resolve;
            Resolver::new(self, &self.root().clone().into(), &txn_id)
                .consume_and_sign(&self.actor, vec![], token, txn_id.time().into())
                .map_err(TCError::unauthorized)
                .await?
        } else {
            let token = Token::new(
                self.root.clone().into(),
                txn_id.time().into(),
                self.config.request_ttl,
                self.actor.id().clone(),
                vec![],
            );
            let signed = self.actor.sign_token(&token).map_err(TCError::internal)?;
            let claims = token.claims();
            (signed, claims)
        };

        self.txn_server.new_txn(self.clone(), txn_id, token).await
    }

    /// Read a simple value.
    pub async fn fetch<T: DeserializeOwned>(
        &self,
        txn_id: &TxnId,
        link: &Link,
        key: &Value,
    ) -> TCResult<T> {
        self.client.fetch(txn_id, link, key).await
    }

    /// Read the [`State`] with the given `key` at `link`.
    pub async fn get(&self, txn: &Txn, link: Link, key: Value) -> TCResult<State> {
        debug!("GET {}: {}", link, key);
        match link.host() {
            None if link.path().is_empty() && key.is_none() => {
                let public_key = Bytes::from(self.actor.public_key().as_bytes().to_vec());
                Ok(State::from(Value::from(public_key)))
            }
            None => self.kernel.get(txn, link.path(), key).await,
            Some(host) if host == self.root() => self.kernel.get(txn, link.path(), key).await,
            _ => self.client.get(txn.clone(), link, key).await,
        }
    }

    /// Update the [`State`] with the given `key` at `link` to `value`.
    pub fn put<'a>(
        &'a self,
        txn: &'a Txn,
        link: Link,
        key: Value,
        value: State,
    ) -> Pin<Box<dyn Future<Output = TCResult<()>> + Send + 'a>> {
        Box::pin(async move {
            debug!("PUT {}: {} <- {}", link, key, value);

            match link.host() {
                None => self.kernel.put(txn, link.path(), key, value).await,
                Some(host) if host == self.root() => {
                    self.kernel.put(txn, link.path(), key, value).await
                }
                _ => self.client.put(txn.clone(), link, key, value).await,
            }
        })
    }

    /// Execute the POST op at `subject` with the `params`
    pub async fn post(&self, txn: &Txn, link: Link, params: State) -> TCResult<State> {
        debug!("POST to {} with params {}", link, params);

        match link.host() {
            None => self.kernel.post(txn, link.path(), params).await,
            Some(host) if host == self.root() => self.kernel.post(txn, link.path(), params).await,
            _ => self.client.post(txn.clone(), link, params).await,
        }
    }

    /// Delete the [`State`] with the given `key` at `link`.
    pub async fn delete(&self, txn: &Txn, link: Link, key: Value) -> TCResult<()> {
        debug!("DELETE {}: {}", link, key);
        match link.host() {
            None => self.kernel.delete(txn, link.path(), key).await,
            Some(host) if host == self.root() => self.kernel.delete(txn, link.path(), key).await,
            _ => self.client.delete(txn, link, key).await,
        }
    }

    /// Start this `Gateway`'s server
    pub fn listen(
        self: Arc<Self>,
    ) -> Pin<Box<impl Future<Output = Result<(), Box<dyn std::error::Error>>> + 'static>> {
        Box::pin(async move {
            match try_join!(self.clone().http_listen(), self.clone().replicate()) {
                Ok(_) => Ok(()),
                Err(cause) => Err(cause),
            }
        })
    }

    async fn replicate(self: Arc<Self>) -> Result<(), Box<dyn std::error::Error>> {
        let result = async move {
            for cluster in self.kernel.hosted() {
                let gateway = self.clone();

                if cluster.link().host().is_none() {
                    continue;
                }

                let txn = gateway.new_txn(TxnId::new(Self::time()), None).await?;
                let txn = cluster.claim(&txn).await?;

                let cluster_link = cluster.link().clone();
                let self_link = txn.link(cluster_link.path().clone());
                gateway
                    .client
                    .put(
                        txn.clone(),
                        cluster_link.clone(),
                        Value::None,
                        self_link.into(),
                    )
                    .await?;

                // send a commit message
                cluster.route(&[]).unwrap().post().unwrap()(txn, Map::default()).await?;
            }

            TCResult::Ok(())
        };

        match result.await {
            Ok(()) => Result::<(), Box<dyn std::error::Error>>::Ok(()),
            Err(cause) => {
                let e: Box<dyn std::error::Error> = Box::new(cause);
                Err(e)
            }
        }
    }

    fn http_listen(
        self: Arc<Self>,
    ) -> std::pin::Pin<Box<impl futures::Future<Output = Result<(), Box<dyn std::error::Error>>>>>
    {
        let http_addr = (self.config.addr, self.config.http_port).into();
        let server = crate::http::HTTPServer::new(self);
        let listener = server.listen(http_addr).map_err(|e| {
            let e: Box<dyn std::error::Error> = Box::new(e);
            e
        });

        Box::pin(listener)
    }
}
