//! [`Gateway`] handles network traffic.

use std::net::IpAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{Future, TryFutureExt};
use log::debug;

use tc_error::*;
use tc_value::{Link, LinkHost, LinkProtocol, Value};
use tcgeneric::{NetworkTime, TCBoxTryFuture, TCPathBuf};

use crate::http;
use crate::kernel::{Dispatch, Kernel};
use crate::state::State;
use crate::txn::*;

type Error = Box<dyn std::error::Error + Send + Sync>;

/// Configuration for [`Gateway`].
#[derive(Clone)]
pub struct Config {
    pub addr: IpAddr,
    pub http_port: u16,
    pub request_ttl: Duration,
}

impl Config {
    /// Construct the [`LinkHost`] of a [`Gateway`]
    pub fn host(&self) -> LinkHost {
        (self.addr, self.http_port).into()
    }
}

/// A client used by [`Gateway`]
#[async_trait]
pub trait Client {
    /// Read a simple value.
    async fn fetch<T: destream::FromStream<Context = ()>>(
        &self,
        txn_id: &TxnId,
        link: &Link,
        key: &Value,
    ) -> TCResult<T>;

    /// Read a [`State`].
    async fn get(&self, txn: Txn, link: Link, key: Value) -> TCResult<State>;

    /// Set `key` = `value` within the state referred to by `link`.
    async fn put(&self, txn: Txn, link: Link, key: Value, value: State) -> TCResult<()>;

    /// Execute a remote POST op.
    async fn post(&self, txn: Txn, link: Link, params: State) -> TCResult<State>;

    /// Delete `key` from the state referred to by `link`.
    async fn delete(&self, txn: &Txn, link: Link, key: Value) -> TCResult<()>;
}

/// A server used by [`Gateway`].
#[async_trait]
pub trait Server {
    type Error: std::error::Error;

    /// Handle incoming requests.
    async fn listen(self, port: u16) -> Result<(), Self::Error>;
}

/// Responsible for handling inbound and outbound traffic over the network.
pub struct Gateway {
    config: Config,
    kernel: Kernel,
    txn_server: TxnServer,
    host: LinkHost,
    client: http::Client,
    actor: Actor,
}

impl Gateway {
    /// Return the current timestamp.
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    /// Initialize a new `Gateway`
    pub fn new(config: Config, kernel: Kernel, txn_server: TxnServer) -> Self {
        let root = LinkHost::from((
            LinkProtocol::HTTP,
            config.addr.clone(),
            Some(config.http_port),
        ));

        Self {
            config,
            kernel,
            txn_server,
            host: root,
            client: http::Client::new(),
            actor: Actor::new(Link::default().into()),
        }
    }

    /// Return the configured maximum request time-to-live (timeout duration).
    pub fn request_ttl(&self) -> Duration {
        self.config.request_ttl
    }

    /// Return the network address of this `Gateway`
    pub fn host(&self) -> &LinkHost {
        &self.host
    }

    /// Return a [`Link`] to the given path at this host.
    pub fn link(&self, path: TCPathBuf) -> Link {
        Link::from((self.host.clone(), path))
    }

    /// Return a new, signed auth token with no claims.
    pub fn new_token(&self, txn_id: &TxnId) -> TCResult<(String, Claims)> {
        let token = Token::new(
            self.host.clone().into(),
            txn_id.time().into(),
            self.config.request_ttl,
            self.actor.id().clone(),
            vec![],
        );

        let signed = self.actor.sign_token(&token).map_err(TCError::internal)?;
        let claims = token.claims();
        Ok((signed, claims))
    }

    /// Authorize a transaction to execute on this host.
    pub async fn new_txn(self: &Arc<Self>, txn_id: TxnId, token: Option<String>) -> TCResult<Txn> {
        let token = if let Some(token) = token {
            use rjwt::Resolve;
            Resolver::new(self, &self.host().clone().into(), &txn_id)
                .consume_and_sign(&self.actor, vec![], token, txn_id.time().into())
                .map_err(TCError::unauthorized)
                .await?
        } else {
            self.new_token(&txn_id)?
        };

        self.txn_server.new_txn(self.clone(), txn_id, token).await
    }

    /// Read a simple value.
    // TODO: accept a Borrow<Link>
    pub async fn fetch<T: destream::FromStream<Context = ()>>(
        &self,
        txn_id: &TxnId,
        link: &Link,
        key: &Value,
    ) -> TCResult<T> {
        self.client.fetch(txn_id, link, key).await
    }

    /// Read the [`State`] at `link` with the given `key`.
    // TODO: accept a Borrow<Link>
    pub async fn get(&self, txn: &Txn, link: Link, key: Value) -> TCResult<State> {
        debug!("GET {}: {}", link, key);
        match link.host() {
            None if link.path().is_empty() && key.is_none() => {
                let public_key = Bytes::from(self.actor.public_key().as_bytes().to_vec());
                Ok(State::from(Value::from(public_key)))
            }
            None => self.kernel.get(txn, link.path(), key).await,
            Some(host) if host == self.host() => self.kernel.get(txn, link.path(), key).await,
            _ => self.client.get(txn.clone(), link, key).await,
        }
    }

    /// Update the [`State`] with the given `key` at `link` to `value`.
    // TODO: accept a Borrow<Link>
    pub fn put<'a>(
        &'a self,
        txn: &'a Txn,
        link: Link,
        key: Value,
        value: State,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            debug!("PUT {}: {} <- {}", link, key, value);

            match link.host() {
                None => self.kernel.put(txn, link.path(), key, value).await,
                Some(host) if host == self.host() => {
                    self.kernel.put(txn, link.path(), key, value).await
                }
                _ => self.client.put(txn.clone(), link, key, value).await,
            }
        })
    }

    /// Execute the POST op at `link` with the `params`
    // TODO: accept a Borrow<Link>
    pub async fn post(&self, txn: &Txn, link: Link, params: State) -> TCResult<State> {
        debug!("POST to {} with params {}", link, params);

        match link.host() {
            None => self.kernel.post(txn, link.path(), params).await,
            Some(host) if host == self.host() => self.kernel.post(txn, link.path(), params).await,
            _ => self.client.post(txn.clone(), link, params).await,
        }
    }

    /// Delete the [`State`] at `link` with the given `key`.
    // TODO: accept a Borrow<Link>
    pub fn delete<'a>(&'a self, txn: &'a Txn, link: Link, key: Value) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            debug!("DELETE {}: {}", link, key);
            match link.host() {
                None => self.kernel.delete(txn, link.path(), key).await,
                Some(host) if host == self.host() => {
                    self.kernel.delete(txn, link.path(), key).await
                }
                _ => self.client.delete(txn, link, key).await,
            }
        })
    }

    /// Start this `Gateway`'s server
    pub fn listen(self: Arc<Self>) -> Pin<Box<impl Future<Output = Result<(), Error>> + 'static>> {
        Box::pin(self.http_listen())
    }

    fn http_listen(
        self: Arc<Self>,
    ) -> std::pin::Pin<Box<impl futures::Future<Output = Result<(), Error>>>> {
        let port = self.config.http_port;
        let server = crate::http::HTTPServer::new(self);
        let listener = server.listen(port).map_err(|e| {
            let e: Error = Box::new(e);
            e
        });

        Box::pin(listener)
    }
}
