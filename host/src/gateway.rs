//! [`Gateway`] handles network traffic.

use std::fmt;
use std::net::IpAddr;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{Future, TryFutureExt};
use log::debug;
use tokio::time::Duration;
use url::Url;

use tc_error::*;
use tc_value::{Host, Link, Protocol, ToUrl, Value};
use tcgeneric::{NetworkTime, PathSegment, TCBoxTryFuture, TCPath, TCPathBuf};

use crate::kernel::{Dispatch, Kernel};
use crate::state::State;
use crate::txn::*;
use crate::{http, TokioError};

const INTERVAL: Duration = Duration::from_millis(100);

/// Configuration for [`Gateway`].
#[derive(Clone)]
pub struct Config {
    pub addr: IpAddr,
    pub http_port: u16,
    pub request_ttl: Duration,
}

impl Config {
    /// Construct the [`Host``] of a [`Gateway`]
    pub fn host(&self) -> Host {
        (Protocol::HTTP, self.addr.clone().into(), self.http_port).into()
    }
}

/// A client used by [`Gateway`]
#[async_trait]
pub trait Client {
    /// Read a simple value.
    async fn fetch<T>(&self, txn_id: &TxnId, link: ToUrl<'_>, key: &Value) -> TCResult<T>
    where
        T: destream::FromStream<Context = ()>;

    /// Read a [`State`].
    async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State>;

    /// Set `key` = `value` within the state referred to by `link`.
    async fn put(&self, txn: &Txn, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()>;

    /// Execute a remote POST op.
    async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: State) -> TCResult<State>;

    /// Delete `key` from the state referred to by `link`.
    async fn delete(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<()>;
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
    actor: Actor,
    config: Config,
    client: http::Client,
    host: Host,
    kernel: Kernel,
    txn_server: TxnServer,
}

impl Gateway {
    /// Return the current timestamp.
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    /// Initialize a new `Gateway`
    pub fn new(config: Config, kernel: Kernel, txn_server: TxnServer) -> Arc<Self> {
        let root = Host::from((Protocol::HTTP, config.addr.clone().into(), config.http_port));

        let gateway = Arc::new(Self {
            config,
            kernel,
            txn_server,
            host: root,
            client: http::Client::new(),
            actor: Actor::new(Link::default().into()),
        });

        spawn_cleanup_thread(gateway.clone());

        gateway
    }

    /// Return the configured maximum request time-to-live (timeout duration).
    pub fn request_ttl(&self) -> Duration {
        self.config.request_ttl
    }

    /// Return the network address of this `Gateway`
    pub fn host(&self) -> &Host {
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

        let signed = self
            .actor
            .sign_token(&token)
            .map_err(|cause| unexpected!("signing error").consume(cause))?;

        let claims = token.claims();

        Ok((signed, claims))
    }

    /// Authorize a transaction to execute on this host.
    pub async fn new_txn(self: &Arc<Self>, txn_id: TxnId, token: Option<String>) -> TCResult<Txn> {
        let token = if let Some(token) = token {
            use rjwt::Resolve;
            Resolver::new(self, &self.host().clone().into(), &txn_id)
                .consume_and_sign(&self.actor, vec![], token, txn_id.time().into())
                .map_err(|cause| unauthorized!("credential error").consume(cause))
                .await?
        } else {
            self.new_token(&txn_id)?
        };

        self.txn_server.new_txn(self.clone(), txn_id, token).await
    }

    /// Read a simple value.
    pub async fn fetch<T>(&self, txn_id: &TxnId, link: ToUrl<'_>, key: &Value) -> TCResult<T>
    where
        T: destream::FromStream<Context = ()>,
    {
        self.client.fetch(txn_id, link, key).await
    }

    /// Read the [`State`] at `link` with the given `key`.
    pub async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State> {
        debug!("GET {}: {}", link, key);

        match link.host() {
            None if link.path().is_empty() && key.is_none() => {
                let public_key = Bytes::from(self.actor.public_key().as_bytes().to_vec());
                Ok(State::from(Value::from(public_key)))
            }
            None => self.kernel.get(txn, link.path(), key).await,
            Some(host) if host == self.host() => self.kernel.get(txn, link.path(), key).await,
            _ => self.client.get(txn, link, key).await,
        }
    }

    /// Update the [`State`] with the given `key` at `link` to `value`.
    pub fn put<'a>(
        &'a self,
        txn: &'a Txn,
        link: ToUrl<'a>,
        key: Value,
        value: State,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            debug!("PUT {}: {} <- {:?}", link, key, value);

            match link.host() {
                None => self.kernel.put(txn, link.path(), key, value).await,
                Some(host) if host == self.host() => {
                    self.kernel.put(txn, link.path(), key, value).await
                }
                _ => self.client.put(txn, link, key, value).await,
            }
        })
    }

    /// Execute the POST op at `link` with the `params`
    pub async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: State) -> TCResult<State> {
        debug!("POST to {} with params {:?}", link, params);

        match link.host() {
            None => self.kernel.post(txn, link.path(), params).await,
            Some(host) if host == self.host() => self.kernel.post(txn, link.path(), params).await,
            _ => self.client.post(txn, link, params).await,
        }
    }

    /// Delete the [`State`] at `link` with the given `key`.
    pub fn delete<'a>(
        &'a self,
        txn: &'a Txn,
        link: ToUrl<'a>,
        key: Value,
    ) -> TCBoxTryFuture<'a, ()> {
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
    pub fn listen(
        self: Arc<Self>,
    ) -> Pin<Box<impl Future<Output = Result<(), TokioError>> + 'static>> {
        let port = self.config.http_port;
        let server = crate::http::HTTPServer::new(self);
        let listener = server.listen(port).map_err(|e| {
            let e: TokioError = Box::new(e);
            e
        });

        Box::pin(listener)
    }

    pub(crate) async fn finalize(&self, txn_id: TxnId) {
        self.kernel.finalize(txn_id).await;
    }
}

fn spawn_cleanup_thread(gateway: Arc<Gateway>) {
    let mut interval = tokio::time::interval(INTERVAL);

    tokio::spawn(async move {
        loop {
            interval.tick().await;

            gateway
                .txn_server
                .finalize_expired(&gateway, Gateway::time())
                .await;
        }
    });
}
