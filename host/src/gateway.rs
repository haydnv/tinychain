//! [`Gateway`] handles network traffic.

use std::net::IpAddr;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{Future, TryFutureExt};
use log::debug;
use tokio::time::{Duration, MissedTickBehavior};

use tc_error::*;
use tc_fs::{Actor, Gateway as GatewayInstance, SignedToken, Token, TxnServer};
use tc_state::State;
use tc_transact::TxnId;
use tc_value::{Host, Link, Protocol, ToUrl, Value};
use tcgeneric::{NetworkTime, TCBoxFuture, TCBoxTryFuture, TCPathBuf};

use crate::kernel::{Dispatch, Kernel};
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

struct Inner {
    actor: Actor,
    config: Config,
    client: http::Client,
    host: Host,
    kernel: Kernel,
    txn_server: TxnServer,
}

/// Responsible for handling inbound and outbound traffic over the network.
#[derive(Clone)]
pub struct Gateway {
    inner: Arc<Inner>,
}

impl Gateway {
    /// Return the current timestamp.
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    /// Initialize a new `Gateway`
    pub fn new(config: Config, kernel: Kernel, txn_server: TxnServer) -> Self {
        let root = Host::from((Protocol::HTTP, config.addr.clone().into(), config.http_port));

        let inner = Arc::new(Inner {
            config,
            kernel,
            txn_server,
            host: root,
            client: http::Client::new(),
            actor: Actor::new(Link::default().into()),
        });

        let gateway = Gateway { inner };

        spawn_cleanup_thread(gateway.clone());

        gateway
    }

    /// Return the configured maximum request time-to-live (timeout duration).
    pub fn request_ttl(&self) -> Duration {
        self.inner.config.request_ttl
    }

    /// Return a new, signed auth token with no claims.
    pub fn new_token(&self, txn_id: &TxnId) -> TCResult<SignedToken> {
        let token = Token::new(
            self.inner.host.clone().into(),
            txn_id.time().into(),
            self.inner.config.request_ttl,
            self.inner.actor.id().clone(),
            vec![],
        );

        let signed = self
            .inner
            .actor
            .sign_token(token)
            .map_err(|cause| internal!("signing error").consume(cause))?;

        Ok(signed)
    }

    /// Authorize a transaction to execute on this host.
    pub fn new_txn(self, txn_id: TxnId, token: Option<SignedToken>) -> TCResult<Txn> {
        let token = if let Some(token) = token {
            self.inner.actor.consume_and_sign(
                token,
                self.host().clone().into(),
                vec![],
                txn_id.time().into(),
            )?
        } else {
            self.new_token(&txn_id)?
        };

        let txn_server = self.inner.txn_server.clone();

        txn_server.new_txn(Arc::new(self), txn_id, token)
    }

    /// Start this `Gateway`'s server
    pub fn listen(self) -> Pin<Box<impl Future<Output = Result<(), TokioError>> + 'static>> {
        let port = self.inner.config.http_port;
        let server = http::HTTPServer::new(self);
        let listener = server.listen(port).map_err(|e| {
            let e: TokioError = Box::new(e);
            e
        });

        Box::pin(listener)
    }
}

impl GatewayInstance for Gateway {
    type State = State;

    fn host(&self) -> &Host {
        &self.inner.host
    }

    fn link(&self, path: TCPathBuf) -> Link {
        Link::from((self.inner.host.clone(), path))
    }

    fn fetch<'a>(
        &'a self,
        txn_id: &'a TxnId,
        link: ToUrl<'a>,
        key: &'a Value,
    ) -> TCBoxTryFuture<Value> {
        Box::pin(self.inner.client.fetch(txn_id, link, key))
    }

    fn get<'a>(&'a self, txn: &'a Txn, link: ToUrl<'a>, key: Value) -> TCBoxTryFuture<'a, State> {
        debug!("GET {}: {}", link, key);

        Box::pin(async move {
            match link.host() {
                None if link.path().is_empty() && key.is_none() => {
                    let public_key = Bytes::from(self.inner.actor.public_key().as_bytes().to_vec());
                    Ok(State::from(Value::from(public_key)))
                }
                None => self.inner.kernel.get(txn, link.path(), key).await,
                Some(host) if host == self.host() => {
                    self.inner.kernel.get(txn, link.path(), key).await
                }
                _ => self.inner.client.get(txn, link, key).await,
            }
        })
    }

    fn put<'a>(
        &'a self,
        txn: &'a Txn,
        link: ToUrl<'a>,
        key: Value,
        value: State,
    ) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            debug!("PUT {}: {} <- {:?}", link, key, value);

            match link.host() {
                None => self.inner.kernel.put(txn, link.path(), key, value).await,
                Some(host) if host == self.host() => {
                    self.inner.kernel.put(txn, link.path(), key, value).await
                }
                _ => self.inner.client.put(txn, link, key, value).await,
            }
        })
    }

    fn post<'a>(
        &'a self,
        txn: &'a Txn,
        link: ToUrl<'a>,
        params: State,
    ) -> TCBoxTryFuture<'a, State> {
        debug!("POST to {} with params {:?}", link, params);

        Box::pin(async move {
            match link.host() {
                None => self.inner.kernel.post(txn, link.path(), params).await,
                Some(host) if host == self.host() => {
                    self.inner.kernel.post(txn, link.path(), params).await
                }
                _ => self.inner.client.post(txn, link, params).await,
            }
        })
    }

    fn delete<'a>(&'a self, txn: &'a Txn, link: ToUrl<'a>, key: Value) -> TCBoxTryFuture<'a, ()> {
        Box::pin(async move {
            debug!("DELETE {}: {}", link, key);

            match link.host() {
                None => self.inner.kernel.delete(txn, link.path(), key).await,
                Some(host) if host == self.host() => {
                    self.inner.kernel.delete(txn, link.path(), key).await
                }
                _ => self.inner.client.delete(txn, link, key).await,
            }
        })
    }

    fn finalize(&self, txn_id: TxnId) -> TCBoxFuture<()> {
        Box::pin(self.inner.kernel.finalize(txn_id))
    }
}

fn spawn_cleanup_thread(gateway: Gateway) {
    let mut interval = tokio::time::interval(INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

    tokio::spawn(async move {
        loop {
            interval.tick().await;

            gateway
                .inner
                .txn_server
                .finalize_expired(&gateway, Gateway::time())
                .await;
        }
    });
}
