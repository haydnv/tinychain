use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use serde::de::DeserializeOwned;

use error::*;
use futures::future::{try_join_all, Future, TryFutureExt};
use generic::NetworkTime;

use crate::http;
use crate::kernel::Kernel;
use crate::scalar::{Link, LinkHost, LinkProtocol, Value};
use crate::state::State;
use crate::txn::*;

pub struct Config {
    pub addr: IpAddr,
    pub http_port: u16,
    pub request_ttl: Duration,
}

#[async_trait]
pub trait Client {
    async fn fetch<T: DeserializeOwned>(
        &self,
        txn_id: &TxnId,
        link: &Link,
        key: &Value,
    ) -> TCResult<T>;

    async fn get(&self, txn: Txn, link: Link, key: Value, auth: Option<String>) -> TCResult<State>;

    async fn put(
        &self,
        txn_id: Txn,
        link: Link,
        key: Value,
        value: State,
        auth: Option<String>,
    ) -> TCResult<()>;

    async fn post(
        &self,
        txn: Txn,
        link: Link,
        params: State,
        auth: Option<String>,
    ) -> TCResult<State>;

    async fn delete(
        &self,
        txn_id: TxnId,
        link: Link,
        key: Value,
        auth: Option<String>,
    ) -> TCResult<()>;
}

#[async_trait]
pub trait Server {
    type Error: std::error::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error>;
}

pub struct Gateway {
    config: Config,
    kernel: Kernel,
    txn_server: TxnServer,
    root: LinkHost,
    client: http::Client,
    actor: Actor,
}

impl Gateway {
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

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

    pub fn root(&self) -> &LinkHost {
        &self.root
    }

    pub async fn new_txn(self: &Arc<Self>, txn_id: TxnId, token: Option<String>) -> TCResult<Txn> {
        let token = if let Some(token) = token {
            use rjwt::Resolve;
            Resolver::new(self, &txn_id)
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

    pub async fn fetch<T: DeserializeOwned>(
        &self,
        txn_id: &TxnId,
        subject: &Link,
        key: &Value,
    ) -> TCResult<T> {
        self.client.fetch(txn_id, subject, key).await
    }

    pub async fn get(&self, txn: &Txn, subject: Link, key: Value) -> TCResult<State> {
        match subject.host() {
            None if subject.path().is_empty() => {
                let public_key = Bytes::from(self.actor.public_key().as_bytes().to_vec());
                Ok(State::from(Value::from(public_key)))
            }
            None => self.kernel.get(txn, subject.path(), key).await,
            Some(host) if host == self.root() => self.kernel.get(txn, subject.path(), key).await,
            _ => {
                let auth = None; // TODO
                self.client.get(txn.clone(), subject, key, auth).await
            }
        }
    }

    pub async fn put(&self, txn: &Txn, subject: Link, key: Value, value: State) -> TCResult<()> {
        match subject.host() {
            None => self.kernel.put(txn, subject.path(), key, value).await,
            Some(host) if host == self.root() => {
                self.kernel.put(txn, subject.path(), key, value).await
            }
            _ => {
                let auth = None; // TODO
                self.client
                    .put(txn.clone(), subject, key, value, auth)
                    .await
            }
        }
    }

    pub async fn post(&self, txn: &Txn, subject: Link, params: State) -> TCResult<State> {
        match subject.host() {
            None => self.kernel.post(txn, subject.path(), params).await,
            Some(host) if host == self.root() => {
                self.kernel.post(txn, subject.path(), params).await
            }
            _ => {
                let auth = None; // TODO
                self.client.post(txn.clone(), subject, params, auth).await
            }
        }
    }

    pub fn listen(
        self: Arc<Self>,
    ) -> Pin<Box<impl Future<Output = Result<(), Box<dyn std::error::Error>>>>> {
        let servers = vec![self.http_listen()];

        Box::pin(try_join_all(servers).map_ok(|_| ()))
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
