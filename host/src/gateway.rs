use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use error::*;
use futures::future::{try_join_all, Future, TryFutureExt};
use generic::NetworkTime;

use crate::http;
use crate::kernel::Kernel;
use crate::scalar::{Link, LinkHost, LinkProtocol, Value};
use crate::state::State;
use crate::txn::*;

#[async_trait]
pub trait Client {
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

    async fn delete(&self, link: Link, key: Value, auth: Option<String>) -> TCResult<()>;
}

#[async_trait]
pub trait Server {
    type Error: std::error::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error>;
}

pub struct Gateway {
    kernel: Kernel,
    txn_server: TxnServer,
    addr: IpAddr,
    http_port: u16,
    request_ttl: Duration,
    client: http::Client,
}

impl Gateway {
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    pub async fn authenticate(&self, _token: &str) -> TCResult<Token> {
        Err(TCError::not_implemented("Gateway::authenticate"))
    }

    pub fn new(
        kernel: Kernel,
        txn_server: TxnServer,
        addr: IpAddr,
        http_port: u16,
        request_ttl: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            kernel,
            addr,
            txn_server,
            http_port,
            request_ttl,
            client: http::Client::new(),
        })
    }

    pub fn issue_token(&self) -> Token {
        <Token as TokenExt>::new(
            self.root(),
            Self::time(),
            self.request_ttl,
            Value::None,
            vec![],
        )
    }

    pub fn root(&self) -> Link {
        let host = LinkHost::from((LinkProtocol::HTTP, self.addr.clone(), Some(self.http_port)));
        host.into()
    }

    pub async fn new_txn(self: &Arc<Self>, request: Request) -> TCResult<Txn> {
        let this = self.clone();
        self.txn_server.new_txn(this, request).await
    }

    pub async fn get(&self, txn: &Txn, subject: Link, key: Value) -> TCResult<State> {
        if subject.host().is_none() {
            self.kernel.get(txn, subject.path(), key).await
        } else {
            // TODO: auth
            self.client.get(txn.clone(), subject, key, None).await
        }
    }

    pub async fn put(&self, txn: &Txn, subject: Link, key: Value, value: State) -> TCResult<()> {
        if subject.host().is_none() {
            self.kernel.put(txn, subject.path(), key, value).await
        } else {
            // TODO: auth
            self.client
                .put(txn.clone(), subject, key, value, None)
                .await
        }
    }

    pub async fn post(&self, txn: &Txn, subject: Link, params: State) -> TCResult<State> {
        if subject.host().is_none() {
            self.kernel.post(txn, subject.path(), params).await
        } else {
            // TODO: auth
            self.client.post(txn.clone(), subject, params, None).await
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
        let port = self.http_port;
        let http_addr = (self.addr, port).into();
        let server = crate::http::HTTPServer::new(self);
        let listener = server.listen(http_addr).map_err(|e| {
            let e: Box<dyn std::error::Error> = Box::new(e);
            e
        });

        Box::pin(listener)
    }
}
