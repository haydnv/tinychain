use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use auth::Token;
use error::*;
use futures::future::{try_join_all, Future, TryFutureExt};
use generic::NetworkTime;

use crate::kernel::Kernel;
use crate::scalar::{Link, LinkHost, LinkProtocol, Value};
use crate::state::State;
use crate::txn::*;

const DEFAULT_TTL: Duration = Duration::from_secs(30);

#[async_trait::async_trait]
pub trait Server {
    type Error: std::error::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error>;
}

pub struct Gateway {
    kernel: Kernel,
    txn_server: TxnServer,
    addr: IpAddr,
    config: HashMap<LinkProtocol, u16>,
}

impl Gateway {
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    pub async fn authenticate(&self, _token: &str) -> TCResult<Token> {
        Err(TCError::not_implemented("Gateway::authenticate"))
    }

    pub fn issue_token(&self) -> TCResult<Token> {
        let host = self.root().ok_or_else(|| {
            TCError::unsupported(
                "Cannot issue an auth token without a running server to authenticate it",
            )
        })?;

        Ok(Token::new(
            host,
            Self::time(),
            DEFAULT_TTL,
            Value::None,
            vec![],
        ))
    }

    pub fn new(
        kernel: Kernel,
        txn_server: TxnServer,
        addr: IpAddr,
        config: HashMap<LinkProtocol, u16>,
    ) -> Arc<Self> {
        Arc::new(Self {
            kernel,
            addr,
            config,
            txn_server,
        })
    }

    pub fn root(&self) -> Option<Link> {
        if let Some(port) = self.config.get(&LinkProtocol::HTTP) {
            let host = LinkHost::from((LinkProtocol::HTTP, self.addr.clone(), Some(*port)));
            Some(host.into())
        } else {
            None
        }
    }

    pub async fn new_txn(self: &Arc<Self>, request: Request) -> TCResult<Txn> {
        let this = self.clone();
        self.txn_server.new_txn(this, request).await
    }

    pub async fn get(&self, txn: &Txn, subject: Link, key: Value) -> TCResult<State> {
        if subject.host().is_none() {
            self.kernel.get(txn, subject.path(), key).await
        } else {
            Err(TCError::not_implemented("remote GET"))
        }
    }

    pub async fn post(&self, txn: &Txn, subject: Link, params: State) -> TCResult<State> {
        if subject.host().is_none() {
            self.kernel.post(txn, subject.path(), params).await
        } else {
            Err(TCError::not_implemented("remote GET"))
        }
    }

    pub fn listen(
        self: Arc<Self>,
    ) -> Pin<Box<impl Future<Output = Result<(), Box<dyn std::error::Error>>>>> {
        #[allow(unused_mut)]
        let mut servers = Vec::<
            Box<dyn Future<Output = Result<(), Box<dyn std::error::Error>>> + Unpin>,
        >::with_capacity(1);

        #[cfg(feature = "http")]
        {
            servers.push(Box::new(self.http_listen()));
        }

        Box::pin(try_join_all(servers).map_ok(|_| ()))
    }

    #[cfg(feature = "http")]
    fn http_listen(
        self: Arc<Self>,
    ) -> std::pin::Pin<Box<impl futures::Future<Output = Result<(), Box<dyn std::error::Error>>>>>
    {
        let port = self.config.get(&LinkProtocol::HTTP).unwrap();
        let http_addr = (self.addr, *port).into();
        let server = crate::http::HTTPServer::new(self);
        let listener = server.listen(http_addr).map_err(|e| {
            let e: Box<dyn std::error::Error> = Box::new(e);
            e
        });

        Box::pin(listener)
    }
}
