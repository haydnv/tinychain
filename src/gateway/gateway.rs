use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;

use futures::stream::Stream;

use crate::auth::{Auth, Token};
use crate::block::dir::Dir;
use crate::class::{ResponseStream, State, TCResult};
use crate::error;
use crate::kernel;
use crate::transaction::{Txn, TxnId};
use crate::value::link::{Link, LinkHost};
use crate::value::{Value, ValueId};

use super::{Hosted, NetworkTime, Server};

pub struct Gateway {
    peers: Vec<LinkHost>,
    hosted: Hosted,
    workspace: Arc<Dir>,
}

impl Gateway {
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    pub fn new(peers: Vec<LinkHost>, hosted: Hosted, workspace: Arc<Dir>) -> Gateway {
        Gateway {
            peers,
            hosted,
            workspace,
        }
    }

    pub async fn authenticate(&self, _token: &str) -> TCResult<Token> {
        Err(error::not_implemented("Gateway::authenticate"))
    }

    pub async fn transaction(self: &Arc<Self>) -> TCResult<Arc<Txn>> {
        Txn::new(self.clone(), self.workspace.clone()).await
    }

    pub async fn http_listen(
        self: Arc<Self>,
        address: IpAddr,
        port: u16,
    ) -> Result<(), hyper::Error> {
        let server = Arc::new(super::HttpServer::new((address, port).into()));
        server.listen(self).await
    }

    pub async fn get(
        &self,
        subject: &Link,
        selector: Value,
        _auth: &Auth,
        txn: Option<Arc<Txn>>,
    ) -> TCResult<State> {
        if let Some(_) = subject.host() {
            Err(error::not_implemented("Gateway::get over the network"))
        } else {
            let path = subject.path();
            if path[0] == "sbin" {
                kernel::get(path, selector, txn).await
            } else {
                Err(error::not_implemented("Gateway::get over the network"))
            }
        }
    }

    pub async fn put(
        &self,
        _subject: &Link,
        _selector: Value,
        _state: State,
        _auth: &Auth,
        _txn_id: Option<TxnId>,
    ) -> TCResult<State> {
        Err(error::not_implemented("Gateway::put"))
    }

    pub async fn post<S: Stream<Item = (ValueId, Value)> + Unpin>(
        self: Arc<Self>,
        subject: &Link,
        data: S,
        capture: HashSet<ValueId>,
        auth: &Auth,
        txn_id: Option<TxnId>,
    ) -> TCResult<ResponseStream> {
        println!("Gateway::post {}", subject);

        if subject.host().is_none() {
            let workspace = self.workspace.clone();
            let txn = match txn_id {
                None => Txn::new(self, workspace).await?,
                Some(_txn_id) => return Err(error::not_implemented("Gateway::Post with txn_id")),
            };

            let path = subject.path();
            if path[0] == "sbin" {
                kernel::post(txn, &path.slice_from(1), data, capture, auth).await
            } else {
                Err(error::not_found(path))
            }
        } else {
            Err(error::not_implemented("Gateway::post over the network"))
        }
    }
}
