use std::collections::HashSet;
use std::convert::TryInto;
use std::net::IpAddr;
use std::sync::Arc;

use futures::future::TryFutureExt;
use futures::stream::Stream;

use crate::auth::{Auth, Token};
use crate::block::Dir;
use crate::class::{State, TCResult, TCStream};
use crate::error;
use crate::kernel;
use crate::transaction::{Txn, TxnId};
use crate::value::link::{Link, LinkHost};
use crate::value::{Value, ValueId};

use super::http;
use super::{Hosted, NetworkTime, Server};

pub struct Gateway {
    peers: Vec<LinkHost>,
    adapters: Vec<Link>,
    hosted: Hosted,
    workspace: Arc<Dir>,
    client: http::Client,
    request_limit: usize,
}

impl Gateway {
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    pub fn new(
        peers: Vec<LinkHost>,
        adapters: Vec<Link>,
        hosted: Hosted,
        workspace: Arc<Dir>,
        request_limit: usize,
    ) -> TCResult<Gateway> {
        let mut adapter_uris = HashSet::new();
        for adapter in &adapters {
            if adapter.host().is_none() {
                return Err(error::bad_request("An adapter requires a host", adapter));
            }

            if adapter_uris.contains(adapter.path()) {
                return Err(error::bad_request(
                    "Duplicate adapter provided",
                    adapter.path(),
                ));
            } else {
                adapter_uris.insert(adapter.path().clone());
            }
        }

        let client = http::Client::new(request_limit);

        Ok(Gateway {
            peers,
            adapters,
            hosted,
            workspace,
            client,
            request_limit,
        })
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
        let server = Arc::new(super::HttpServer::new(
            (address, port).into(),
            self.request_limit,
        ));
        server.clone().listen(self).await
    }

    pub async fn get(
        self: Arc<Self>,
        subject: &Link,
        key: Value,
        auth: Auth,
        txn: Option<Arc<Txn>>,
    ) -> TCResult<State> {
        println!("Gateway::get {}", subject);

        if subject.host().is_some() {
            Err(error::not_implemented("Gateway::get over the network"))
        } else if subject.path().len() > 1 {
            let path = subject.path();
            if path[0] == "sbin" {
                kernel::get(path, key, txn).await
            } else if path[0] == "ext" {
                for adapter in &self.adapters {
                    if path.starts_with(adapter.path()) {
                        let host = adapter.host().as_ref().unwrap();
                        let dest: Link = (host.clone(), path.clone()).into();
                        return self
                            .client
                            .get(dest, &key, auth, txn)
                            .map_ok(State::Value)
                            .await;
                    }
                }

                Err(error::not_found(subject))
            } else if let Some((suffix, cluster)) = self.hosted.get(path) {
                println!("Gateway::get {}{}: {}", cluster, suffix, key);
                cluster.get(self.clone(), txn, suffix, key, auth).await
            } else {
                Err(error::not_found(path))
            }
        } else {
            Err(error::not_found(subject))
        }
    }

    pub async fn put(
        &self,
        subject: &Link,
        selector: Value,
        state: State,
        auth: &Auth,
        txn: Option<Arc<Txn>>,
    ) -> TCResult<State> {
        println!("Gateway::put {}: {} <- {}", subject, selector, state);

        if subject.host().is_some() {
            Err(error::not_implemented("Gateway::put over the network"))
        } else {
            let path = subject.path();
            if path[0] == "sbin" {
                return Err(error::method_not_allowed("/sbin is immutable"));
            }

            if let Some((suffix, cluster)) = self.hosted.get(path) {
                let txn =
                    txn.ok_or_else(|| error::unsupported("Cluster::put requires a Transaction"))?;

                println!("Gateway::put {}: {} <- {}", cluster, selector, state);

                if suffix.is_empty() {
                    cluster
                        .put(txn, selector.try_into()?, state.try_into()?, auth)
                        .map_ok(State::Cluster)
                        .await
                } else {
                    Err(error::not_found(suffix))
                }
            } else {
                Err(error::not_implemented("Peer cluster discovery"))
            }
        }
    }

    pub async fn post<S: Stream<Item = (ValueId, Value)> + Unpin>(
        self: Arc<Self>,
        subject: &Link,
        data: S,
        capture: &[ValueId],
        auth: Auth,
        txn_id: Option<TxnId>,
    ) -> TCResult<Vec<TCStream<Value>>> {
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
