use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;

use futures::future::{FutureExt, TryFutureExt};
use futures::stream::{self, FuturesUnordered, StreamExt};

use crate::auth::{Auth, Token};
use crate::block::Dir;
use crate::class::{State, TCBoxTryFuture, TCResult};
use crate::error;
use crate::kernel;
use crate::scalar::{Link, LinkHost, Scalar, TryCastInto, Value, ValueId};
use crate::transaction::Txn;

use super::http;
use super::{Hosted, NetworkTime, Server};

const ERR_BAD_POSTDATA: &str = "POST requires a list of (ValueId, Value) tuples, not";

pub struct Gateway {
    peers: Vec<LinkHost>,
    adapters: Vec<Link>,
    hosted: Hosted,
    workspace: Arc<Dir>,
    client: http::Client,
    request_limit: usize,
    request_ttl: u32,
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
        request_ttl: u32,
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
            request_ttl,
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
            self.request_ttl,
        ));

        server.listen(self).await
    }

    pub async fn discover(
        self: Arc<Self>,
        subject: &Link,
        auth: Auth,
        txn: Option<Arc<Txn>>,
    ) -> TCResult<Vec<LinkHost>> {
        let mut requests = FuturesUnordered::new();
        for peer in &self.peers {
            requests.push(
                self.client
                    .get(subject, &Value::None, &auth, &txn)
                    .map(move |response| (peer.clone(), response)),
            );
        }

        let mut found = Vec::with_capacity(self.peers.len());
        while let Some((peer, response)) = requests.next().await {
            match (peer, response) {
                (peer, Ok(_)) => found.push(peer),
                (peer, Err(cause)) => println!("GET {}{}: {}", peer, subject, cause),
            }
        }

        Err(error::not_implemented("Gateway::discover"))
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
                            .get(&dest, &key, &auth, &txn)
                            .map_ok(State::Scalar)
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
        self: Arc<Self>,
        subject: &Link,
        selector: Value,
        state: State,
        auth: &Auth,
        txn: Option<Arc<Txn>>,
    ) -> TCResult<()> {
        println!("Gateway::put {}: {} <- {}", subject, selector, state);

        if subject.host().is_some() {
            Err(error::not_implemented("Gateway::put over the network"))
        } else {
            let path = subject.path();
            if path[0] == "sbin" {
                return Err(error::method_not_allowed("/sbin is immutable"));
            }

            if let Some((suffix, cluster)) = self.hosted.get(path) {
                println!(
                    "Gateway::put {}{}: {} <- {}",
                    cluster, suffix, selector, state
                );

                cluster
                    .put(self.clone(), txn, path, selector, state, auth)
                    .await
            } else {
                Err(error::not_implemented("Peer cluster discovery"))
            }
        }
    }

    pub fn post<'a>(
        self: Arc<Self>,
        subject: Link,
        data: Scalar,
        auth: Auth,
        txn: Option<Arc<Txn>>,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            println!("Gateway::post {}", subject);

            let txn = if let Some(txn) = txn {
                txn
            } else {
                Txn::new(self.clone(), self.workspace.clone()).await?
            };

            if subject.host().is_none() && !subject.path().is_empty() && subject.path()[0] == "sbin"
            {
                return kernel::post(txn, subject.into_path(), data, auth).await;
            }

            let data: Vec<(ValueId, Scalar)> =
                data.try_cast_into(|v| error::bad_request(ERR_BAD_POSTDATA, v))?;
            let data = stream::iter(data);

            if subject.host().is_none() {
                let path = subject.path();
                if let Some((suffix, cluster)) = self.hosted.get(path) {
                    cluster.post(txn, suffix, data, auth).await
                } else {
                    Err(error::not_implemented("Peer discovery"))
                }
            } else {
                self.client.post(subject, data, auth, None).await
            }
        })
    }
}
