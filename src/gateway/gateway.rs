use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::Duration;

use futures::future::TryFutureExt;
use futures::stream;
use log::debug;

use crate::auth::Token;
use crate::block::Dir;
use crate::class::{Public, State, TCBoxTryFuture, TCResult};
use crate::error;
use crate::kernel;
use crate::request::Request;
use crate::scalar::{Id, Link, Scalar, TCPath, TryCastInto, Value};
use crate::transaction::{Txn, TxnServer};

use super::http;
use super::{Hosted, NetworkTime, Server};

const ERR_BAD_POST_DATA: &str = "POST requires a list of (Id, Value) tuples, not";

pub struct Gateway {
    adapters: Vec<Link>,
    hosted: Hosted,
    client: http::Client,
    request_limit: usize,
    request_ttl: Duration,
    txn_server: TxnServer,
}

impl Gateway {
    pub fn time() -> NetworkTime {
        NetworkTime::now()
    }

    pub fn new(
        adapters: Vec<Link>,
        hosted: Hosted,
        workspace: Arc<Dir>,
        request_limit: usize,
        request_ttl: Duration,
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

        let client = http::Client::new(request_ttl, request_limit);
        let txn_server = TxnServer::new(workspace.clone());

        Ok(Gateway {
            adapters,
            hosted,
            client,
            request_limit,
            request_ttl,
            txn_server,
        })
    }

    pub async fn authenticate(&self, _token: &str) -> TCResult<Token> {
        Err(error::not_implemented("Gateway::authenticate"))
    }

    pub async fn transaction(self: &Arc<Self>, request: &Request) -> TCResult<Txn> {
        self.txn_server
            .new_txn(self.clone(), request.txn_id().clone())
            .await
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

    pub async fn get(
        &self,
        request: &Request,
        txn: &Txn,
        subject: &Link,
        key: Value,
    ) -> TCResult<State> {
        debug!("Gateway::get {}", subject);

        if subject.host().is_some() {
            Err(error::not_implemented("Gateway::get over the network"))
        } else if subject.path().as_slice().len() > 1 {
            let path = subject.path();
            if &path[0] == "sbin" {
                kernel::get(txn, &path[..], key).await
            } else if &path[0] == "ext" {
                for adapter in &self.adapters {
                    if path.as_slice().starts_with(adapter.path()) {
                        let host = adapter.host().as_ref().unwrap();
                        let dest: Link = (host.clone(), path.clone()).into();
                        return self
                            .client
                            .get(&request, txn, &dest, &key)
                            .map_ok(State::Scalar)
                            .await;
                    }
                }

                Err(error::not_found(subject))
            } else if let Some((suffix, cluster)) = self.hosted.get(path) {
                debug!("Gateway::get {}{}: {}", cluster, TCPath::from(suffix), key);
                cluster.get(request, txn, &suffix[..], key).await
            } else {
                Err(error::not_found(path))
            }
        } else {
            Err(error::not_found(subject))
        }
    }

    pub async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        subject: &Link,
        selector: Value,
        state: State,
    ) -> TCResult<()> {
        debug!("Gateway::put {}: {} <- {}", subject, selector, state);

        if subject.host().is_some() {
            Err(error::not_implemented("Gateway::put over the network"))
        } else {
            let path = subject.path();
            if &path[0] == "sbin" {
                return Err(error::method_not_allowed("/sbin is immutable"));
            }

            if let Some((suffix, cluster)) = self.hosted.get(path) {
                debug!(
                    "Gateway::put {}{}: {} <- {}",
                    cluster,
                    TCPath::from(suffix),
                    selector,
                    state
                );

                cluster.put(request, txn, suffix, selector, state).await
            } else {
                Err(error::path_not_found(path))
            }
        }
    }

    pub fn post<'a>(
        &'a self,
        request: &'a Request,
        txn: &'a Txn,
        subject: Link,
        data: Scalar,
    ) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            debug!("Gateway::post {}", subject);

            if subject.host().is_none()
                && !subject.path().is_empty()
                && &subject.path()[0] == "sbin"
            {
                return kernel::post(request, txn, &subject.into_path()[..], data).await;
            }

            let data: Vec<(Id, Scalar)> =
                data.try_cast_into(|v| error::bad_request(ERR_BAD_POST_DATA, v))?;

            if subject.host().is_none() {
                let path = subject.path();
                if let Some((suffix, cluster)) = self.hosted.get(path) {
                    cluster
                        .post(request, txn, suffix, data.into_iter().collect())
                        .await
                } else {
                    Err(error::path_not_found(path))
                }
            } else {
                self.client
                    .post(request, txn, subject, stream::iter(data.into_iter()))
                    .await
            }
        })
    }

    pub async fn delete(
        &self,
        request: &Request,
        txn: &Txn,
        subject: &Link,
        key: Value,
    ) -> TCResult<()> {
        debug!("Gateway::delete {}", subject);
        let path = subject.path();

        if subject.host().is_some() {
            Err(error::not_implemented("Gateway::delete over the network"))
        } else if path.len() > 1 {
            if let Some((suffix, cluster)) = self.hosted.get(path) {
                debug!(
                    "Gateway::delete {}{}: {}",
                    cluster,
                    TCPath::from(suffix),
                    key
                );
                cluster.delete(request, txn, &suffix[..], key).await
            } else {
                match path[0].as_str() {
                    "sbin" => match path[1].as_str() {
                        "chain" | "cluster" | "collection" | "object" | "op" | "slice"
                        | "value" => Err(error::method_not_allowed(&path[1])),
                        other => Err(error::not_found(other)),
                    },
                    other => Err(error::not_found(other)),
                }
            }
        } else if path.len() == 1 {
            if path[0].as_str() == "/sbin" {
                Err(error::method_not_allowed(path))
            } else {
                Err(error::not_found(path))
            }
        } else {
            Err(error::method_not_allowed(path))
        }
    }
}
