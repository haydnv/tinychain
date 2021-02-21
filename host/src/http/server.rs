use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use destream::de::FromStream;
use futures::{future, stream, StreamExt, TryFutureExt, TryStreamExt};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Response};
use log::debug;
use serde::de::DeserializeOwned;

use tc_error::*;
use tc_transact::{IntoView, TxnId};
use tcgeneric::{NetworkTime, TCPathBuf};

use crate::gateway::Gateway;
use crate::state::State;
use crate::txn::*;

const CONTENT_TYPE: &str = "application/json";

type GetParams = HashMap<String, String>;

/// Tinychain's HTTP server. Should only be used through a [`Gateway`].
pub struct HTTPServer {
    gateway: Arc<Gateway>,
}

impl HTTPServer {
    pub fn new(gateway: Arc<Gateway>) -> Self {
        Self { gateway }
    }

    async fn handle(
        self: Arc<Self>,
        request: hyper::Request<Body>,
    ) -> Result<Response<Body>, hyper::Error> {
        let (params, txn) = match self.process_headers(&request).await {
            Ok((params, txn)) => (params, txn),
            Err(cause) => return Ok(transform_error(cause)),
        };

        match self.route(&txn, params, request).await {
            Ok(state) => match destream_json::encode(state.into_view(txn)) {
                Ok(response) => {
                    let mut response = Response::new(Body::wrap_stream(
                        response.chain(stream::once(future::ready(Ok(b"\n".to_vec())))),
                    ));

                    response
                        .headers_mut()
                        .insert(hyper::header::CONTENT_TYPE, CONTENT_TYPE.parse().unwrap());

                    Ok(response)
                }
                Err(cause) => Ok(transform_error(TCError::internal(cause))),
            },
            Err(cause) => Ok(transform_error(cause)),
        }
    }

    async fn process_headers(
        &self,
        http_request: &hyper::Request<Body>,
    ) -> TCResult<(GetParams, Txn)> {
        let mut params = http_request
            .uri()
            .query()
            .map(|v| {
                debug!("param {}", v);
                url::form_urlencoded::parse(v.as_bytes())
                    .into_owned()
                    .collect()
            })
            .unwrap_or_else(HashMap::new);

        let token = if let Some(header) = http_request.headers().get(hyper::header::AUTHORIZATION) {
            let token = header.to_str().map_err(|e| {
                TCError::unauthorized(format!("unable to parse authorization header: {}", e))
            })?;

            if token.starts_with("Bearer") {
                Some(token[6..].trim().to_string())
            } else {
                return Err(TCError::unauthorized(format!(
                    "unable to parse authorization header: {} (should start with \"Bearer\"",
                    token
                )));
            }
        } else {
            None
        };

        let txn_id = if let Some(txn_id) = params.remove("txn_id") {
            txn_id.parse()?
        } else {
            TxnId::new(NetworkTime::now())
        };

        let txn = self.gateway.new_txn(txn_id, token).await?;
        Ok((params, txn))
    }

    async fn route(
        &self,
        txn: &Txn,
        mut params: GetParams,
        http_request: hyper::Request<Body>,
    ) -> TCResult<State> {
        let path: TCPathBuf = http_request.uri().path().parse()?;

        match http_request.method() {
            &hyper::Method::GET => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                self.gateway.get(txn, path.into(), key).await
            }

            &hyper::Method::PUT => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                let value = destream_body(http_request.into_body(), txn.clone()).await?;
                self.gateway
                    .put(txn, path.into(), key, value)
                    .map_ok(State::from)
                    .await
            }

            &hyper::Method::POST => {
                let data = destream_body(http_request.into_body(), txn.clone()).await?;
                self.gateway.post(txn, path.into(), data).await
            }

            other => Err(TCError::method_not_allowed(other)),
        }
    }
}

#[async_trait]
impl crate::gateway::Server for HTTPServer {
    type Error = hyper::Error;

    async fn listen(self, addr: SocketAddr) -> Result<(), Self::Error> {
        println!("HTTP server listening on {}", &addr);
        let server = Arc::new(self);

        let new_service = make_service_fn(move |_| {
            let server = server.clone();
            async {
                Ok::<_, hyper::Error>(service_fn(move |req| {
                    let server = server.clone();
                    HTTPServer::handle(server, req)
                }))
            }
        });

        hyper::Server::bind(&addr)
            .serve(new_service)
            .with_graceful_shutdown(shutdown_signal())
            .await
    }
}

async fn destream_body(body: hyper::Body, txn: Txn) -> TCResult<State> {
    let data = body
        .map_ok(|bytes| bytes.to_vec())
        .map_err(destream::de::Error::custom);

    let mut decoder = destream_json::de::Decoder::from(data);
    State::from_stream(txn, &mut decoder)
        .map_err(|e| TCError::bad_request("error deserializing HTTP request body", e))
        .await
}

fn get_param<T: DeserializeOwned>(
    params: &mut HashMap<String, String>,
    name: &str,
) -> TCResult<Option<T>> {
    if let Some(param) = params.remove(name) {
        let val: T = serde_json::from_str(&param).map_err(|e| {
            TCError::bad_request(&format!("Unable to parse URI parameter '{}'", name), e)
        })?;

        Ok(Some(val))
    } else {
        Ok(None)
    }
}

fn transform_error(err: TCError) -> hyper::Response<Body> {
    let mut response = hyper::Response::new(Body::from(format!("{}\r\n", err.message())));

    use hyper::StatusCode;
    use tc_error::ErrorType::*;
    *response.status_mut() = match err.code() {
        BadGateway => StatusCode::BAD_GATEWAY,
        BadRequest => StatusCode::BAD_REQUEST,
        Forbidden => StatusCode::FORBIDDEN,
        Conflict => StatusCode::CONFLICT,
        Internal => StatusCode::INTERNAL_SERVER_ERROR,
        MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
        NotFound => StatusCode::NOT_FOUND,
        NotImplemented => StatusCode::NOT_IMPLEMENTED,
        Timeout => StatusCode::REQUEST_TIMEOUT,
        Unauthorized => StatusCode::UNAUTHORIZED,
    };

    response
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.expect("SIGTERM handler")
}
