use std::collections::HashMap;
use std::fmt;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use destream::de::Error;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Response};
use serde::de::DeserializeOwned;

use tc_error::*;
use tc_transact::{IntoView, TxnId};
use tcgeneric::{NetworkTime, TCPathBuf};

use crate::gateway::Gateway;
use crate::state::State;
use crate::txn::*;

use super::{Accept, Encoding};

type GetParams = HashMap<String, String>;

/// TinyChain's HTTP server. Should only be used through a [`Gateway`].
pub struct HTTPServer {
    gateway: Arc<Gateway>,
}

impl HTTPServer {
    pub fn new(gateway: Arc<Gateway>) -> Self {
        Self { gateway }
    }

    async fn handle_timeout(
        self: Arc<Self>,
        request: hyper::Request<Body>,
    ) -> Result<Response<Body>, hyper::Error> {
        match tokio::time::timeout(self.gateway.request_ttl(), self.handle(request)).await {
            Ok(result) => result,
            Err(cause) => Ok(transform_error(
                timeout!("request timed out").consume(cause),
                Encoding::default(),
            )),
        }
    }

    async fn handle(
        self: Arc<Self>,
        request: hyper::Request<Body>,
    ) -> Result<Response<Body>, hyper::Error> {
        let (params, txn, accept_encoding, request_encoding) =
            match self.process_headers(&request).await {
                Ok(header_data) => header_data,
                Err(cause) => return Ok(transform_error(cause, Encoding::default())),
            };

        let state = match self.route(request_encoding, &txn, params, request).await {
            Ok(state) => state,
            Err(cause) => return Ok(transform_error(cause, accept_encoding)),
        };

        let view = match state.into_view(txn).await {
            Ok(view) => view,
            Err(cause) => return Ok(transform_error(cause, accept_encoding)),
        };

        let body = match accept_encoding {
            Encoding::Json => match destream_json::encode(view) {
                Ok(response) => Body::wrap_stream(response.chain(delimiter(b"\n"))),
                Err(cause) => {
                    return Ok(transform_error(
                        unexpected!("JSON encoding error").consume(cause),
                        Encoding::Json,
                    ))
                }
            },
            Encoding::Tbon => match tbon::en::encode(view) {
                Ok(response) => {
                    let response =
                        response.map_err(|cause| unexpected!("TBON encoding error").consume(cause));

                    Body::wrap_stream(response)
                }
                Err(cause) => {
                    return Ok(transform_error(
                        unexpected!("TBON encoding error").consume(cause),
                        Encoding::Tbon,
                    ))
                }
            },
        };

        let mut response = Response::new(body);

        response.headers_mut().insert(
            hyper::header::CONTENT_TYPE,
            accept_encoding
                .as_str()
                .parse()
                .expect("content type header"),
        );

        Ok(response)
    }

    async fn process_headers(
        &self,
        http_request: &hyper::Request<Body>,
    ) -> TCResult<(GetParams, Txn, Encoding, Encoding)> {
        let content_type =
            if let Some(header) = http_request.headers().get(hyper::header::CONTENT_TYPE) {
                header
                    .to_str()
                    .map_err(|cause| bad_request!("invalid Content-Type header").consume(cause))?
                    .parse()?
            } else {
                Encoding::default()
            };

        let accept_encoding = http_request.headers().get(hyper::header::ACCEPT_ENCODING);
        let accept_encoding = Encoding::parse_header(accept_encoding)?;

        let mut params = http_request
            .uri()
            .query()
            .map(|v| {
                url::form_urlencoded::parse(v.as_bytes())
                    .into_owned()
                    .collect()
            })
            .unwrap_or_else(HashMap::new);

        let token = if let Some(header) = http_request.headers().get(hyper::header::AUTHORIZATION) {
            let token = header
                .to_str()
                .map_err(|e| unauthorized!("unable to parse authorization header: {}", e))?;

            if token.starts_with("Bearer") {
                Some(token[6..].trim().to_string())
            } else {
                return Err(unauthorized!(
                    "unable to parse authorization header: {} (should start with \"Bearer\"",
                    token
                ));
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
        Ok((params, txn, accept_encoding, content_type))
    }

    async fn route(
        &self,
        encoding: Encoding,
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
                let value = destream_body(http_request.into_body(), encoding, txn.clone()).await?;
                let result = self
                    .gateway
                    .put(txn, path.into(), key, value)
                    .map_ok(State::from)
                    .await;

                #[cfg(debug_assertions)]
                log::trace!("PUT request completed with result {:?}", result);

                result
            }

            &hyper::Method::POST => {
                let data = destream_body(http_request.into_body(), encoding, txn.clone()).await?;
                self.gateway.post(txn, path.into(), data).await
            }

            &hyper::Method::DELETE => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                self.gateway
                    .delete(txn, path.into(), key)
                    .map_ok(State::from)
                    .await
            }

            other => Err(TCError::method_not_allowed(other, self, path)),
        }
    }
}

#[async_trait]
impl crate::gateway::Server for HTTPServer {
    type Error = hyper::Error;

    async fn listen(self, port: u16) -> Result<(), Self::Error> {
        println!("HTTP server listening on port {}...", port);
        println!();

        let server = Arc::new(self);

        let new_service = make_service_fn(move |_| {
            let server = server.clone();

            async {
                Ok::<_, hyper::Error>(service_fn(move |req| {
                    let server = server.clone();
                    HTTPServer::handle_timeout(server, req)
                }))
            }
        });

        let addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), port);
        hyper::Server::bind(&addr)
            .serve(new_service)
            .with_graceful_shutdown(shutdown_signal())
            .await
    }
}

impl fmt::Display for HTTPServer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("HTTP server")
    }
}

async fn destream_body(body: hyper::Body, encoding: Encoding, txn: Txn) -> TCResult<State> {
    const ERR_DESERIALIZE: &str = "error deserializing HTTP request body";

    match encoding {
        Encoding::Json => {
            destream_json::try_decode(txn, body)
                .map_err(|cause| bad_request!("{}", ERR_DESERIALIZE).consume(cause))
                .await
        }
        Encoding::Tbon => {
            tbon::de::try_decode(txn, body)
                .map_err(|cause| bad_request!("{}", ERR_DESERIALIZE).consume(cause))
                .await
        }
    }
}

fn get_param<T: DeserializeOwned>(
    params: &mut HashMap<String, String>,
    name: &str,
) -> TCResult<Option<T>> {
    if let Some(param) = params.remove(name) {
        let val: T = serde_json::from_str(&param).map_err(|cause| {
            TCError::invalid_value(param, format!("URI parameter {}", name)).consume(cause)
        })?;

        Ok(Some(val))
    } else {
        Ok(None)
    }
}

fn transform_error(err: TCError, encoding: Encoding) -> hyper::Response<Body> {
    use hyper::StatusCode;
    use tc_error::ErrorKind::*;

    let code = match err.code() {
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
        Unavailable => StatusCode::SERVICE_UNAVAILABLE,
    };

    let body = match encoding {
        Encoding::Json => {
            let encoded = destream_json::encode(err).expect("encode error");
            let encoded = encoded.chain(delimiter(b"\n"));
            Body::wrap_stream(encoded)
        }
        Encoding::Tbon => Body::wrap_stream(tbon::en::encode(err).expect("encode error")),
    };

    let mut response = hyper::Response::new(body);

    response.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        encoding.as_str().parse().expect("content encoding"),
    );

    *response.status_mut() = code;

    response
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.expect("SIGTERM handler")
}

fn delimiter<E>(content: &'static [u8]) -> impl Stream<Item = Result<Bytes, E>> {
    stream::once(future::ready(Ok(Bytes::from_static(content))))
}
