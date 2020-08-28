use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::de::DeserializeOwned;

use crate::auth::Token;
use crate::class::{State, TCResult, TCStream};
use crate::error;
use crate::gateway::{Gateway, Protocol};
use crate::value::link::*;
use crate::value::{Value, ValueId};

const META: &str = "_meta";

pub struct Http {
    address: SocketAddr,
    gateway: Arc<Gateway>,
    request_limit: usize,
}

impl Http {
    pub fn new(address: SocketAddr, gateway: Arc<Gateway>, request_limit: usize) -> Arc<Http> {
        Arc::new(Http {
            address,
            gateway,
            request_limit,
        })
    }

    fn get_param<T: DeserializeOwned>(
        params: &mut HashMap<String, String>,
        name: &str,
    ) -> TCResult<Option<T>> {
        if let Some(param) = params.remove(name) {
            let val: T = serde_json::from_str(&param).map_err(|e| {
                error::bad_request(&format!("Unable to parse URI parameter '{}'", name), e)
            })?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    fn transform_error(err: error::TCError) -> Response<Body> {
        let mut response = Response::new(Body::from(err.message().to_string()));

        use error::Code::*;
        *response.status_mut() = match err.reason() {
            BadRequest => StatusCode::BAD_REQUEST,
            Conflict => StatusCode::CONFLICT,
            Forbidden => StatusCode::FORBIDDEN,
            Internal => StatusCode::INTERNAL_SERVER_ERROR,
            MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
            NotFound => StatusCode::NOT_FOUND,
            NotImplemented => StatusCode::NOT_IMPLEMENTED,
            RequestTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            Unauthorized => StatusCode::UNAUTHORIZED,
        };

        response
    }

    async fn handle(
        self: Arc<Self>,
        request: Request<Body>,
    ) -> Result<Response<Body>, hyper::Error> {
        match self.authenticate_and_route(request).await {
            Err(cause) => Ok(Http::transform_error(cause)),
            Ok(response) => Ok(Response::new(Body::wrap_stream(response))),
        }
    }

    async fn authenticate_and_route(
        self: Arc<Self>,
        request: Request<Body>,
    ) -> TCResult<TCStream<TCResult<Bytes>>> {
        // TODO: use self.request_limit to restrict request size

        let token: Option<Token> = if let Some(header) = request.headers().get("Authorization") {
            let token = header
                .to_str()
                .map_err(|e| error::bad_request("Unable to parse Authorization header", e))?;
            Some(self.gateway.authenticate(token).await?)
        } else {
            None
        };

        let uri = request.uri().clone();
        let path: TCPath = uri.path().parse()?;
        let mut params: HashMap<String, String> = uri
            .query()
            .map(|v| {
                println!("param {}", v);
                url::form_urlencoded::parse(v.as_bytes())
                    .into_owned()
                    .collect()
            })
            .unwrap_or_else(HashMap::new);

        match request.method() {
            &Method::GET => {
                let id = Http::get_param(&mut params, "key")?
                    .ok_or_else(|| error::bad_request("Missing URI parameter", "'key'"))?;
                let state = self
                    .gateway
                    .get(&path.clone().into(), id, &token, None)
                    .await?;

                match state {
                    State::Value(value) => {
                        let value = serde_json::to_string_pretty(&value)
                            .map(Bytes::from)
                            .map_err(error::TCError::from);
                        Ok(Box::pin(stream::once(future::ready(value))))
                    }
                    _other => Ok(Box::pin(stream::once(future::ready(Err(
                        error::not_implemented("serializing a State over the network"),
                    ))))),
                }
            }
            &Method::PUT => Err(error::not_implemented("HTTP PUT")),
            &Method::POST => {
                println!("POST {}", path);
                let values = String::from_utf8(hyper::body::to_bytes(request).await?.to_vec())
                    .map_err(|e| error::bad_request("Unable to parse request body", e))?;
                let values: Vec<(ValueId, Value)> = serde_json::from_str(&values).map_err(|e| {
                    error::bad_request(&format!("Deserialization error {} when parsing", e), values)
                })?;

                let capture: Option<Vec<String>> = Http::get_param(&mut params, "capture")?;
                let capture = if let Some(value_ids) = capture {
                    value_ids
                        .into_iter()
                        .map(|id| id.parse())
                        .collect::<TCResult<HashSet<ValueId>>>()?
                } else {
                    HashSet::new()
                };

                let state = self
                    .gateway
                    .clone()
                    .post(
                        &path.clone().into(),
                        stream::iter(values.into_iter()),
                        &token,
                        None,
                    )
                    .await?;

                println!("post context has {} states", state.len());

                let response = state
                    .into_iter()
                    .filter(move |(name, _)| capture.contains(name))
                    .map(|(name, state)| {
                        println!("txn state {}: {}", name, state);
                        let values: TCStream<Value> = match state {
                            State::Value(value) => Box::pin(stream::once(future::ready(value))),
                            _other => Box::pin(stream::empty()),
                        };
                        (name, values)
                    });

                response_map(stream::iter(response))
            }
            other => Err(error::method_not_allowed(format!(
                "Tinychain does not support {}",
                other
            ))),
        }
    }
}

fn response_list<S: Stream<Item = Value> + Send + Sync + Unpin + 'static>(
    s: S,
) -> TCStream<TCResult<Bytes>> {
    let start = stream_delimiter(Bytes::copy_from_slice(b"["));

    let items = s
        .map(|v| serde_json::to_string_pretty(&v))
        .map_ok(Bytes::from)
        .map_err(error::TCError::from);
    let items: TCStream<TCResult<Bytes>> = Box::pin(items);

    let end = stream_delimiter(Bytes::copy_from_slice(b"]"));

    Box::pin(start.chain(items).chain(end))
}

fn response_map<
    V: Stream<Item = Value> + Send + Sync + Unpin + 'static,
    S: Stream<Item = (ValueId, V)> + Send + Sync + Unpin + 'static,
>(
    s: S,
) -> TCResult<TCStream<TCResult<Bytes>>> {
    let start = stream_delimiter(Bytes::copy_from_slice(b"{"));

    let meta = stream_delimiter(Bytes::copy_from_slice(
        format!("\"{}\": {{}}", META).as_bytes(), // TODO: include execution metadata
    ));

    let items = s
        .map(|(name, values)| {
            let start = Bytes::copy_from_slice(format!("\"{}\": ", name).as_bytes());
            let start = stream_delimiter(start);
            let values = response_list(values);
            let end = stream_delimiter(Bytes::copy_from_slice(b", "));

            Box::pin(start.chain(values).chain(end))
        })
        .flatten();
    let items: TCStream<TCResult<Bytes>> = Box::pin(items.chain(meta));

    let end = stream_delimiter(Bytes::copy_from_slice(b"}"));

    Ok(Box::pin(start.chain(items).chain(end)))
}

fn stream_delimiter(token: Bytes) -> TCStream<TCResult<Bytes>> {
    Box::pin(stream::once(future::ready(Ok(token))))
}

#[async_trait]
impl Protocol for Http {
    type Error = hyper::Error;

    async fn listen(self: Arc<Self>) -> Result<(), Self::Error> {
        Server::bind(&self.address)
            .serve(make_service_fn(|_conn| {
                let self_clone = self.clone();
                async { Ok::<_, Infallible>(service_fn(move |req| self_clone.clone().handle(req))) }
            }))
            .await
    }
}
