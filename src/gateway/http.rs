use std::collections::{HashMap, HashSet};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::future;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, StatusCode};
use serde::de::DeserializeOwned;

use crate::auth::{Auth, Token};
use crate::class::{State, TCResult, TCStream};
use crate::error;
use crate::transaction::Txn;
use crate::value::link::*;
use crate::value::{Value, ValueId};

use super::Gateway;

const META: &str = "_meta";
const TIMEOUT: Duration = Duration::from_secs(30);
const ERR_DECODE: &str = "(unable to decode error message)";

pub struct Client {
    client: hyper::Client<hyper::client::HttpConnector, Body>,
    response_limit: usize,
}

impl Client {
    pub fn new(response_limit: usize) -> Client {
        let client = hyper::Client::builder()
            .pool_idle_timeout(TIMEOUT)
            .http2_only(true)
            .build_http();

        Client {
            client,
            response_limit,
        }
    }

    pub async fn get(
        &self,
        link: Link,
        key: &Value,
        auth: &Auth,
        txn: Option<Arc<Txn>>,
    ) -> TCResult<Value> {
        if auth.is_some() {
            return Err(error::not_implemented("Authorization"));
        }

        if txn.is_some() {
            return Err(error::not_implemented("Cross-service transactions"));
        }

        let host = link
            .host()
            .as_ref()
            .ok_or_else(|| error::bad_request("No host to resolve", &link))?;

        let host = if let Some(port) = host.port() {
            format!("{}:{}", host.address(), port)
        } else {
            host.address().to_string()
        };

        let path_and_query = if key == &Value::None {
            link.path().to_string()
        } else {
            let key: String = serde_json::to_string(key).map_err(error::TCError::from)?;
            format!("{}?key={}", link.path(), key)
        };

        let uri = format!("http://{}{}", host, path_and_query)
            .parse()
            .map_err(|err| error::bad_request("Unable to encode link URI", err))?;

        match self.client.get(uri).await {
            Err(cause) => Err(error::transport(cause)),
            Ok(response) if response.status() != 200 => {
                let status = response.status().as_u16();
                let msg = if let Ok(msg) = hyper::body::to_bytes(response).await {
                    if let Ok(msg) = String::from_utf8(msg.to_vec()) {
                        msg
                    } else {
                        ERR_DECODE.to_string()
                    }
                } else {
                    ERR_DECODE.to_string()
                };

                Err(error::TCError::of(status.into(), msg))
            }
            Ok(mut response) => deserialize_body(response.body_mut(), self.response_limit).await,
        }
    }
}

// TODO: implement request size limit
pub struct Server {
    address: SocketAddr,
    request_limit: usize,
}

impl Server {
    pub fn new(address: SocketAddr, request_limit: usize) -> Server {
        Server {
            address,
            request_limit,
        }
    }

    async fn handle(
        self: Arc<Self>,
        gateway: Arc<Gateway>,
        request: Request<Body>,
    ) -> Result<Response<Body>, hyper::Error> {
        match self.authenticate_and_route(gateway, request).await {
            Err(cause) => Ok(transform_error(cause)),
            Ok(response) => Ok(Response::new(Body::wrap_stream(response))),
        }
    }

    async fn authenticate_and_route(
        self: Arc<Self>,
        gateway: Arc<Gateway>,
        mut request: Request<Body>,
    ) -> TCResult<TCStream<TCResult<Bytes>>> {
        let token: Option<Token> = if let Some(header) = request.headers().get("Authorization") {
            let token = header
                .to_str()
                .map_err(|e| error::bad_request("Unable to parse Authorization header", e))?;
            Some(gateway.authenticate(token).await?)
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
                let id = get_param(&mut params, "key")?
                    .ok_or_else(|| error::bad_request("Missing URI parameter", "'key'"))?;
                let state = gateway.get(&path.clone().into(), id, &token, None).await?;

                match state {
                    State::Value(value) => {
                        let value = serde_json::to_string_pretty(&value)
                            .map(|json| format!("{}\r\n", json))
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
                let values: Vec<(ValueId, Value)> =
                    deserialize_body(request.body_mut(), self.request_limit).await?;

                let capture: Option<Vec<String>> = get_param(&mut params, "capture")?;
                let capture = if let Some(value_ids) = capture {
                    value_ids
                        .into_iter()
                        .map(|id| id.parse())
                        .collect::<TCResult<HashSet<ValueId>>>()?
                } else {
                    HashSet::new()
                };

                let response = gateway
                    .clone()
                    .post(
                        &path.clone().into(),
                        stream::iter(values.into_iter()),
                        capture,
                        &token,
                        None,
                    )
                    .await?;

                response_map(response)
            }
            other => Err(error::method_not_allowed(format!(
                "Tinychain does not support {}",
                other
            ))),
        }
    }
}

async fn deserialize_body<D: DeserializeOwned>(
    body: &mut hyper::Body,
    max_size: usize,
) -> TCResult<D> {
    let mut buffer = vec![];
    while let Some(chunk) = body.next().await {
        buffer.extend(chunk?.to_vec());

        if buffer.len() > max_size {
            return Err(error::too_large(max_size));
        }
    }

    let data = String::from_utf8(buffer)
        .map_err(|e| error::bad_request("Unable to parse request body", e))?;

    serde_json::from_str(&data)
        .map_err(|e| error::bad_request(&format!("Deserialization error {} when parsing", e), data))
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
impl super::Server for Server {
    type Error = hyper::Error;

    async fn listen(self: Arc<Self>, gateway: Arc<Gateway>) -> Result<(), Self::Error> {
        hyper::Server::bind(&self.address)
            .serve(make_service_fn(|_conn| {
                let this = self.clone();
                let gateway = gateway.clone();
                async {
                    Ok::<_, Infallible>(service_fn(move |request| {
                        this.clone().handle(gateway.clone(), request)
                    }))
                }
            }))
            .await
    }
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
        Ok => StatusCode::OK,
        BadRequest => StatusCode::BAD_REQUEST,
        Conflict => StatusCode::CONFLICT,
        Forbidden => StatusCode::FORBIDDEN,
        Internal => StatusCode::INTERNAL_SERVER_ERROR,
        MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
        NotFound => StatusCode::NOT_FOUND,
        NotImplemented => StatusCode::NOT_IMPLEMENTED,
        TooLarge => StatusCode::PAYLOAD_TOO_LARGE,
        Transport => StatusCode::from_u16(499).unwrap(), // custom status code
        Unauthorized => StatusCode::UNAUTHORIZED,
        Unknown => StatusCode::INTERNAL_SERVER_ERROR,
    };

    response
}
