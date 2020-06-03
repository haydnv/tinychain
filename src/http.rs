use std::collections::HashMap;
use std::convert::{Infallible, TryInto};
use std::io::{self, BufReader, Write};
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{BufMut, Bytes, BytesMut};
use futures::executor::block_on;
use futures::future;
use futures::stream::{self, Stream, StreamExt};
use hyper::header::HeaderValue;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};

use crate::auth::Token;
use crate::error;
use crate::gateway::op;
use crate::gateway::{Gateway, Protocol};
use crate::host::Host;
use crate::internal::Dir;
use crate::state::State;
use crate::transaction::Txn;
use crate::value::link::*;
use crate::value::{TCRef, TCResult, Value, ValueId};

struct StreamReader<S: Stream<Item = Result<Bytes, hyper::Error>>> {
    source: S,
    buffered: Bytes,
}

impl<S: Stream<Item = Result<Bytes, hyper::Error>>> From<S> for StreamReader<S> {
    fn from(source: S) -> StreamReader<S> {
        StreamReader {
            source,
            buffered: Bytes::from(&[][..]),
        }
    }
}

impl<S: Stream<Item = Result<Bytes, hyper::Error>> + Unpin> io::Read for StreamReader<S> {
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        if buf.len() < self.buffered.len() {
            self.buffered = Bytes::copy_from_slice(&self.buffered[buf.len()..]);
            buf.write(&self.buffered[..buf.len()])
        } else {
            let mut buffer = BytesMut::with_capacity(buf.len());
            buffer.put(&self.buffered[..]);

            loop {
                if let Some(chunk) = block_on(self.source.next()) {
                    match chunk {
                        Ok(chunk) => buffer.put(chunk),
                        Err(cause) => {
                            return Err(io::Error::new(io::ErrorKind::InvalidInput, cause))
                        }
                    }
                } else {
                    break;
                }

                if buffer.len() > buf.len() {
                    break;
                }
            }

            let buffer = buffer.freeze();
            if buffer.is_empty() {
                Ok(0)
            } else if buffer.len() < buf.len() {
                self.buffered = Bytes::from(&[][..]);
                buf.write(&buffer[..])
            } else {
                self.buffered = Bytes::copy_from_slice(&buffer[buf.len()..]);
                buf.write(&buffer[..buf.len()])
            }
        }
    }
}

pub struct Http {
    address: SocketAddr,
    gateway: Arc<Gateway>,
    workspace: Arc<Dir>,
}

impl Http {
    fn get_param(mut params: HashMap<String, String>, name: &str) -> TCResult<Value> {
        if let Some(param) = params.remove(name) {
            serde_json::from_str(&param).map_err(|e| {
                error::bad_request(&format!("Unable to parse URI parameter '{}'", name), e)
            })
        } else {
            Err(error::bad_request("Missing parameter", "key"))
        }
    }

    async fn handle(self: Arc<Self>, req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
        match self.authenticate_and_route(req).await {
            Ok(stream) => Ok(Response::new(Body::wrap_stream(stream))),
            Err(cause) => Ok(transform_error(cause)),
        }
    }

    async fn authenticate_and_route(
        self: Arc<Self>,
        mut req: Request<Body>,
    ) -> TCResult<impl Stream<Item = TCResult<Bytes>>> {
        let _token: Option<Token> = if let Some(header) = req.headers().get("Authorization") {
            let token = header
                .to_str()
                .map_err(|e| error::bad_request("Unable to parse Authorization header", e))?;
            Some(self.gateway.authenticate(token).await?)
        } else {
            None
        };

        let uri = req.uri().clone();
        let path: TCPath = uri.path().parse()?;
        let params: HashMap<String, String> = uri
            .query()
            .map(|v| {
                url::form_urlencoded::parse(v.as_bytes())
                    .into_owned()
                    .collect()
            })
            .unwrap_or_else(HashMap::new);

        match req.method() {
            &Method::GET => {
                let id = Http::get_param(params, "key")?;
                let op: op::Get = (id,).into();
                Http::encode_response(self.gateway.get(path.into(), op).await?)
            }
            &Method::PUT => {
                let reader = StreamReader::from(req.body_mut());
                let reader = BufReader::new(reader);
                let op = op::Put::from(stream::iter(serde_json::from_reader(reader).into_iter()));
                Http::encode_response(self.gateway.put(path.into(), op).await?)
            }
            &Method::POST => Err(error::not_implemented()),
            other => Err(error::bad_request(
                "Tinychain does not support this HTTP method",
                other,
            )),
        }
    }

    fn encode_response(state: State) -> TCResult<impl Stream<Item = TCResult<Bytes>>> {
        match state {
            State::Value(value) => Ok(stream::once(future::ready(
                serde_json::to_string(&value)
                    .map(|s| s.into())
                    .map_err(|e| {
                        error::internal(format!("Unable to serialize the requested value: {}", e))
                    }),
            ))),
            // TODO: support State::Stream
            other => Err(error::bad_request(
                "Unable to serialize requested State",
                other,
            )),
        }
    }
}

#[async_trait]
impl Protocol for Http {
    type Config = SocketAddr;
    type Error = hyper::Error;

    fn new(address: SocketAddr, gateway: Arc<Gateway>, workspace: Arc<Dir>) -> Arc<Http> {
        Arc::new(Http {
            address,
            gateway,
            workspace,
        })
    }

    async fn listen(self: Arc<Self>) -> Result<(), Self::Error> {
        Server::bind(&self.address)
            .serve(make_service_fn(|_conn| {
                let self_clone = self.clone();
                async { Ok::<_, Infallible>(service_fn(move |req| self_clone.clone().handle(req))) }
            }))
            .await
    }
}

const UNSERIALIZABLE: &str =
    "The request completed successfully but some of the response could not be serialized";

fn line_numbers(s: &str) -> String {
    s.lines()
        .enumerate()
        .map(|(i, l)| format!("{} {}", i + 1, l))
        .collect::<Vec<String>>()
        .join("\n")
}

fn transform_error(err: error::TCError) -> Response<Body> {
    let mut response = Response::new(Body::from(err.message().to_string()));
    *response.status_mut() = match err.reason() {
        error::Code::BadRequest => StatusCode::BAD_REQUEST,
        error::Code::Forbidden => StatusCode::FORBIDDEN,
        error::Code::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        error::Code::MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
        error::Code::NotFound => StatusCode::NOT_FOUND,
        error::Code::NotImplemented => StatusCode::NOT_IMPLEMENTED,
        error::Code::Unauthorized => StatusCode::UNAUTHORIZED,
    };
    response
}

// TODO: DELETE BELOW THIS LINE!
pub async fn listen(
    host: Arc<Host>,
    address: &SocketAddr,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let make_svc = make_service_fn(|_conn| {
        let host = host.clone();
        async { Ok::<_, Infallible>(service_fn(move |req| handle_old(host.clone(), req))) }
    });

    let server = Server::bind(address).serve(make_svc);

    println!("Listening on http://{}", address);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
    }

    Ok(())
}

async fn get<'a>(
    txn: &'a Arc<Txn<'a>>,
    path: TCPath,
    key: Value,
    auth: &Option<Token>,
) -> TCResult<State> {
    txn.get(path.into(), key, auth).await
}

async fn post<'a>(
    txn: &'a Arc<Txn<'a>>,
    path: &TCPath,
    mut args: HashMap<ValueId, Value>,
    auth: &'a Option<Token>,
) -> TCResult<State> {
    if path == "/sbin/transact" {
        let capture: Vec<ValueId> = args
            .remove(&"capture".parse().unwrap())
            .map(|v| v.try_into())
            .unwrap_or_else(|| Ok(Vec::new()))?;
        let mut values: Vec<(ValueId, Value)> = args
            .remove(&"values".parse().unwrap())
            .map(|v| v.try_into())
            .unwrap_or_else(|| Ok(Vec::new()))?;
        txn.extend(values.drain(..), auth).await?;

        let mut results: Vec<Value> = Vec::with_capacity(capture.len());
        match txn.resolve(capture.into_iter().collect()).await {
            Ok(responses) => {
                for (id, r) in responses {
                    match r {
                        State::Value(val) => {
                            results.push((TCRef::from(id), val).into());
                        }
                        other => {
                            txn.rollback().await;
                            return Err(error::bad_request(
                                "Attempt to capture an unserializable value",
                                other,
                            ));
                        }
                    }
                }
            }
            Err(cause) => {
                return Err(cause);
            }
        };

        txn.commit().await;

        Ok(State::Value(results.into()))
    } else {
        Err(error::method_not_allowed(path))
    }
}

async fn route<'a>(
    txn: &'a Arc<Txn<'a>>,
    method: Method,
    path: &str,
    params: HashMap<String, String>,
    body: Vec<u8>,
    auth: &'a Option<Token>,
) -> TCResult<Vec<u8>> {
    let path: TCPath = path.parse()?;

    match method {
        Method::GET => {
            let key = if let Some(key) = params.get("key") {
                serde_json::from_str::<Value>(key)
                    .map_err(|e| error::bad_request("Unable to parse 'key' param", e))?
            } else {
                Value::None
            };

            match get(txn, path, key, &auth).await? {
                State::Value(val) => Ok(serde_json::to_string_pretty(&val)?.as_bytes().to_vec()),
                state => Err(error::bad_request(
                    "Attempt to GET unserializable state {}",
                    state,
                )),
            }
        }
        Method::POST => {
            let args: HashMap<ValueId, Value> = match serde_json::from_slice(&body) {
                Ok(params) => params,
                Err(cause) => {
                    let body = line_numbers(std::str::from_utf8(&body).unwrap());
                    return Err(error::bad_request(
                        &format!("{}\n\nUnable to parse request", body),
                        cause,
                    ));
                }
            };

            match post(txn, &path, args, auth).await? {
                State::Value(v) => serde_json::to_string_pretty(&v)
                    .and_then(|s| Ok(s.into_bytes()))
                    .or_else(|e| Err(error::bad_request(UNSERIALIZABLE, e))),
                other => Err(error::bad_request(UNSERIALIZABLE, other)),
            }
        }
        _ => Err(error::not_found(path)),
    }
}

async fn handle_old(host: Arc<Host>, req: Request<Body>) -> Result<Response<Body>, hyper::Error> {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path();

    let params: HashMap<String, String> = uri
        .query()
        .map(|v| {
            url::form_urlencoded::parse(v.as_bytes())
                .into_owned()
                .collect()
        })
        .unwrap_or_else(HashMap::new);

    let txn = match host.new_transaction().await {
        Ok(txn) => txn,
        Err(cause) => return Ok(transform_error(cause)),
    };

    let token = if let Some(header) = req.headers().get("Authorization") {
        match validate_token(txn.clone(), header).await {
            Ok(token) => Some(token),
            Err(cause) => return Ok(transform_error(cause)),
        }
    } else {
        None
    };

    let body = &hyper::body::to_bytes(req.into_body()).await?;
    match route(&txn, method, path, params, body.to_vec(), &token).await {
        Ok(bytes) => Ok(Response::new(Body::from(bytes))),
        Err(cause) => Ok(transform_error(cause)),
    }
}

async fn validate_token(txn: Arc<Txn<'_>>, auth_header: &HeaderValue) -> TCResult<Token> {
    match auth_header.to_str() {
        Ok(t) => {
            if t.starts_with("Bearer: ") {
                let token: Token = t[8..].parse()?;
                let value_id: ValueId = "__actor_id".parse().unwrap();
                txn.push((value_id.clone(), token.actor_id().into()), &None)
                    .await?;
                txn.resolve(vec![value_id.clone()])
                    .await
                    .map(|_actor| Err(error::not_implemented()))?
            } else {
                Err(error::unauthorized(&format!(
                    "Invalid authorization header: {}",
                    t
                )))
            }
        }
        Err(cause) => Err(error::unauthorized(&cause.to_string())),
    }
}
