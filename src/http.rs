use std::collections::HashMap;
use std::convert::Infallible;
use std::io::{self, BufReader, Write};
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{BufMut, Bytes, BytesMut};
use futures::executor::block_on;
use futures::future;
use futures::stream::{self, Stream};
use futures_util::stream::StreamExt;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::de::DeserializeOwned;

use crate::auth::Token;
use crate::class::{State, TCResult, TCStream};
use crate::error;
use crate::gateway::{Gateway, Protocol};
use crate::transaction::TxnId;
use crate::value::link::*;
use crate::value::Value;

struct StreamReader<S: Stream<Item = Result<Bytes, hyper::Error>>> {
    source: S,
    buffered: Bytes,
    size_cutoff: usize,
    total_offset: usize,
}

impl<S: Stream<Item = Result<Bytes, hyper::Error>>> StreamReader<S> {
    fn new(source: S, size_cutoff: usize) -> StreamReader<S> {
        StreamReader {
            source,
            buffered: Bytes::from(&[][..]),
            size_cutoff,
            total_offset: 0,
        }
    }
}

impl<S: Stream<Item = Result<Bytes, hyper::Error>> + Unpin> io::Read for StreamReader<S> {
    fn read(&mut self, mut buf: &mut [u8]) -> io::Result<usize> {
        if self.total_offset > self.size_cutoff {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                error::request_too_large(self.size_cutoff),
            ));
        }

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
                self.total_offset += buffer.len();
                buf.write(&buffer[..])
            } else {
                self.buffered = Bytes::copy_from_slice(&buffer[buf.len()..]);
                self.total_offset += buf.len();
                buf.write(&buffer[..buf.len()])
            }
        }
    }
}

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
        mut request: Request<Body>,
    ) -> TCResult<TCStream<TCResult<Bytes>>> {
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
                url::form_urlencoded::parse(v.as_bytes())
                    .into_owned()
                    .collect()
            })
            .unwrap_or_else(HashMap::new);

        let txn_id: Option<TxnId> = Http::get_param(&mut params, "txn_id")?;

        match request.method() {
            &Method::GET => {
                let id = Http::get_param(&mut params, "key")?
                    .ok_or_else(|| error::bad_request("Missing URI parameter", "'key'"))?;
                let state = self
                    .gateway
                    .get(&path.clone().into(), id, &token, txn_id.clone())
                    .await?;
                let start_delimiter: TCStream<TCResult<Bytes>> = Box::pin(stream::once(
                    future::ready(Ok(Bytes::copy_from_slice(b"["))),
                ));

                let response: TCStream<TCResult<Bytes>> = Box::pin(match state {
                    State::Value(value) => {
                        stream::once(future::ready(match serde_json::to_string_pretty(&value) {
                            Ok(s) => Ok(Bytes::from(format!("{},", s))),
                            Err(cause) => Err(cause.into()),
                        }))
                    }
                    _ => stream::once(future::ready(Err(error::not_implemented()))),
                });

                let end_delimiter: TCStream<TCResult<Bytes>> = Box::pin(stream::once(
                    future::ready(Ok(Bytes::copy_from_slice(b"]"))),
                ));

                Ok(Box::pin(
                    start_delimiter.chain(response).chain(end_delimiter),
                ))
            }
            &Method::PUT => {
                let reader = StreamReader::new(request.body_mut(), self.request_limit);
                let reader = BufReader::new(reader);
                for op in serde_json::from_reader(reader).into_iter() {
                    let (_selector, _state): (Value, Value) = op;
                    todo!()
                }

                Ok(Box::pin(stream::empty()))
            }
            &Method::POST => Err(error::not_implemented()),
            other => Err(error::method_not_allowed(format!(
                "Tinychain does not support {}",
                other
            ))),
        }
    }
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
