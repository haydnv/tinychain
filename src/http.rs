use std::collections::HashMap;
use std::convert::Infallible;
use std::io::{self, BufReader, Write};
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{BufMut, Bytes, BytesMut};
use futures::executor::block_on;
use futures::stream::{self, Stream};
use futures_util::stream::StreamExt;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::de::DeserializeOwned;

use crate::auth::Token;
use crate::error;
use crate::gateway::{Gateway, Protocol};
use crate::internal::Dir;
use crate::transaction::TxnId;
use crate::value::link::*;
use crate::value::{TCResult, Value};

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
                error::bad_request(
                    "This request exceeds the maximum allowed size",
                    self.size_cutoff,
                ),
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
    workspace: Arc<Dir>,
    request_limit: usize,
}

impl Http {
    fn new(
        address: SocketAddr,
        gateway: Arc<Gateway>,
        workspace: Arc<Dir>,
        request_limit: usize,
    ) -> Arc<Http> {
        Arc::new(Http {
            address,
            gateway,
            workspace,
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

    async fn handle(
        self: Arc<Self>,
        request: Request<Body>,
    ) -> Result<Response<Body>, hyper::Error> {
        match self.authenticate_and_route(request).await {
            Err(cause) => Ok(Http::transform_error(cause)),
            Ok(()) => Ok(Response::new(Body::from(""))),
        }
    }

    async fn authenticate_and_route(self: Arc<Self>, mut request: Request<Body>) -> TCResult<()> {
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

        let _txn_id: Option<TxnId> = Http::get_param(&mut params, "txn_id")?;

        // TODO: stream the response back to the client
        match request.method() {
            &Method::GET => {
                let id = Http::get_param(&mut params, "key")?
                    .ok_or_else(|| error::bad_request("Missing URI parameter", "'key'"))?;
                let mut data = self.gateway.get(&path.clone().into(), id, &token).await?;
                while let Some(_state) = data.next().await {
                    // TODO: serialize & write to output stream
                }
                Ok(())
            }
            &Method::PUT => {
                let reader = StreamReader::new(request.body_mut(), self.request_limit);
                let reader = BufReader::new(reader);
                for op in serde_json::from_reader(reader).into_iter() {
                    let (selector, state): (Value, Value) = op;
                    self.gateway
                        .put(&path.clone().into(), selector, state.into(), &token)
                        .await?;
                }
                Ok(())
            }
            &Method::POST => {
                let reader = StreamReader::new(request.body_mut(), self.request_limit);
                let reader = BufReader::new(reader);
                let op = stream::iter(serde_json::from_reader(reader).into_iter());
                self.gateway.post(&path.clone().into(), op, &token).await?;
                Ok(())
            }
            other => Err(error::bad_request(
                "Tinychain does not support this HTTP method",
                other,
            )),
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
