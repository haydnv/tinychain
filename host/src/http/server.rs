use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use http_body_util::combinators::UnsyncBoxBody;
use http_body_util::{BodyStream, StreamBody};
use hyper::body::{Body, Bytes, Frame};
use hyper::server::conn::http2;
use hyper::service::service_fn;
use hyper_util::rt::{TokioExecutor, TokioIo};
use log::{info, trace, warn};
use serde::de::DeserializeOwned;
use tokio::net::TcpListener;

use tc_error::*;
use tc_fs::{Gateway as GatewayInstance, Resolver};
use tc_state::State;
use tc_transact::{IntoView, Transaction, TxnId};
use tcgeneric::{NetworkTime, TCPathBuf};

use crate::gateway::Gateway;
use crate::txn::Txn;

use super::{Accept, Encoding};

type GetParams = HashMap<String, String>;

/// TinyChain's HTTP server. Should only be used through a [`Gateway`].
pub struct HTTPServer {
    gateway: Gateway,
}

impl HTTPServer {
    pub fn new(gateway: Gateway) -> Self {
        Self { gateway }
    }

    async fn handle_timeout<B>(
        self: Arc<Self>,
        request: hyper::Request<B>,
    ) -> Result<hyper::Response<UnsyncBoxBody<Bytes, TCError>>, hyper::Error>
    where
        B: Body<Data = Bytes> + Send + Unpin,
        B::Error: std::error::Error,
    {
        match tokio::time::timeout(self.gateway.request_ttl(), self.handle(request)).await {
            Ok(result) => result,
            Err(cause) => Ok(transform_error(
                timeout!("request timed out").consume(cause),
                Encoding::default(),
            )),
        }
    }

    async fn handle<B>(
        self: Arc<Self>,
        request: hyper::Request<B>,
    ) -> Result<hyper::Response<UnsyncBoxBody<Bytes, TCError>>, hyper::Error>
    where
        B: Body<Data = Bytes> + Send + Unpin,
        B::Error: std::error::Error,
    {
        let (params, txn, accept_encoding, request_encoding) =
            match self.process_headers(&request).await {
                Ok(header_data) => header_data,
                Err(cause) => return Ok(transform_error(cause, Encoding::default())),
            };

        trace!(
            "headers check out, routing request to {}...",
            request.uri().path()
        );

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
                Ok(response) => {
                    let response = response
                        .chain(delimiter(b"\n"))
                        .map_ok(Frame::data)
                        .map_err(|cause| internal!("JSON encoding error").consume(cause));

                    UnsyncBoxBody::new(StreamBody::new(response))
                }
                Err(cause) => {
                    return Ok(transform_error(
                        internal!("JSON encoding error").consume(cause),
                        Encoding::Json,
                    ))
                }
            },
            Encoding::Tbon => match tbon::en::encode(view) {
                Ok(response) => {
                    let response = response
                        .map_ok(Frame::data)
                        .map_err(|cause| internal!("TBON encoding error").consume(cause));

                    UnsyncBoxBody::new(StreamBody::new(response))
                }
                Err(cause) => {
                    return Ok(transform_error(
                        internal!("TBON encoding error").consume(cause),
                        Encoding::Tbon,
                    ))
                }
            },
        };

        let mut response = hyper::Response::new(body);

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
        http_request: &hyper::Request<impl Body>,
    ) -> TCResult<(GetParams, Txn, Encoding, Encoding)> {
        trace!("reading headers");

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

        let now = NetworkTime::now();
        let txn_id = if let Some(txn_id) = params.remove("txn_id") {
            txn_id.parse()?
        } else {
            TxnId::new(now)
        };

        let token = if let Some(token) = token {
            let resolver = Resolver::new(&self.gateway, &txn_id);
            let token = rjwt::Resolve::verify(&resolver, token, now.into()).await?;
            Some(token)
        } else {
            None
        };

        trace!("token is {token:?}, creating new txn...");

        let txn = self.gateway.clone().new_txn(txn_id, token)?;

        trace!("transaction {txn_id} created");

        Ok((params, txn, accept_encoding, content_type))
    }

    async fn route<B>(
        &self,
        encoding: Encoding,
        txn: &Txn,
        mut params: GetParams,
        http_request: hyper::Request<B>,
    ) -> TCResult<State>
    where
        B: Body<Data = Bytes> + Send + Unpin,
        B::Error: std::error::Error,
    {
        let path: TCPathBuf = http_request.uri().path().parse()?;

        match http_request.method() {
            &hyper::Method::GET => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                self.gateway.get(txn, path.into(), key).await
            }

            &hyper::Method::PUT => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                let sub_txn = txn.subcontext_unique();
                let value = destream_body(http_request.into_body(), encoding, sub_txn).await?;
                let result = self
                    .gateway
                    .put(txn, path.into(), key, value)
                    .map_ok(State::from)
                    .await;

                #[cfg(debug_assertions)]
                trace!("PUT request completed with result {:?}", result);

                result
            }

            &hyper::Method::POST => {
                let sub_txn = txn.subcontext_unique();
                let data = destream_body(http_request.into_body(), encoding, sub_txn).await?;
                self.gateway.post(txn, path.into(), data).await
            }

            &hyper::Method::DELETE => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();

                self.gateway
                    .delete(txn, path.into(), key)
                    .map_ok(State::from)
                    .await
            }

            other => Err(TCError::method_not_allowed(other, "HTTP server", path)),
        }
    }
}

#[async_trait]
impl crate::gateway::Server for HTTPServer {
    async fn listen(self, port: u16) -> TCResult<()> {
        let addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), port);
        let listener = TcpListener::bind(addr)
            .map_err(|cause| internal!("could not bind to Ipv4 interface: {cause}"))
            .await?;

        let server = Arc::new(self);
        let exec = TokioExecutor::new();

        info!("HTTP server listening on port {}...", port);

        loop {
            let (stream, _) = listener
                .accept()
                .map_err(|cause| internal!("could not accept TCP connection: {cause}"))
                .await?;

            let exec = exec.clone();
            let server = server.clone();
            let service = service_fn(move |request| server.clone().handle_timeout(request));

            tokio::task::spawn(async move {
                let io = TokioIo::new(stream);

                if let Err(err) = http2::Builder::new(exec)
                    .serve_connection(io, service)
                    .await
                {
                    warn!("HTTP connection error: {:?}", err);
                }
            });
        }
    }
}

async fn destream_body<B>(body: B, encoding: Encoding, txn: Txn) -> TCResult<State>
where
    B: Body<Data = Bytes> + Send + Unpin,
    B::Error: std::error::Error,
{
    const ERR_DESERIALIZE: &str = "error deserializing HTTP request body";

    let body = BodyStream::new(body).map_ok(|frame| {
        frame
            .into_data()
            .unwrap_or_else(|_err| Bytes::from_static(&[]))
    });

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
        let val: T = serde_json::from_str(&param)
            .map_err(|cause| TCError::unexpected(param, name).consume(cause))?;

        Ok(Some(val))
    } else {
        Ok(None)
    }
}

fn transform_error(
    err: TCError,
    encoding: Encoding,
) -> hyper::Response<UnsyncBoxBody<Bytes, TCError>> {
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
            let encoded = encoded
                .chain(delimiter(b"\n"))
                .map_ok(Frame::data)
                .map_err(|cause| internal!("JSON encoding error").consume(cause));

            UnsyncBoxBody::new(StreamBody::new(encoded))
        }
        Encoding::Tbon => {
            let encoded = tbon::en::encode(err).expect("encode error");
            let encoded = encoded
                .map_ok(Frame::data)
                .map_err(|cause| internal!("TBON encoding error").consume(cause));

            UnsyncBoxBody::new(StreamBody::new(encoded))
        }
    };

    let mut response = hyper::Response::new(body);

    response.headers_mut().insert(
        hyper::header::CONTENT_TYPE,
        encoding.as_str().parse().expect("content encoding"),
    );

    *response.status_mut() = code;

    response
}

fn delimiter<E>(content: &'static [u8]) -> impl Stream<Item = Result<Bytes, E>> {
    stream::once(future::ready(Ok(Bytes::from_static(content))))
}
