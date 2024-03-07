use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use http_body_util::combinators::UnsyncBoxBody;
use http_body_util::{BodyStream, StreamBody};
use hyper::body::{Body, Bytes, Frame};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use log::{info, trace, warn};
use serde::de::DeserializeOwned;
use tokio::net::TcpListener;

use tc_error::*;
use tc_server::{IntoView, State, Transaction, Txn, TxnId};
use tcgeneric::{NetworkTime, TCPathBuf};

use super::{Accept, Encoding};

type GetParams = HashMap<String, String>;

/// An HTTP server
pub struct Server {
    server: tc_server::Server,
    request_ttl: Duration,
}

impl Server {
    pub fn new(server: tc_server::Server, request_ttl: Duration) -> Self {
        Self {
            server,
            request_ttl,
        }
    }

    async fn process_headers(
        &self,
        http_request: &hyper::Request<impl Body>,
    ) -> TCResult<(GetParams, TxnId, Option<String>, Encoding, Encoding)> {
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

        let txn_id = if let Some(txn_id) = params.remove("txn_id") {
            txn_id.parse()?
        } else {
            TxnId::new(NetworkTime::now())
        };

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

        Ok((params, txn_id, token, accept_encoding, content_type))
    }

    async fn route<B>(
        &self,
        encoding: Encoding,
        txn_id: TxnId,
        token: Option<String>,
        mut params: GetParams,
        http_request: hyper::Request<B>,
    ) -> TCResult<(Txn, State)>
    where
        B: Body<Data = Bytes> + Send + Unpin,
        B::Error: std::error::Error,
    {
        let path: TCPathBuf = http_request.uri().path().parse()?;

        let txn = self.server.get_txn(txn_id, token).await?;

        let endpoint = self.server.authorize_claim_and_route(&path, &txn)?;

        let state = match http_request.method() {
            &hyper::Method::GET => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                endpoint.get(key)?.await
            }

            &hyper::Method::PUT => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                let sub_txn = txn.subcontext_unique();
                let value = destream_body(http_request.into_body(), encoding, sub_txn).await?;
                endpoint.put(key, value)?.map_ok(State::from).await
            }

            &hyper::Method::POST => {
                let sub_txn = txn.subcontext_unique();
                let data = destream_body(http_request.into_body(), encoding, sub_txn).await?;
                let params = data.try_into()?;
                endpoint.post(params)?.await
            }

            &hyper::Method::DELETE => {
                let key = get_param(&mut params, "key")?.unwrap_or_default();
                endpoint.delete(key)?.map_ok(State::from).await
            }

            other => {
                std::mem::drop(endpoint);
                Err(TCError::method_not_allowed(other, &path))
            }
        }?;

        Ok((txn, state))
    }

    async fn handle<B>(
        self: Arc<Self>,
        request: hyper::Request<B>,
    ) -> Result<hyper::Response<UnsyncBoxBody<Bytes, TCError>>, hyper::Error>
    where
        B: Body<Data = Bytes> + Send + Unpin,
        B::Error: std::error::Error,
    {
        let (params, txn_id, token, accept_encoding, request_encoding) =
            match self.process_headers(&request).await {
                Ok(header_data) => header_data,
                Err(cause) => return Ok(transform_error(cause, Encoding::default())),
            };

        trace!(
            "headers check out, routing request to {}...",
            request.uri().path()
        );

        let (txn, state) = match self
            .route(request_encoding, txn_id, token, params, request)
            .await
        {
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

    async fn handle_timeout<B>(
        self: Arc<Self>,
        request: hyper::Request<B>,
    ) -> Result<hyper::Response<UnsyncBoxBody<Bytes, TCError>>, hyper::Error>
    where
        B: Body<Data = Bytes> + Send + Unpin,
        B::Error: std::error::Error,
    {
        match tokio::time::timeout(self.request_ttl, self.handle(request)).await {
            Ok(result) => result,
            Err(cause) => Ok(transform_error(
                timeout!("request timed out").consume(cause),
                Encoding::default(),
            )),
        }
    }

    /// Listen for incoming HTTP requests on the given `port`.
    pub async fn listen(self, port: u16) -> TCResult<()> {
        let addr = SocketAddr::new(Ipv4Addr::UNSPECIFIED.into(), port);
        let listener = TcpListener::bind(addr)
            .map_err(|cause| internal!("could not bind to Ipv4 interface: {cause}"))
            .await?;

        let server = Arc::new(self);

        info!("HTTP server listening on port {}...", port);

        loop {
            let (stream, _) = listener
                .accept()
                .map_err(|cause| internal!("could not accept TCP connection: {cause}"))
                .await?;

            let io = TokioIo::new(stream);
            let server = server.clone();
            let service = service_fn(move |request| server.clone().handle_timeout(request));

            tokio::task::spawn(async move {
                if let Err(err) = http1::Builder::new().serve_connection(io, service).await {
                    warn!("HTTP connection error: {:?}", err);
                }
            });
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

fn delimiter<E>(content: &'static [u8]) -> impl Stream<Item = Result<Bytes, E>> {
    stream::once(future::ready(Ok(Bytes::from_static(content))))
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
