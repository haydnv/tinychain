use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
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
    ) -> TCResult<(GetParams, Option<String>, Encoding, Encoding)> {
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

        Ok((params, token, accept_encoding, content_type))
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

        todo!()
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
