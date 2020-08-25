use std::collections::HashMap;
use std::convert::{Infallible, TryFrom};
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
use crate::transaction::TxnId;
use crate::value::link::*;
use crate::value::string::TCString;
use crate::value::{Value, ValueId};

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

                match state {
                    State::Value(value) => {
                        let value = serde_json::to_string_pretty(&value)
                            .map(Bytes::from)
                            .map_err(error::TCError::from);
                        Ok(Box::pin(stream::once(future::ready(value))))
                    }
                    _other => Ok(Box::pin(stream::once(future::ready(Err(
                        error::not_implemented(),
                    ))))),
                }
            }
            &Method::PUT => Err(error::not_implemented()),
            &Method::POST => {
                println!("POST {}", path);
                let values = String::from_utf8(hyper::body::to_bytes(request).await?.to_vec())
                    .map_err(|e| error::bad_request("Unable to parse request body", e))?;
                let values: HashMap<ValueId, Value> =
                    serde_json::from_str(&values).map_err(|e| {
                        error::bad_request(
                            &format!("Deserialization error {} when parsing", e),
                            values,
                        )
                    })?;

                let capture: Option<Value> = Http::get_param(&mut params, "txn_id")?;
                let capture: Vec<ValueId> = match capture {
                    Some(Value::TCString(TCString::Id(id))) => vec![id],
                    Some(Value::Tuple(ids)) => ids
                        .into_iter()
                        .map(ValueId::try_from)
                        .collect::<TCResult<Vec<ValueId>>>()?,
                    None => vec![],
                    Some(other) => {
                        return Err(error::bad_request(
                            "Expected a list of ValueIds, found",
                            other,
                        ))
                    }
                };
                println!(
                    "capture {}",
                    capture
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                );

                let state = self
                    .gateway
                    .clone()
                    .post(
                        &path.clone().into(),
                        stream::iter(values.into_iter()),
                        &token,
                        txn_id,
                    )
                    .await?;

                println!("post context has {} states", state.len());

                Ok(Box::pin(stream::empty()))
            }
            other => Err(error::method_not_allowed(format!(
                "Tinychain does not support {}",
                other
            ))),
        }
    }
}

fn response_stream<S: Stream<Item = Value> + Send + Sync + Unpin + 'static>(
    s: S,
) -> TCResult<TCStream<TCResult<Bytes>>> {
    let start_delimiter: TCStream<TCResult<Bytes>> = Box::pin(stream::once(future::ready(Ok(
        Bytes::copy_from_slice(b"["),
    ))));

    let response = s
        .map(|v| serde_json::to_string_pretty(&v))
        .map_ok(Bytes::from)
        .map_err(error::TCError::from);
    let response: TCStream<TCResult<Bytes>> = Box::pin(response);

    let end_delimiter: TCStream<TCResult<Bytes>> = Box::pin(stream::once(future::ready(Ok(
        Bytes::copy_from_slice(b"]"),
    ))));

    Ok(Box::pin(
        start_delimiter.chain(response).chain(end_delimiter),
    ))
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
