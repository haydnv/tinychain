use std::time::Duration;

use async_trait::async_trait;
use futures::{StreamExt, TryFutureExt, TryStreamExt};
use hyper::body::{Body, HttpBody};
use hyper::client::HttpConnector;

use error::*;
use generic::{label, Map};
use transact::{IntoView, Transaction};
use value::{Link, Value};

use crate::state::State;
use crate::txn::Txn;

const IDLE_TIMEOUT: u64 = 30;

pub struct Client {
    client: hyper::Client<HttpConnector, Body>,
}

impl Client {
    pub fn new() -> Self {
        let client = hyper::Client::builder()
            .pool_idle_timeout(Duration::from_secs(IDLE_TIMEOUT))
            .http2_only(true)
            .build_http();

        Self { client }
    }
}

#[async_trait]
impl crate::gateway::Client for Client {
    async fn get(&self, txn: Txn, link: Link, key: Value, auth: Option<String>) -> TCResult<State> {
        let uri = uri(&link, key)?;
        let req = req_builder("GET", uri, auth);

        let response = self
            .client
            .request(req.body(Body::empty()).unwrap())
            .map_err(|e| TCError::bad_gateway(e))
            .await?;

        if response.status().is_success() {
            destream_json::try_decode(txn, response.into_body().map_ok(|bytes| bytes.to_vec()))
                .map_err(|e| {
                    TCError::bad_request(format!("error decoding response from {}", link), e)
                })
                .await
        } else {
            let err = transform_error(link, response).await;
            Err(err)
        }
    }

    async fn put(
        &self,
        txn: Txn,
        link: Link,
        key: Value,
        value: State,
        auth: Option<String>,
    ) -> TCResult<()> {
        let uri = uri(&link, key)?;
        let req = req_builder("PUT", uri, auth);

        let body = destream_json::encode(value.into_view(txn))
            .map_err(|e| TCError::bad_request("unable to encode stream", e))?;

        let response = self
            .client
            .request(req.body(Body::wrap_stream(body)).unwrap())
            .map_err(|e| TCError::bad_gateway(e))
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let err = transform_error(link, response).await;
            Err(err)
        }
    }

    async fn post(
        &self,
        txn: Txn,
        link: Link,
        params: Map<State>,
        auth: Option<String>,
    ) -> TCResult<State> {
        let req = req_builder("POST", link.to_string(), auth);

        let subcontext = txn.subcontext(label("_params").into()).await?;
        let body = destream_json::encode(State::Map(params).into_view(subcontext))
            .map_err(|e| TCError::bad_request("unable to encode stream", e))?;

        let response = self
            .client
            .request(req.body(Body::wrap_stream(body)).unwrap())
            .map_err(|e| TCError::bad_gateway(e))
            .await?;

        if response.status().is_success() {
            destream_json::try_decode(txn, response.into_body().map_ok(|bytes| bytes.to_vec()))
                .map_err(|e| {
                    TCError::bad_request(format!("error decoding response from {}", link), e)
                })
                .await
        } else {
            let err = transform_error(link, response).await;
            Err(err)
        }
    }

    async fn delete(&self, link: Link, key: Value, auth: Option<String>) -> TCResult<()> {
        let uri = uri(&link, key)?;
        let req = req_builder("GET", uri, auth);

        let response = self
            .client
            .request(req.body(Body::empty()).unwrap())
            .map_err(|e| TCError::bad_gateway(e))
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let err = transform_error(link, response).await;
            Err(err)
        }
    }
}

fn uri(link: &Link, key: Value) -> TCResult<String> {
    if key.is_none() {
        Ok(link.to_string())
    } else {
        let key_json = serde_json::to_string(&key)
            .map_err(|_| TCError::bad_request("unable to encode key", key))?;

        Ok(format!("{}?key={}", link, key_json))
    }
}

fn req_builder(method: &str, uri: String, auth: Option<String>) -> http::request::Builder {
    let req = hyper::Request::builder().method(method).uri(uri);

    if let Some(token) = auth {
        req.header("Authorization: Bearer {}", token)
    } else {
        req
    }
}

async fn transform_error(source: Link, response: hyper::Response<Body>) -> TCError {
    const MAX_ERR_SIZE: usize = 5000;

    let status = response.status();

    let mut body = response.into_body();
    let mut err = Vec::new();
    while !body.is_end_stream() && err.len() < MAX_ERR_SIZE {
        if let Some(Ok(buf)) = body.next().await {
            err.extend(buf);
        } else {
            break;
        }
    }

    let message = if let Ok(message) = String::from_utf8(err) {
        format!("error from upstream host {}: {}", source, message)
    } else {
        format!("error from upstream host {}", source)
    };

    use hyper::StatusCode;
    let code = match status {
        StatusCode::BAD_REQUEST => ErrorType::BadRequest,
        StatusCode::CONFLICT => ErrorType::Conflict,
        StatusCode::FORBIDDEN => ErrorType::Forbidden,
        StatusCode::INTERNAL_SERVER_ERROR => ErrorType::Internal,
        StatusCode::GATEWAY_TIMEOUT => ErrorType::Timeout,
        StatusCode::METHOD_NOT_ALLOWED => ErrorType::MethodNotAllowed,
        StatusCode::NOT_FOUND => ErrorType::NotFound,
        StatusCode::NOT_IMPLEMENTED => ErrorType::NotImplemented,
        StatusCode::UNAUTHORIZED => ErrorType::Unauthorized,
        StatusCode::REQUEST_TIMEOUT => ErrorType::Timeout,
        _ => ErrorType::BadGateway,
    };

    TCError::new(code, message)
}
