use std::time::Duration;

use async_trait::async_trait;
use futures::{StreamExt, TryFutureExt, TryStreamExt};
use hyper::body::{Body, HttpBody};
use hyper::client::HttpConnector;
use log::debug;
use serde::de::DeserializeOwned;
use url::Url;

use tc_error::*;
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::label;

use crate::state::State;
use crate::txn::Txn;

const IDLE_TIMEOUT: u64 = 30;
const ERR_NO_OWNER: &str = "an ownerless transaction may not make outgoing requests";

/// A Tinychain HTTP client. Should only be used through a `Gateway`.
pub struct Client {
    client: hyper::Client<HttpConnector, Body>,
}

impl Client {
    /// Construct a new `Client`.
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
    async fn fetch<T: DeserializeOwned>(
        &self,
        txn_id: &TxnId,
        link: &Link,
        key: &Value,
    ) -> TCResult<T> {
        let uri = url(link, txn_id, key)?;
        debug!("FETCH {}", uri);
        let req = req_builder("GET", uri, None);

        let response = self
            .client
            .request(req.body(Body::empty()).unwrap())
            .map_err(|e| TCError::bad_gateway(e))
            .await?;

        if response.status().is_success() {
            let mut response = response.into_body();
            let mut body = Vec::new();
            while let Some(chunk) = response
                .try_next()
                .map_err(|e| {
                    TCError::bad_request(format!("error decoding response from {}", link), e)
                })
                .await?
            {
                body.extend_from_slice(&chunk);
            }

            serde_json::from_slice(&body).map_err(|e| {
                TCError::bad_request(format!("error decoding response from {}", link), e)
            })
        } else {
            let err = transform_error(link, response).await;
            Err(err)
        }
    }

    async fn get(&self, txn: Txn, link: Link, key: Value) -> TCResult<State> {
        if txn.owner().is_none() {
            return Err(TCError::unsupported(ERR_NO_OWNER));
        }

        let uri = url(&link, txn.id(), &key)?;
        let req = req_builder("GET", uri, Some(txn.request().token()));

        let txn = txn.subcontext_tmp().await?;
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
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }

    async fn put(&self, txn: Txn, link: Link, key: Value, value: State) -> TCResult<()> {
        if txn.owner().is_none() {
            return Err(TCError::unsupported(ERR_NO_OWNER));
        }

        let uri = url(&link, txn.id(), &key)?;
        let req = req_builder("PUT", uri, Some(txn.request().token()));

        let txn = txn.subcontext_tmp().await?;
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
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }

    async fn post(&self, txn: Txn, link: Link, params: State) -> TCResult<State> {
        if txn.owner().is_none() {
            return Err(TCError::unsupported(ERR_NO_OWNER));
        }

        let uri = url(&link, txn.id(), &Value::default())?;
        let req = req_builder("POST", uri, Some(txn.request().token()));

        let txn = txn.subcontext_tmp().await?;
        let subcontext = txn.subcontext(label("_params").into()).await?;
        let body = destream_json::encode(params.into_view(subcontext))
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
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }

    async fn delete(&self, txn: &Txn, link: Link, key: Value) -> TCResult<()> {
        if txn.owner().is_none() {
            return Err(TCError::unsupported(ERR_NO_OWNER));
        }

        let uri = url(&link, txn.id(), &key)?;
        let req = req_builder("DELETE", uri, Some(txn.request().token()));

        let response = self
            .client
            .request(req.body(Body::empty()).unwrap())
            .map_err(|e| TCError::bad_gateway(e))
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }
}

fn url(link: &Link, txn_id: &TxnId, key: &Value) -> TCResult<Url> {
    let mut url =
        Url::parse(&link.to_string()).map_err(|e| TCError::bad_request("invalid URL", e))?;

    url.query_pairs_mut()
        .append_pair("txn_id", &txn_id.to_string());

    if key.is_some() {
        let key_json = serde_json::to_string(&key)
            .map_err(|_| TCError::bad_request("unable to encode key", key))?;

        url.query_pairs_mut().append_pair("key", &key_json);
    }

    Ok(url)
}

fn req_builder(method: &str, url: Url, auth: Option<&str>) -> http::request::Builder {
    let req = hyper::Request::builder()
        .method(method)
        .uri(url.to_string());

    if let Some(token) = auth {
        req.header(hyper::header::AUTHORIZATION, format!("Bearer {}", token))
    } else {
        req
    }
}

async fn transform_error(source: &Link, response: hyper::Response<Body>) -> TCError {
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
