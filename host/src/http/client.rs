use std::time::Duration;

use async_trait::async_trait;
use futures::{StreamExt, TryFutureExt, TryStreamExt};
use hyper::body::{Body, HttpBody};
use hyper::client::HttpConnector;
use log::debug;
use url::Url;

use tc_error::*;
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::label;

use crate::gateway::ToUrl;
use crate::http::Encoding;
use crate::state::State;
use crate::txn::Txn;

const IDLE_TIMEOUT: u64 = 30;
const ERR_NO_OWNER: &str = "an ownerless transaction may not make outgoing requests";

/// A TinyChain HTTP client. Should only be used through a `Gateway`.
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
    async fn fetch<T>(&self, txn_id: &TxnId, link: ToUrl<'_>, key: &Value) -> TCResult<T>
    where
        T: destream::FromStream<Context = ()>,
    {
        let uri = build_url(&link, txn_id, key)?;

        debug!("FETCH {}", uri);

        let req = req_builder("GET", uri, None);

        let response = self
            .client
            .request(req.body(Body::empty()).unwrap())
            .map_err(|cause| bad_gateway!("error from host at {}", link).consume(cause))
            .await?;

        if response.status().is_success() {
            let body = response.into_body();

            tbon::de::try_decode((), body)
                .map_err(|cause| {
                    bad_gateway!("error decoding response from {}", link).consume(cause)
                })
                .await
        } else {
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }

    async fn get(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<State> {
        if txn.owner().is_none() {
            return Err(bad_request!("{}", ERR_NO_OWNER));
        }

        let uri = build_url(&link, txn.id(), &key)?;
        let req = req_builder("GET", uri, Some(txn.request().token()));

        let txn = txn.subcontext_unique().await?;
        let response = self
            .client
            .request(req.body(Body::empty()).unwrap())
            .map_err(|cause| bad_gateway!("error from host at {}", link).consume(cause))
            .await?;

        if response.status().is_success() {
            tbon::de::try_decode(txn, response.into_body())
                .map_err(|cause| {
                    bad_gateway!("error decoding response from {}", link).consume(cause)
                })
                .await
        } else {
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }

    async fn put(&self, txn: &Txn, link: ToUrl<'_>, key: Value, value: State) -> TCResult<()> {
        if txn.owner().is_none() {
            return Err(bad_request!("{}", ERR_NO_OWNER));
        }

        let uri = build_url(&link, txn.id(), &key)?;
        let req = req_builder("PUT", uri, Some(txn.request().token()))
            .header(hyper::header::CONTENT_TYPE, Encoding::Tbon.as_str());

        let txn = txn.subcontext_unique().await?;
        let view = value.into_view(txn).await?;
        let body = tbon::en::encode(view)
            .map_err(|cause| bad_request!("unable to encode stream").consume(cause))?;

        let body = req
            .body(Body::wrap_stream(body.map_err(|cause| {
                unexpected!("TBON encoding error").consume(cause)
            })))
            .expect("request body");

        let response = self
            .client
            .request(body)
            .map_err(|cause| bad_gateway!("error from host at {}", link).consume(cause))
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }

    async fn post(&self, txn: &Txn, link: ToUrl<'_>, params: State) -> TCResult<State> {
        if txn.owner().is_none() {
            return Err(bad_request!("{}", ERR_NO_OWNER));
        }

        let uri = build_url(&link, txn.id(), &Value::default())?;
        let req = req_builder("POST", uri, Some(txn.request().token()))
            .header(hyper::header::CONTENT_TYPE, Encoding::Tbon.as_str());

        let txn = txn.subcontext_unique().await?;
        let subcontext = txn.subcontext(label("_params").into()).await?;
        let params_view = params.clone().into_view(subcontext).await?;
        let body = tbon::en::encode(params_view)
            .map_err(|cause| bad_request!("unable to encode stream").consume(cause))?;

        let body = req
            .body(Body::wrap_stream(body.map_err(|cause| {
                unexpected!("TBON encoding error").consume(cause)
            })))
            .expect("request body");

        let response = self
            .client
            .request(body)
            .map_err(|cause| bad_gateway!("error from host at {}", link).consume(cause))
            .await?;

        if response.status().is_success() {
            tbon::de::try_decode(txn, response.into_body())
                .map_err(|cause| {
                    bad_gateway!("error decoding response from {}: {:?}", link, params)
                        .consume(cause)
                })
                .await
        } else {
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }

    async fn delete(&self, txn: &Txn, link: ToUrl<'_>, key: Value) -> TCResult<()> {
        if txn.owner().is_none() {
            return Err(bad_request!("{}", ERR_NO_OWNER));
        }

        let uri = build_url(&link, txn.id(), &key)?;
        let req = req_builder("DELETE", uri, Some(txn.request().token()));

        let response = self
            .client
            .request(req.body(Body::empty()).unwrap())
            .map_err(|cause| bad_gateway!("error from host at {}", link).consume(cause))
            .await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let err = transform_error(&link, response).await;
            Err(err)
        }
    }
}

#[inline]
fn build_url(link: &ToUrl<'_>, txn_id: &TxnId, key: &Value) -> TCResult<Url> {
    let mut url = link.to_url();

    url.query_pairs_mut()
        .append_pair("txn_id", &txn_id.to_string());

    if key.is_some() {
        let key_json = serde_json::to_string(&key)
            .map_err(|cause| unexpected!("unable to encode key {}", key).consume(cause))?;

        url.query_pairs_mut().append_pair("key", &key_json);
    }

    Ok(url)
}

fn req_builder(method: &str, url: Url, auth: Option<&str>) -> http::request::Builder {
    let req = hyper::Request::builder()
        .method(method)
        .header(hyper::header::ACCEPT_ENCODING, Encoding::Tbon.as_str())
        .uri(url.to_string());

    if let Some(token) = auth {
        req.header(hyper::header::AUTHORIZATION, format!("Bearer {}", token))
    } else {
        req
    }
}

async fn transform_error(source: &ToUrl<'_>, response: hyper::Response<Body>) -> TCError {
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
        StatusCode::BAD_REQUEST => ErrorKind::BadRequest,
        StatusCode::CONFLICT => ErrorKind::Conflict,
        StatusCode::FORBIDDEN => ErrorKind::Forbidden,
        StatusCode::INTERNAL_SERVER_ERROR => ErrorKind::Internal,
        StatusCode::GATEWAY_TIMEOUT => ErrorKind::Timeout,
        StatusCode::METHOD_NOT_ALLOWED => ErrorKind::MethodNotAllowed,
        StatusCode::NOT_FOUND => ErrorKind::NotFound,
        StatusCode::NOT_IMPLEMENTED => ErrorKind::NotImplemented,
        StatusCode::UNAUTHORIZED => ErrorKind::Unauthorized,
        StatusCode::REQUEST_TIMEOUT => ErrorKind::Timeout,
        _ => ErrorKind::BadGateway,
    };

    TCError::new(code, message)
}
