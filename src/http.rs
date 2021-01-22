use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response};
use log::debug;
use serde::de::DeserializeOwned;

use error::*;
use generic::NetworkTime;

use crate::state::State;
use crate::txn::TxnId;

const CONTENT_TYPE: &str = "application/json";

pub struct HTTPServer {
    addr: SocketAddr,
}

impl HTTPServer {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    async fn handle(request: Request<Body>) -> Result<Response<Body>, hyper::Error> {
        match Self::route(request) {
            Ok(state) => match destream_json::encode(state) {
                Ok(response) => {
                    let mut response = Response::new(Body::wrap_stream(response));
                    response
                        .headers_mut()
                        .insert(hyper::header::CONTENT_TYPE, CONTENT_TYPE.parse().unwrap());

                    Ok(response)
                }
                Err(cause) => Ok(transform_error(TCError::internal(cause))),
            },
            Err(cause) => Ok(transform_error(cause)),
        }
    }

    fn route(request: hyper::Request<Body>) -> TCResult<State> {
        let mut params = request
            .uri()
            .query()
            .map(|v| {
                debug!("param {}", v);
                url::form_urlencoded::parse(v.as_bytes())
                    .into_owned()
                    .collect()
            })
            .unwrap_or_else(HashMap::new);

        let _txn_id = if let Some(txn_id) = get_param(&mut params, "txn_id")? {
            txn_id
        } else {
            TxnId::new(NetworkTime::now())
        };

        Ok(State::Scalar(scalar::Scalar::Value(scalar::Value::String(
            "Hello, world!".into(),
        ))))
    }
}

#[async_trait]
impl super::Server for HTTPServer {
    type Error = hyper::Error;

    async fn listen(self) -> Result<(), Self::Error> {
        println!("HTTP server listening on {}", self.addr);
        let this = Arc::new(self);

        hyper::Server::bind(&this.addr)
            .serve(make_service_fn(|_| async {
                Ok::<_, hyper::Error>(service_fn(HTTPServer::handle))
            }))
            .await
    }
}

fn get_param<T: DeserializeOwned>(
    params: &mut HashMap<String, String>,
    name: &str,
) -> TCResult<Option<T>> {
    if let Some(param) = params.remove(name) {
        let val: T = serde_json::from_str(&param).map_err(|e| {
            TCError::bad_request(&format!("Unable to parse URI parameter '{}'", name), e)
        })?;

        Ok(Some(val))
    } else {
        Ok(None)
    }
}

fn transform_error(err: TCError) -> hyper::Response<Body> {
    let mut response = hyper::Response::new(Body::from(format!("{}\r\n", err.message())));

    use error::ErrorType::*;
    use hyper::StatusCode;
    *response.status_mut() = match err.code() {
        BadRequest => StatusCode::BAD_REQUEST,
        Forbidden => StatusCode::FORBIDDEN,
        Conflict => StatusCode::CONFLICT,
        Internal => StatusCode::INTERNAL_SERVER_ERROR,
        MethodNotAllowed => StatusCode::METHOD_NOT_ALLOWED,
        NotFound => StatusCode::NOT_FOUND,
        Timeout => StatusCode::REQUEST_TIMEOUT,
        Unauthorized => StatusCode::UNAUTHORIZED,
    };

    response
}
